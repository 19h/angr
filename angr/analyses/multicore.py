# pylint:disable=import-outside-toplevel
"""
Multicore analysis framework for angr.

This module provides infrastructure for running analyses in parallel across multiple CPU cores.
It includes:
- MulticoreAnalysisMixin: A mixin class that adds multicore capabilities to any analysis
- ParallelAnalysisManager: A manager for running analysis tasks across multiple processes
- Utility functions for common parallel analysis patterns
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import queue
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, Future, as_completed
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from angr.utils.mp import mp_context, Initializer

if TYPE_CHECKING:
    from angr.project import Project
    from angr.knowledge_base import KnowledgeBase

_l = logging.getLogger(name=__name__)

# Type variables for generic parallel analysis
TaskType = TypeVar("TaskType")
ResultType = TypeVar("ResultType")


class ParallelizationMode(Enum):
    """
    Mode of parallelization to use.
    """

    MULTIPROCESSING = auto()  # Use multiple processes (good for CPU-bound tasks, escapes GIL)
    THREADING = auto()  # Use multiple threads (good for I/O-bound tasks or Z3 solving)
    SEQUENTIAL = auto()  # No parallelization (useful for debugging)


@dataclass
class WorkerResult(Generic[ResultType]):
    """
    Result from a worker process/thread.
    """

    task_id: Any
    result: ResultType | None = None
    error: Exception | None = None
    success: bool = True


@dataclass
class ParallelAnalysisConfig:
    """
    Configuration for parallel analysis execution.
    """

    workers: int = 0  # 0 means auto-detect (use cpu_count)
    mode: ParallelizationMode = ParallelizationMode.MULTIPROCESSING
    chunk_size: int = 1  # Number of tasks per worker batch
    timeout: float | None = None  # Timeout per task in seconds
    progress_callback: Callable[[int, int, str | None], None] | None = None  # (completed, total, description)
    fail_fast: bool = False  # Stop on first error
    low_priority: bool = False  # Run at lower priority

    def __post_init__(self):
        if self.workers == 0:
            self.workers = multiprocessing.cpu_count()
        # Ensure at least 1 worker
        self.workers = max(1, self.workers)


class ParallelTaskExecutor(Generic[TaskType, ResultType]):
    """
    Generic parallel task executor that can run tasks across multiple processes or threads.

    This class handles the complexity of managing worker pools, distributing tasks,
    collecting results, and handling errors in a unified way.
    """

    def __init__(
        self,
        config: ParallelAnalysisConfig,
        worker_func: Callable[[TaskType], ResultType],
        initializer: Callable[[], None] | None = None,
        initializer_args: tuple = (),
    ):
        """
        Initialize the parallel task executor.

        :param config: Configuration for parallel execution
        :param worker_func: Function to execute for each task
        :param initializer: Optional function to run in each worker at startup
        :param initializer_args: Arguments to pass to initializer
        """
        self.config = config
        self.worker_func = worker_func
        self.initializer = initializer
        self.initializer_args = initializer_args

        self._completed = 0
        self._total = 0
        self._results: list[WorkerResult[ResultType]] = []
        self._lock = threading.Lock()

    def _update_progress(self, task_id: Any, description: str | None = None):
        """Update progress tracking."""
        with self._lock:
            self._completed += 1
            if self.config.progress_callback:
                self.config.progress_callback(self._completed, self._total, description)

    def _wrap_worker_func(self, task: TaskType) -> WorkerResult[ResultType]:
        """Wrap the worker function to catch exceptions and return WorkerResult."""
        task_id = id(task) if not hasattr(task, "__hash__") else hash(task)
        try:
            result = self.worker_func(task)
            return WorkerResult(task_id=task_id, result=result, success=True)
        except Exception as e:
            _l.error("Worker error processing task %s: %s", task_id, e, exc_info=True)
            return WorkerResult(task_id=task_id, error=e, success=False)

    def execute(self, tasks: Iterable[TaskType]) -> list[WorkerResult[ResultType]]:
        """
        Execute all tasks in parallel and return results.

        :param tasks: Iterable of tasks to execute
        :return: List of WorkerResult objects
        """
        task_list = list(tasks)
        self._total = len(task_list)
        self._completed = 0
        self._results = []

        if not task_list:
            return []

        if self.config.mode == ParallelizationMode.SEQUENTIAL:
            return self._execute_sequential(task_list)
        elif self.config.mode == ParallelizationMode.THREADING:
            return self._execute_threaded(task_list)
        else:
            return self._execute_multiprocess(task_list)

    def _execute_sequential(self, tasks: list[TaskType]) -> list[WorkerResult[ResultType]]:
        """Execute tasks sequentially (for debugging)."""
        results = []
        for i, task in enumerate(tasks):
            result = self._wrap_worker_func(task)
            results.append(result)
            self._update_progress(result.task_id)
            if not result.success and self.config.fail_fast:
                break
        return results

    def _execute_threaded(self, tasks: list[TaskType]) -> list[WorkerResult[ResultType]]:
        """Execute tasks using thread pool."""
        results = []

        with ThreadPoolExecutor(max_workers=self.config.workers) as executor:
            futures: dict[Future, TaskType] = {
                executor.submit(self._wrap_worker_func, task): task for task in tasks
            }

            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                self._update_progress(result.task_id)
                if not result.success and self.config.fail_fast:
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    break

        return results

    def _execute_multiprocess(self, tasks: list[TaskType]) -> list[WorkerResult[ResultType]]:
        """Execute tasks using process pool."""
        results = []
        ctx = mp_context()

        # Create initializer wrapper if needed
        init_func = None
        init_args = ()
        if self.initializer:
            init_func = self.initializer
            init_args = self.initializer_args

        # Create a wrapper closure that can be pickled
        # We store the worker_func reference separately to avoid pickling self
        worker_func = self.worker_func

        with ctx.Pool(
            processes=self.config.workers, initializer=init_func, initargs=init_args
        ) as pool:
            # Use imap_unordered for better load balancing
            # Use a standalone wrapper function to avoid pickling issues with bound methods
            for result in pool.imap_unordered(
                _mp_worker_wrapper, [(worker_func, task) for task in tasks], chunksize=self.config.chunk_size
            ):
                results.append(result)
                self._update_progress(result.task_id)
                if not result.success and self.config.fail_fast:
                    pool.terminate()
                    break

        return results


def _mp_worker_wrapper(args: tuple) -> WorkerResult:
    """
    Standalone wrapper function for multiprocessing that can be pickled.
    Takes (worker_func, task) tuple and returns WorkerResult.
    """
    worker_func, task = args
    task_id = id(task) if not hasattr(task, "__hash__") else hash(task)
    try:
        result = worker_func(task)
        return WorkerResult(task_id=task_id, result=result, success=True)
    except Exception as e:
        _l.error("Worker error processing task %s: %s", task_id, e, exc_info=True)
        return WorkerResult(task_id=task_id, error=e, success=False)


class MulticoreAnalysisMixin:
    """
    Mixin class that adds multicore capabilities to an Analysis.

    Usage:
        class MyParallelAnalysis(MulticoreAnalysisMixin, Analysis):
            def __init__(self, workers=0, ...):
                self._init_multicore(workers=workers)
                # ... rest of init
    """

    _multicore_config: ParallelAnalysisConfig

    def _init_multicore(
        self,
        workers: int = 0,
        mode: ParallelizationMode = ParallelizationMode.MULTIPROCESSING,
        chunk_size: int = 1,
        timeout: float | None = None,
        fail_fast: bool = False,
        low_priority: bool = False,
    ):
        """
        Initialize multicore capabilities.

        :param workers: Number of worker processes/threads (0 = auto-detect)
        :param mode: Parallelization mode (multiprocessing, threading, or sequential)
        :param chunk_size: Number of tasks per batch
        :param timeout: Timeout per task in seconds
        :param fail_fast: Stop on first error
        :param low_priority: Run at lower priority
        """
        # Get progress callback from parent Analysis class if available
        progress_callback = None
        if hasattr(self, "_update_progress"):

            def wrapped_progress(completed: int, total: int, description: str | None):
                percentage = (completed / total * 100.0) if total > 0 else 0
                # Access through self to get the bound method
                getattr(self, "_update_progress")(percentage, text=description)

            progress_callback = wrapped_progress

        self._multicore_config = ParallelAnalysisConfig(
            workers=workers,
            mode=mode,
            chunk_size=chunk_size,
            timeout=timeout,
            progress_callback=progress_callback,
            fail_fast=fail_fast,
            low_priority=low_priority,
        )

    def _execute_parallel(
        self,
        tasks: Iterable[TaskType],
        worker_func: Callable[[TaskType], ResultType],
        initializer: Callable[[], None] | None = None,
        initializer_args: tuple = (),
    ) -> list[WorkerResult[ResultType]]:
        """
        Execute tasks in parallel using configured settings.

        :param tasks: Iterable of tasks to execute
        :param worker_func: Function to execute for each task
        :param initializer: Optional function to run in each worker at startup
        :param initializer_args: Arguments to pass to initializer
        :return: List of WorkerResult objects
        """
        executor = ParallelTaskExecutor(
            config=self._multicore_config,
            worker_func=worker_func,
            initializer=initializer,
            initializer_args=initializer_args,
        )
        return executor.execute(tasks)


class ParallelFunctionAnalysisBase(MulticoreAnalysisMixin, ABC):
    """
    Base class for analyses that operate on functions in parallel.

    Subclasses should implement:
    - _analyze_function: The actual analysis logic for a single function
    - _merge_results: How to merge results from all functions
    """

    def __init__(
        self,
        project: Project,
        kb: KnowledgeBase | None = None,
        workers: int = 0,
        mode: ParallelizationMode = ParallelizationMode.MULTIPROCESSING,
        func_addrs: Iterable[int] | None = None,
        skip_alignment: bool = True,
        skip_simprocedures: bool = True,
        max_function_size: int | None = None,
        max_function_blocks: int | None = None,
    ):
        """
        Initialize parallel function analysis.

        :param project: angr Project
        :param kb: KnowledgeBase (default: project.kb)
        :param workers: Number of workers (0 = auto-detect)
        :param mode: Parallelization mode
        :param func_addrs: Specific function addresses to analyze (None = all functions)
        :param skip_alignment: Skip alignment functions
        :param skip_simprocedures: Skip SimProcedure functions
        :param max_function_size: Maximum function size to analyze
        :param max_function_blocks: Maximum number of blocks in function to analyze
        """
        self.project = project
        self.kb = kb or project.kb
        self._skip_alignment = skip_alignment
        self._skip_simprocedures = skip_simprocedures
        self._max_function_size = max_function_size
        self._max_function_blocks = max_function_blocks

        self._init_multicore(workers=workers, mode=mode)

        # Determine functions to analyze
        if func_addrs is not None:
            self._func_addrs = list(func_addrs)
        else:
            self._func_addrs = self._get_analyzable_functions()

        self.results: dict[int, Any] = {}
        self.errors: dict[int, Exception] = {}

    def _get_analyzable_functions(self) -> list[int]:
        """Get list of function addresses that should be analyzed."""
        func_addrs = []

        for func in self.kb.functions.values():
            # Skip alignment functions
            if self._skip_alignment and func.is_alignment:
                continue

            # Skip SimProcedures
            if self._skip_simprocedures and func.is_simprocedure:
                continue

            # Check function size
            if self._max_function_size is not None:
                func_size = sum(block.size for block in func.blocks if block.size is not None)
                if func_size > self._max_function_size:
                    _l.debug("Skipping %r: size %d > %d", func, func_size, self._max_function_size)
                    continue

            # Check block count
            if self._max_function_blocks is not None:
                if len(func.block_addrs_set) > self._max_function_blocks:
                    _l.debug(
                        "Skipping %r: %d blocks > %d",
                        func,
                        len(func.block_addrs_set),
                        self._max_function_blocks,
                    )
                    continue

            func_addrs.append(func.addr)

        return func_addrs

    @abstractmethod
    def _analyze_function(self, func_addr: int) -> Any:
        """
        Analyze a single function. Must be implemented by subclasses.

        This method should be safe to call from a worker process.

        :param func_addr: Address of function to analyze
        :return: Analysis result for this function
        """
        raise NotImplementedError()

    def _merge_results(self, results: list[WorkerResult]) -> None:
        """
        Merge results from parallel analysis into self.results and self.errors.

        Override this method to customize result merging.

        :param results: List of WorkerResult objects
        """
        for worker_result in results:
            if worker_result.success:
                self.results[worker_result.task_id] = worker_result.result
            else:
                self.errors[worker_result.task_id] = worker_result.error

    def analyze(self) -> None:
        """Run the parallel analysis."""
        if not self._func_addrs:
            _l.info("No functions to analyze")
            return

        _l.info(
            "Starting parallel analysis of %d functions with %d workers",
            len(self._func_addrs),
            self._multicore_config.workers,
        )

        results = self._execute_parallel(
            tasks=self._func_addrs,
            worker_func=self._analyze_function,
            initializer=_worker_initializer,
            initializer_args=(),
        )

        self._merge_results(results)

        _l.info(
            "Parallel analysis complete: %d succeeded, %d failed",
            len(self.results),
            len(self.errors),
        )


def _worker_initializer():
    """
    Initialize worker process.
    Calls all registered initializers from angr.utils.mp.Initializer.
    """
    Initializer.get().initialize()


def run_parallel_analysis(
    project: Project,
    func_addrs: Iterable[int],
    analysis_func: Callable[[Project, int], Any],
    workers: int = 0,
    mode: ParallelizationMode = ParallelizationMode.MULTIPROCESSING,
    progress_callback: Callable[[int, int, str | None], None] | None = None,
) -> dict[int, Any]:
    """
    Convenience function to run an analysis function on multiple functions in parallel.

    :param project: angr Project
    :param func_addrs: Iterable of function addresses to analyze
    :param analysis_func: Function that takes (project, func_addr) and returns analysis result
    :param workers: Number of workers (0 = auto-detect)
    :param mode: Parallelization mode
    :param progress_callback: Optional callback for progress updates
    :return: Dictionary mapping func_addr to analysis result
    """
    config = ParallelAnalysisConfig(
        workers=workers, mode=mode, progress_callback=progress_callback
    )

    # Create a wrapper that captures project
    def worker_func(func_addr: int) -> Any:
        return analysis_func(project, func_addr)

    executor = ParallelTaskExecutor(
        config=config,
        worker_func=worker_func,
        initializer=_worker_initializer,
    )

    results = executor.execute(func_addrs)

    return {r.task_id: r.result for r in results if r.success}


class ParallelAnalysisManager:
    """
    Manager for coordinating multiple parallel analyses.

    This class provides a higher-level interface for running analyses
    that may have dependencies between them.
    """

    def __init__(self, project: Project, workers: int = 0):
        """
        Initialize the parallel analysis manager.

        :param project: angr Project
        :param workers: Number of workers (0 = auto-detect)
        """
        self.project = project
        self.workers = workers if workers > 0 else multiprocessing.cpu_count()
        self._results: dict[str, Any] = {}

    def run_analysis(
        self,
        name: str,
        analysis_class: type,
        *args,
        workers: int | None = None,
        **kwargs,
    ) -> Any:
        """
        Run an analysis and store its result.

        :param name: Name to store result under
        :param analysis_class: Analysis class to instantiate
        :param args: Arguments for analysis
        :param workers: Override worker count for this analysis
        :param kwargs: Keyword arguments for analysis
        :return: Analysis result
        """
        effective_workers = workers if workers is not None else self.workers

        # If analysis supports workers parameter, pass it
        if "workers" not in kwargs:
            kwargs["workers"] = effective_workers

        result = self.project.analyses[analysis_class](*args, **kwargs)
        self._results[name] = result
        return result

    def get_result(self, name: str) -> Any:
        """Get stored analysis result by name."""
        return self._results.get(name)

    @property
    def results(self) -> dict[str, Any]:
        """Get all stored results."""
        return self._results.copy()
