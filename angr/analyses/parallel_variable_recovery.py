# pylint:disable=import-outside-toplevel
"""
Parallel variable recovery analysis for angr.

This module provides parallel variable recovery capabilities, allowing variable recovery
to be performed on multiple functions simultaneously across multiple CPU cores.
"""

from __future__ import annotations

import logging
import pickle
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from angr.utils.mp import mp_context, Initializer
from . import Analysis, register_analysis
from .multicore import (
    MulticoreAnalysisMixin,
    ParallelizationMode,
)

if TYPE_CHECKING:
    from angr.knowledge_base import KnowledgeBase
    from angr.project import Project
    from angr.knowledge_plugins.variables.variable_manager import VariableManagerInternal

_l = logging.getLogger(name=__name__)


# Global variables for worker processes
_worker_project: Project | None = None


def _init_worker(project_pickle: bytes) -> None:
    """
    Initialize worker process with project.
    """
    global _worker_project
    Initializer.get().initialize()
    _worker_project = pickle.loads(project_pickle)


def _recover_variables(task: dict[str, Any]) -> dict[str, Any]:
    """
    Recover variables for a single function in a worker process.

    :param task: Dictionary containing func_addr and recovery options
    :return: Dictionary with recovery results
    """
    from .variable_recovery import VariableRecoveryFast

    global _worker_project

    func_addr = task["func_addr"]
    options = task.get("options", {})

    result = {
        "func_addr": func_addr,
        "success": False,
        "variable_manager": None,
        "variables_count": 0,
        "error": None,
    }

    try:
        assert _worker_project is not None, "Worker project not initialized"

        # Get function from project
        func = _worker_project.kb.functions.get_by_addr(func_addr)
        if func is None:
            result["error"] = f"Function at {func_addr:#x} not found"
            return result

        # Run variable recovery with configured options
        low_priority = options.pop("low_priority", False)
        func_graph = options.pop("func_graph", None)

        vr = _worker_project.analyses[VariableRecoveryFast].prep(
            kb=_worker_project.kb,
        )(
            func,
            low_priority=low_priority,
            func_graph=func_graph,
            **options,
        )

        result["success"] = True

        # Get variable count
        var_manager = _worker_project.kb.variables.get_function_manager(func_addr)
        if var_manager:
            result["variables_count"] = len(list(var_manager.get_variables()))
            # Serialize the variable manager for transfer
            result["variable_manager"] = pickle.dumps(var_manager)

    except Exception as e:
        _l.error("Failed to recover variables for function at %#x: %s", func_addr, e, exc_info=True)
        result["error"] = str(e)

    return result


class ParallelVariableRecovery(MulticoreAnalysisMixin, Analysis):
    """
    Parallel variable recovery analysis.

    This analysis performs variable recovery on multiple functions in parallel
    across multiple CPU cores.

    Example usage:
        >>> cfg = proj.analyses.CFGFast()
        >>> parallel_vr = proj.analyses.ParallelVariableRecovery(
        ...     workers=4,
        ...     func_addrs=[0x401000, 0x401100, 0x401200]
        ... )
        >>> print(f"Recovered variables for {parallel_vr.recovered_count} functions")
    """

    def __init__(
        self,
        workers: int = 0,
        mode: ParallelizationMode = ParallelizationMode.MULTIPROCESSING,
        func_addrs: Iterable[int] | None = None,
        skip_alignment: bool = True,
        skip_simprocedures: bool = True,
        skip_plt: bool = True,
        max_function_size: int | None = None,
        max_function_blocks: int | None = None,
        low_priority: bool = False,
        recovery_options: dict[str, Any] | None = None,
    ):
        """
        Initialize parallel variable recovery.

        :param workers: Number of worker processes (0 = auto-detect CPU count)
        :param mode: Parallelization mode (MULTIPROCESSING recommended)
        :param func_addrs: Specific function addresses to analyze (None = all functions)
        :param skip_alignment: Skip alignment functions
        :param skip_simprocedures: Skip SimProcedure functions
        :param skip_plt: Skip PLT stub functions
        :param max_function_size: Maximum function size to analyze (bytes)
        :param max_function_blocks: Maximum number of blocks in function
        :param low_priority: Run at lower priority (release GIL periodically)
        :param recovery_options: Additional options to pass to each VariableRecoveryFast instance
        """
        # Initialize multicore capabilities
        self._init_multicore(workers=workers, mode=mode)

        # Store configuration
        self._skip_alignment = skip_alignment
        self._skip_simprocedures = skip_simprocedures
        self._skip_plt = skip_plt
        self._max_function_size = max_function_size
        self._max_function_blocks = max_function_blocks
        self._low_priority = low_priority
        self._recovery_options = recovery_options or {}

        # Results storage
        self.results: dict[int, int] = {}  # func_addr -> variables_count
        self.errors: dict[int, str] = {}  # func_addr -> error message
        self.recovered_count: int = 0
        self.failed_count: int = 0

        # Get functions to analyze
        if func_addrs is not None:
            self._func_addrs = list(func_addrs)
        else:
            self._func_addrs = self._get_analyzable_functions()

        # Run the analysis
        self._analyze()

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

            # Skip PLT stubs
            if self._skip_plt and func.is_plt:
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

    def _analyze(self) -> None:
        """Run the parallel variable recovery."""
        if not self._func_addrs:
            _l.info("No functions to analyze")
            self._finish_progress()
            return

        _l.info(
            "Starting parallel variable recovery for %d functions with %d workers",
            len(self._func_addrs),
            self._multicore_config.workers,
        )

        # Prepare tasks
        tasks = []
        for func_addr in self._func_addrs:
            task = {
                "func_addr": func_addr,
                "options": {
                    "low_priority": self._low_priority,
                    **self._recovery_options,
                },
            }
            tasks.append(task)

        # Pickle project for workers
        project_pickle = pickle.dumps(self.project)

        # Execute variable recovery in parallel
        if self._multicore_config.mode == ParallelizationMode.SEQUENTIAL:
            # Sequential execution for debugging
            _init_worker(project_pickle)
            for i, task in enumerate(tasks):
                result = _recover_variables(task)
                self._process_result(result)
                percentage = (i + 1) / len(tasks) * 100.0
                func_name = self._get_func_name(task["func_addr"])
                self._update_progress(percentage, text=f"{i + 1}/{len(tasks)} - {func_name}")
        else:
            # Parallel execution
            ctx = mp_context()

            with ctx.Pool(
                processes=self._multicore_config.workers,
                initializer=_init_worker,
                initargs=(project_pickle,),
            ) as pool:
                total = len(tasks)
                for i, result in enumerate(pool.imap_unordered(_recover_variables, tasks)):
                    self._process_result(result)
                    percentage = (i + 1) / total * 100.0
                    func_name = self._get_func_name(result["func_addr"])
                    self._update_progress(percentage, text=f"{i + 1}/{total} - {func_name}")

        self._finish_progress()

        _l.info(
            "Parallel variable recovery complete: %d succeeded, %d failed",
            self.recovered_count,
            self.failed_count,
        )

    def _process_result(self, result: dict[str, Any]) -> None:
        """Process a single variable recovery result."""
        func_addr = result["func_addr"]

        if result["success"]:
            self.results[func_addr] = result["variables_count"]
            self.recovered_count += 1

            # Import variable manager into main KB if available
            if result["variable_manager"]:
                try:
                    var_manager = pickle.loads(result["variable_manager"])
                    self.kb.variables.function_managers[func_addr] = var_manager
                    var_manager.set_manager(self.kb.variables)
                except Exception as e:
                    _l.warning(
                        "Failed to import variable manager for %#x: %s", func_addr, e
                    )
        else:
            self.errors[func_addr] = result["error"] or "Unknown error"
            self.failed_count += 1

    def get_variables_count(self, func_addr: int) -> int | None:
        """
        Get number of variables recovered for a function.

        :param func_addr: Function address
        :return: Number of variables, or None if not available
        """
        return self.results.get(func_addr)

    def get_error(self, func_addr: int) -> str | None:
        """
        Get error message for a failed recovery.

        :param func_addr: Function address
        :return: Error message, or None if no error
        """
        return self.errors.get(func_addr)

    def _get_func_name(self, func_addr: int) -> str:
        """Get function name safely, handling non-existent addresses."""
        try:
            func = self.kb.functions.get_by_addr(func_addr)
            return func.demangled_name if func else f"{func_addr:#x}"
        except KeyError:
            return f"{func_addr:#x}"


register_analysis(ParallelVariableRecovery, "ParallelVariableRecovery")
