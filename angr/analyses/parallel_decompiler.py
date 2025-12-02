# pylint:disable=import-outside-toplevel
"""
Parallel decompiler analysis for angr.

This module provides parallel decompilation capabilities, allowing multiple functions
to be decompiled simultaneously across multiple CPU cores.
"""

from __future__ import annotations

import logging
import pickle
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from angr.utils.mp import mp_context, Initializer
from angr.knowledge_plugins.functions.function import Function
from . import Analysis, register_analysis
from .multicore import (
    MulticoreAnalysisMixin,
    ParallelAnalysisConfig,
    ParallelTaskExecutor,
    ParallelizationMode,
    WorkerResult,
)

if TYPE_CHECKING:
    from angr.knowledge_base import KnowledgeBase
    from angr.project import Project
    from .cfg import CFGFast
    from angr.knowledge_plugins.cfg.cfg_model import CFGModel
    from .decompiler.structured_codegen.c import CStructuredCodeGenerator

_l = logging.getLogger(name=__name__)


# Global variables for worker processes
_worker_project: Project | None = None
_worker_cfg: CFGModel | None = None


def _init_worker(project_pickle: bytes, cfg_pickle: bytes | None) -> None:
    """
    Initialize worker process with project and CFG.
    """
    global _worker_project, _worker_cfg
    Initializer.get().initialize()
    _worker_project = pickle.loads(project_pickle)
    if cfg_pickle is not None:
        _worker_cfg = pickle.loads(cfg_pickle)


def _decompile_function(task: dict[str, Any]) -> dict[str, Any]:
    """
    Decompile a single function in a worker process.

    :param task: Dictionary containing func_addr and decompiler options
    :return: Dictionary with decompilation results
    """
    from angr.analyses.decompiler import Decompiler

    global _worker_project, _worker_cfg

    func_addr = task["func_addr"]
    options = task.get("options", {})

    result = {
        "func_addr": func_addr,
        "success": False,
        "codegen": None,
        "text": None,
        "error": None,
    }

    try:
        assert _worker_project is not None, "Worker project not initialized"

        # Get function from project
        func = _worker_project.kb.functions.get_by_addr(func_addr)
        if func is None:
            result["error"] = f"Function at {func_addr:#x} not found"
            return result

        # Run decompiler with configured options
        cfg = _worker_cfg or options.pop("cfg", None)
        decompiler = _worker_project.analyses[Decompiler].prep(
            kb=_worker_project.kb,
        )(
            func,
            cfg=cfg,
            **options,
        )

        result["success"] = True
        result["text"] = decompiler.codegen.text if decompiler.codegen else None

        # Store in knowledge base if requested
        if options.get("store_in_kb", True) and decompiler.codegen:
            _worker_project.kb.structured_code[(func_addr, "pseudocode")] = decompiler.codegen

    except Exception as e:
        _l.error("Failed to decompile function at %#x: %s", func_addr, e, exc_info=True)
        result["error"] = str(e)

    return result


class ParallelDecompiler(MulticoreAnalysisMixin, Analysis):
    """
    Parallel decompilation analysis.

    This analysis decompiles multiple functions in parallel across multiple CPU cores.

    Example usage:
        >>> cfg = proj.analyses.CFGFast()
        >>> parallel_dec = proj.analyses.ParallelDecompiler(
        ...     cfg=cfg,
        ...     workers=4,
        ...     func_addrs=[0x401000, 0x401100, 0x401200]
        ... )
        >>> for func_addr, text in parallel_dec.results.items():
        ...     print(f"Function at {func_addr:#x}:")
        ...     print(text)
    """

    def __init__(
        self,
        cfg: CFGFast | CFGModel | None = None,
        workers: int = 0,
        mode: ParallelizationMode = ParallelizationMode.MULTIPROCESSING,
        func_addrs: Iterable[int] | None = None,
        skip_alignment: bool = True,
        skip_simprocedures: bool = True,
        skip_plt: bool = True,
        max_function_size: int | None = None,
        max_function_blocks: int | None = None,
        store_in_kb: bool = True,
        decompiler_options: dict[str, Any] | None = None,
    ):
        """
        Initialize parallel decompiler.

        :param cfg: Control flow graph (CFGFast or CFGModel)
        :param workers: Number of worker processes (0 = auto-detect CPU count)
        :param mode: Parallelization mode (MULTIPROCESSING recommended for decompilation)
        :param func_addrs: Specific function addresses to decompile (None = all functions)
        :param skip_alignment: Skip alignment functions
        :param skip_simprocedures: Skip SimProcedure functions
        :param skip_plt: Skip PLT stub functions
        :param max_function_size: Maximum function size to decompile (bytes)
        :param max_function_blocks: Maximum number of blocks in function
        :param store_in_kb: Store decompilation results in KnowledgeBase
        :param decompiler_options: Additional options to pass to each Decompiler instance
        """
        from .cfg import CFGFast

        # Initialize multicore capabilities
        self._init_multicore(workers=workers, mode=mode)

        # Store configuration
        self._cfg: CFGModel | None = cfg.model if isinstance(cfg, CFGFast) else cfg
        self._skip_alignment = skip_alignment
        self._skip_simprocedures = skip_simprocedures
        self._skip_plt = skip_plt
        self._max_function_size = max_function_size
        self._max_function_blocks = max_function_blocks
        self._store_in_kb = store_in_kb
        self._decompiler_options = decompiler_options or {}

        # Results storage
        self.results: dict[int, str] = {}  # func_addr -> decompiled text
        self.errors: dict[int, str] = {}  # func_addr -> error message
        self.decompiled_count: int = 0
        self.failed_count: int = 0

        # Get functions to decompile
        if func_addrs is not None:
            self._func_addrs = list(func_addrs)
        else:
            self._func_addrs = self._get_decompilable_functions()

        # Run the analysis
        self._analyze()

    def _get_decompilable_functions(self) -> list[int]:
        """Get list of function addresses that should be decompiled."""
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
        """Run the parallel decompilation."""
        if not self._func_addrs:
            _l.info("No functions to decompile")
            self._finish_progress()
            return

        _l.info(
            "Starting parallel decompilation of %d functions with %d workers",
            len(self._func_addrs),
            self._multicore_config.workers,
        )

        # Prepare tasks
        tasks = []
        for func_addr in self._func_addrs:
            task = {
                "func_addr": func_addr,
                "options": {
                    "store_in_kb": self._store_in_kb,
                    **self._decompiler_options,
                },
            }
            tasks.append(task)

        # Pickle project and CFG for workers
        project_pickle = pickle.dumps(self.project)
        cfg_pickle = pickle.dumps(self._cfg) if self._cfg else None

        # Execute decompilation in parallel
        if self._multicore_config.mode == ParallelizationMode.SEQUENTIAL:
            # Sequential execution for debugging
            _init_worker(project_pickle, cfg_pickle)
            for i, task in enumerate(tasks):
                result = _decompile_function(task)
                self._process_result(result)
                percentage = (i + 1) / len(tasks) * 100.0
                func = self.kb.functions.get_by_addr(task["func_addr"])
                func_name = func.demangled_name if func else f"{task['func_addr']:#x}"
                self._update_progress(percentage, text=f"{i + 1}/{len(tasks)} - {func_name}")
        else:
            # Parallel execution
            ctx = mp_context()

            with ctx.Pool(
                processes=self._multicore_config.workers,
                initializer=_init_worker,
                initargs=(project_pickle, cfg_pickle),
            ) as pool:
                total = len(tasks)
                for i, result in enumerate(pool.imap_unordered(_decompile_function, tasks)):
                    self._process_result(result)
                    percentage = (i + 1) / total * 100.0
                    func = self.kb.functions.get_by_addr(result["func_addr"])
                    func_name = func.demangled_name if func else f"{result['func_addr']:#x}"
                    self._update_progress(percentage, text=f"{i + 1}/{total} - {func_name}")

        self._finish_progress()

        _l.info(
            "Parallel decompilation complete: %d succeeded, %d failed",
            self.decompiled_count,
            self.failed_count,
        )

    def _process_result(self, result: dict[str, Any]) -> None:
        """Process a single decompilation result."""
        func_addr = result["func_addr"]

        if result["success"]:
            self.results[func_addr] = result["text"]
            self.decompiled_count += 1

            # Store in KB if configured
            if self._store_in_kb and result["text"]:
                # The worker already stored it in its KB, but we need to update ours
                # For now, just store the text result
                pass
        else:
            self.errors[func_addr] = result["error"] or "Unknown error"
            self.failed_count += 1

    def get_decompilation(self, func_addr: int) -> str | None:
        """
        Get decompiled code for a function.

        :param func_addr: Function address
        :return: Decompiled code text, or None if not available
        """
        return self.results.get(func_addr)

    def get_error(self, func_addr: int) -> str | None:
        """
        Get error message for a failed decompilation.

        :param func_addr: Function address
        :return: Error message, or None if no error
        """
        return self.errors.get(func_addr)

    def __iter__(self):
        """Iterate over (func_addr, decompiled_text) pairs."""
        return iter(self.results.items())


register_analysis(ParallelDecompiler, "ParallelDecompiler")
