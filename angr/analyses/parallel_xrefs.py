# pylint:disable=import-outside-toplevel
"""
Parallel cross-reference (XRefs) analysis for angr.

This module provides parallel XRefs analysis capabilities, allowing cross-references
to be computed for multiple functions simultaneously across multiple CPU cores.
"""

from __future__ import annotations

import logging
import pickle
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from angr.utils.mp import mp_context, Initializer
from angr.knowledge_plugins.xrefs import XRef
from . import Analysis, register_analysis
from .multicore import (
    MulticoreAnalysisMixin,
    ParallelizationMode,
)

if TYPE_CHECKING:
    from angr.knowledge_base import KnowledgeBase
    from angr.project import Project

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


def _analyze_xrefs(task: dict[str, Any]) -> dict[str, Any]:
    """
    Analyze cross-references for a single function in a worker process.

    :param task: Dictionary containing func_addr and analysis options
    :return: Dictionary with analysis results
    """
    from .xrefs import XRefsAnalysis

    global _worker_project

    func_addr = task["func_addr"]
    options = task.get("options", {})

    result = {
        "func_addr": func_addr,
        "success": False,
        "xrefs": [],
        "xrefs_count": 0,
        "error": None,
    }

    try:
        assert _worker_project is not None, "Worker project not initialized"

        # Get function from project
        func = _worker_project.kb.functions.get_by_addr(func_addr)
        if func is None:
            result["error"] = f"Function at {func_addr:#x} not found"
            return result

        # Run XRefs analysis
        func_graph = options.pop("func_graph", None)
        max_iterations = options.pop("max_iterations", 1)

        _worker_project.analyses[XRefsAnalysis].prep(kb=_worker_project.kb)(
            func=func,
            func_graph=func_graph,
            max_iterations=max_iterations,
            **options,
        )

        # Collect XRefs for this function
        xrefs_data = []
        for block in func.blocks:
            block_xrefs = _worker_project.kb.xrefs.get_xrefs_by_ins_addr_region(
                block.addr, block.addr + block.size
            )
            for xref in block_xrefs:
                # Serialize XRef data for transfer
                # Note: XRef stores type in .type attribute as an int (XRefType value)
                xref_type = xref.type
                if hasattr(xref_type, 'value'):
                    xref_type = xref_type.value
                elif xref_type is None:
                    xref_type = 0
                xrefs_data.append({
                    "ins_addr": xref.ins_addr,
                    "block_addr": xref.block_addr,
                    "stmt_idx": xref.stmt_idx,
                    "dst": xref.dst,
                    "xref_type": xref_type,
                })

        result["success"] = True
        result["xrefs"] = xrefs_data
        result["xrefs_count"] = len(xrefs_data)

    except Exception as e:
        _l.error("Failed to analyze XRefs for function at %#x: %s", func_addr, e, exc_info=True)
        result["error"] = str(e)

    return result


class ParallelXRefs(MulticoreAnalysisMixin, Analysis):
    """
    Parallel cross-reference analysis.

    This analysis computes cross-references for multiple functions in parallel
    across multiple CPU cores.

    Example usage:
        >>> cfg = proj.analyses.CFGFast()
        >>> parallel_xrefs = proj.analyses.ParallelXRefs(
        ...     workers=4,
        ...     func_addrs=[0x401000, 0x401100, 0x401200]
        ... )
        >>> print(f"Found {parallel_xrefs.total_xrefs_count} cross-references")
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
        max_iterations: int = 1,
        analysis_options: dict[str, Any] | None = None,
    ):
        """
        Initialize parallel XRefs analysis.

        :param workers: Number of worker processes (0 = auto-detect CPU count)
        :param mode: Parallelization mode (MULTIPROCESSING recommended)
        :param func_addrs: Specific function addresses to analyze (None = all functions)
        :param skip_alignment: Skip alignment functions
        :param skip_simprocedures: Skip SimProcedure functions
        :param skip_plt: Skip PLT stub functions
        :param max_function_size: Maximum function size to analyze (bytes)
        :param max_function_blocks: Maximum number of blocks in function
        :param max_iterations: Maximum iterations per block for XRefs analysis
        :param analysis_options: Additional options to pass to each XRefsAnalysis instance
        """
        # Initialize multicore capabilities
        self._init_multicore(workers=workers, mode=mode)

        # Store configuration
        self._skip_alignment = skip_alignment
        self._skip_simprocedures = skip_simprocedures
        self._skip_plt = skip_plt
        self._max_function_size = max_function_size
        self._max_function_blocks = max_function_blocks
        self._max_iterations = max_iterations
        self._analysis_options = analysis_options or {}

        # Results storage
        self.results: dict[int, int] = {}  # func_addr -> xrefs_count
        self.errors: dict[int, str] = {}  # func_addr -> error message
        self.analyzed_count: int = 0
        self.failed_count: int = 0
        self.total_xrefs_count: int = 0

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
        """Run the parallel XRefs analysis."""
        if not self._func_addrs:
            _l.info("No functions to analyze")
            self._finish_progress()
            return

        _l.info(
            "Starting parallel XRefs analysis for %d functions with %d workers",
            len(self._func_addrs),
            self._multicore_config.workers,
        )

        # Prepare tasks
        tasks = []
        for func_addr in self._func_addrs:
            task = {
                "func_addr": func_addr,
                "options": {
                    "max_iterations": self._max_iterations,
                    **self._analysis_options,
                },
            }
            tasks.append(task)

        # Pickle project for workers
        project_pickle = pickle.dumps(self.project)

        # Execute XRefs analysis in parallel
        if self._multicore_config.mode == ParallelizationMode.SEQUENTIAL:
            # Sequential execution for debugging
            _init_worker(project_pickle)
            for i, task in enumerate(tasks):
                result = _analyze_xrefs(task)
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
                for i, result in enumerate(pool.imap_unordered(_analyze_xrefs, tasks)):
                    self._process_result(result)
                    percentage = (i + 1) / total * 100.0
                    func_name = self._get_func_name(result["func_addr"])
                    self._update_progress(percentage, text=f"{i + 1}/{total} - {func_name}")

        self._finish_progress()

        _l.info(
            "Parallel XRefs analysis complete: %d succeeded, %d failed, %d total xrefs",
            self.analyzed_count,
            self.failed_count,
            self.total_xrefs_count,
        )

    def _process_result(self, result: dict[str, Any]) -> None:
        """Process a single XRefs analysis result."""
        func_addr = result["func_addr"]

        if result["success"]:
            self.results[func_addr] = result["xrefs_count"]
            self.analyzed_count += 1
            self.total_xrefs_count += result["xrefs_count"]

            # Import XRefs into main KB
            for xref_data in result["xrefs"]:
                # XRefType is a class with constants (Offset=0, Read=1, Write=2)
                # We pass the integer value directly as xref_type
                xref = XRef(
                    ins_addr=xref_data["ins_addr"],
                    block_addr=xref_data["block_addr"],
                    stmt_idx=xref_data["stmt_idx"],
                    dst=xref_data["dst"],
                    xref_type=xref_data["xref_type"],
                )
                self.kb.xrefs.add_xref(xref)
        else:
            self.errors[func_addr] = result["error"] or "Unknown error"
            self.failed_count += 1

    def get_xrefs_count(self, func_addr: int) -> int | None:
        """
        Get number of XRefs found for a function.

        :param func_addr: Function address
        :return: Number of XRefs, or None if not available
        """
        return self.results.get(func_addr)

    def get_error(self, func_addr: int) -> str | None:
        """
        Get error message for a failed analysis.

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


register_analysis(ParallelXRefs, "ParallelXRefs")
