#!/usr/bin/env python3
# pylint: disable=missing-class-docstring,no-self-use,line-too-long
"""
Tests for multicore analysis framework.
"""
from __future__ import annotations

__package__ = __package__ or "tests.analyses"  # pylint:disable=redefined-builtin

import os
import unittest
import multiprocessing

import angr
from angr.analyses.multicore import (
    MulticoreAnalysisMixin,
    ParallelAnalysisConfig,
    ParallelAnalysisManager,
    ParallelTaskExecutor,
    ParallelizationMode,
    run_parallel_analysis,
)

from tests.common import bin_location


test_location = os.path.join(bin_location, "tests")


class TestParallelAnalysisConfig(unittest.TestCase):
    """Tests for ParallelAnalysisConfig."""

    def test_default_workers(self):
        """Test that default workers is set to CPU count."""
        config = ParallelAnalysisConfig()
        self.assertEqual(config.workers, multiprocessing.cpu_count())

    def test_explicit_workers(self):
        """Test explicit worker count."""
        config = ParallelAnalysisConfig(workers=4)
        self.assertEqual(config.workers, 4)

    def test_zero_workers_becomes_cpu_count(self):
        """Test that 0 workers is converted to CPU count."""
        config = ParallelAnalysisConfig(workers=0)
        self.assertEqual(config.workers, multiprocessing.cpu_count())

    def test_mode_default(self):
        """Test default parallelization mode."""
        config = ParallelAnalysisConfig()
        self.assertEqual(config.mode, ParallelizationMode.MULTIPROCESSING)


class TestParallelTaskExecutor(unittest.TestCase):
    """Tests for ParallelTaskExecutor."""

    def test_sequential_execution(self):
        """Test sequential execution mode."""
        config = ParallelAnalysisConfig(
            workers=1,
            mode=ParallelizationMode.SEQUENTIAL,
        )

        def worker_func(x):
            return x * 2

        executor = ParallelTaskExecutor(config, worker_func)
        results = executor.execute([1, 2, 3, 4, 5])

        self.assertEqual(len(results), 5)
        for r in results:
            self.assertTrue(r.success)

    def test_threaded_execution(self):
        """Test threaded execution mode."""
        config = ParallelAnalysisConfig(
            workers=2,
            mode=ParallelizationMode.THREADING,
        )

        def worker_func(x):
            return x * 2

        executor = ParallelTaskExecutor(config, worker_func)
        results = executor.execute([1, 2, 3, 4, 5])

        self.assertEqual(len(results), 5)
        for r in results:
            self.assertTrue(r.success)

    def test_error_handling(self):
        """Test that errors are captured properly."""
        config = ParallelAnalysisConfig(
            workers=1,
            mode=ParallelizationMode.SEQUENTIAL,
        )

        def worker_func(x):
            if x == 3:
                raise ValueError("Test error")
            return x * 2

        executor = ParallelTaskExecutor(config, worker_func)
        results = executor.execute([1, 2, 3, 4, 5])

        self.assertEqual(len(results), 5)
        # Check that one result has an error
        error_results = [r for r in results if not r.success]
        self.assertEqual(len(error_results), 1)
        self.assertIsInstance(error_results[0].error, ValueError)

    def test_fail_fast(self):
        """Test fail_fast mode stops on first error."""
        config = ParallelAnalysisConfig(
            workers=1,
            mode=ParallelizationMode.SEQUENTIAL,
            fail_fast=True,
        )

        def worker_func(x):
            if x == 2:
                raise ValueError("Test error")
            return x * 2

        executor = ParallelTaskExecutor(config, worker_func)
        results = executor.execute([1, 2, 3, 4, 5])

        # Should stop after error at x=2
        self.assertLessEqual(len(results), 2)


class TestParallelAnalysisManager(unittest.TestCase):
    """Tests for ParallelAnalysisManager."""

    def test_manager_creation(self):
        """Test creating a ParallelAnalysisManager."""
        bin_path = os.path.join(test_location, "x86_64", "fauxware")
        proj = angr.Project(bin_path, auto_load_libs=False)

        manager = ParallelAnalysisManager(proj, workers=2)
        self.assertEqual(manager.workers, 2)
        self.assertEqual(manager.project, proj)


class TestParallelDecompiler(unittest.TestCase):
    """Tests for ParallelDecompiler analysis."""

    def test_parallel_decompiler_sequential(self):
        """Test ParallelDecompiler in sequential mode."""
        bin_path = os.path.join(test_location, "x86_64", "fauxware")
        proj = angr.Project(bin_path, auto_load_libs=False)
        cfg = proj.analyses.CFGFast()

        # Get a few functions to decompile
        func_addrs = list(proj.kb.functions.keys())[:3]

        if not func_addrs:
            self.skipTest("No functions found in binary")

        parallel_dec = proj.analyses.ParallelDecompiler(
            cfg=cfg,
            workers=1,
            mode=ParallelizationMode.SEQUENTIAL,
            func_addrs=func_addrs,
        )

        # Check results
        self.assertGreater(parallel_dec.decompiled_count, 0)
        self.assertLessEqual(parallel_dec.failed_count, len(func_addrs))

    def test_parallel_decompiler_multiprocessing(self):
        """Test ParallelDecompiler in multiprocessing mode."""
        bin_path = os.path.join(test_location, "x86_64", "fauxware")
        proj = angr.Project(bin_path, auto_load_libs=False)
        cfg = proj.analyses.CFGFast()

        # Get a few functions to decompile
        func_addrs = list(proj.kb.functions.keys())[:5]

        if not func_addrs:
            self.skipTest("No functions found in binary")

        parallel_dec = proj.analyses.ParallelDecompiler(
            cfg=cfg,
            workers=2,
            mode=ParallelizationMode.MULTIPROCESSING,
            func_addrs=func_addrs,
        )

        # Check results
        self.assertGreater(parallel_dec.decompiled_count, 0)


class TestParallelVariableRecovery(unittest.TestCase):
    """Tests for ParallelVariableRecovery analysis."""

    def test_parallel_variable_recovery_sequential(self):
        """Test ParallelVariableRecovery in sequential mode."""
        bin_path = os.path.join(test_location, "x86_64", "fauxware")
        proj = angr.Project(bin_path, auto_load_libs=False)
        cfg = proj.analyses.CFGFast()

        # Get a few functions to analyze
        func_addrs = list(proj.kb.functions.keys())[:3]

        if not func_addrs:
            self.skipTest("No functions found in binary")

        parallel_vr = proj.analyses.ParallelVariableRecovery(
            workers=1,
            mode=ParallelizationMode.SEQUENTIAL,
            func_addrs=func_addrs,
        )

        # Check results
        self.assertGreater(parallel_vr.recovered_count, 0)

    def test_parallel_variable_recovery_multiprocessing(self):
        """Test ParallelVariableRecovery in multiprocessing mode."""
        bin_path = os.path.join(test_location, "x86_64", "fauxware")
        proj = angr.Project(bin_path, auto_load_libs=False)
        cfg = proj.analyses.CFGFast()

        # Get a few functions to analyze
        func_addrs = list(proj.kb.functions.keys())[:5]

        if not func_addrs:
            self.skipTest("No functions found in binary")

        parallel_vr = proj.analyses.ParallelVariableRecovery(
            workers=2,
            mode=ParallelizationMode.MULTIPROCESSING,
            func_addrs=func_addrs,
        )

        # Check results
        self.assertGreater(parallel_vr.recovered_count, 0)


class TestParallelXRefs(unittest.TestCase):
    """Tests for ParallelXRefs analysis."""

    def test_parallel_xrefs_sequential(self):
        """Test ParallelXRefs in sequential mode."""
        bin_path = os.path.join(test_location, "x86_64", "fauxware")
        proj = angr.Project(bin_path, auto_load_libs=False)
        cfg = proj.analyses.CFGFast()

        # Get a few functions to analyze
        func_addrs = list(proj.kb.functions.keys())[:3]

        if not func_addrs:
            self.skipTest("No functions found in binary")

        parallel_xrefs = proj.analyses.ParallelXRefs(
            workers=1,
            mode=ParallelizationMode.SEQUENTIAL,
            func_addrs=func_addrs,
        )

        # Check results
        self.assertGreater(parallel_xrefs.analyzed_count, 0)

    def test_parallel_xrefs_multiprocessing(self):
        """Test ParallelXRefs in multiprocessing mode."""
        bin_path = os.path.join(test_location, "x86_64", "fauxware")
        proj = angr.Project(bin_path, auto_load_libs=False)
        cfg = proj.analyses.CFGFast()

        # Get a few functions to analyze
        func_addrs = list(proj.kb.functions.keys())[:5]

        if not func_addrs:
            self.skipTest("No functions found in binary")

        parallel_xrefs = proj.analyses.ParallelXRefs(
            workers=2,
            mode=ParallelizationMode.MULTIPROCESSING,
            func_addrs=func_addrs,
        )

        # Check results
        self.assertGreater(parallel_xrefs.analyzed_count, 0)


class TestRunParallelAnalysis(unittest.TestCase):
    """Tests for run_parallel_analysis convenience function."""

    def test_run_parallel_analysis(self):
        """Test the run_parallel_analysis convenience function."""
        bin_path = os.path.join(test_location, "x86_64", "fauxware")
        proj = angr.Project(bin_path, auto_load_libs=False)
        cfg = proj.analyses.CFGFast()

        # Get a few functions
        func_addrs = list(proj.kb.functions.keys())[:3]

        if not func_addrs:
            self.skipTest("No functions found in binary")

        def analysis_func(project, func_addr):
            func = project.kb.functions.get_by_addr(func_addr)
            return func.name if func else None

        results = run_parallel_analysis(
            proj,
            func_addrs,
            analysis_func,
            workers=2,
            mode=ParallelizationMode.SEQUENTIAL,
        )

        self.assertEqual(len(results), len(func_addrs))


if __name__ == "__main__":
    unittest.main()
