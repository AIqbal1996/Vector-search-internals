"""Unit tests for Phase 7 Benchmark metrics and recall calculations."""

import pytest
import numpy as np
from src.benchmark import BenchmarkRunner


class TestBenchmarkCalculations:
    """Validate benchmark statistical calculations and recall accuracy."""

    def test_calculate_statistics(self):
        # 10 synthetic query measurements: [1ms, 2ms, 3ms, ..., 10ms] in nanoseconds
        latencies_ns = [i * 1_000_000 for i in range(1, 11)]
        stats = BenchmarkRunner._calculate_statistics(latencies_ns)

        assert stats["mean_ms"] == 5.5
        assert stats["median_ms"] == 5.5
        assert stats["min_ms"] == 1.0
        assert stats["max_ms"] == 10.0
        assert stats["p95_ms"] > 9.0
        assert stats["qps"] > 0

    def test_calculate_recall_perfect_match(self):
        ground_truth = [1, 2, 3, 4, 5]
        test_ids = [5, 4, 3, 2, 1]  # Permuted order but identical set
        recall = BenchmarkRunner.calculate_recall(ground_truth, test_ids)
        assert recall == 1.0

    def test_calculate_recall_partial_match(self):
        ground_truth = [1, 2, 3, 4, 5]
        test_ids = [1, 2, 3, 9, 10]  # 3 out of 5 match
        recall = BenchmarkRunner.calculate_recall(ground_truth, test_ids)
        assert recall == 0.6

    def test_calculate_recall_zero_match(self):
        ground_truth = [1, 2, 3]
        test_ids = [4, 5, 6]
        recall = BenchmarkRunner.calculate_recall(ground_truth, test_ids)
        assert recall == 0.0
