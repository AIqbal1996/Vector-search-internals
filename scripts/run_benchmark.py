"""Phase 7: Script to run complete 1k and 10k vector search benchmarks."""

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.benchmark import BenchmarkRunner


def main():
    print("Initiating Phase 7 Performance Benchmarks...")
    runner = BenchmarkRunner()
    
    # Run full benchmarks across 1,000 and 10,000 vectors
    df, summary = runner.run_benchmark_suite(
        dataset_sizes=[1000, 10000],
        num_warmup=10,
        num_queries=100,
        k=10,
        seed=42
    )

    print("\n[SUCCESS] Benchmark suite completed successfully.")
    print("CSV Results:     results/benchmark_results.csv")
    print("Summary JSON:    results/benchmark_summary.json")


if __name__ == "__main__":
    main()
