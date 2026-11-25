"""Phase 5: Execute NumPy vs Qdrant Search Result Validation.

Loads results/numpy_results.json and results/qdrant_results.json,
performs rank-by-rank and score comparison across Cosine, Euclidean, and Dot Product,
computes Recall@k, and exports results/comparison_results.json.
"""

import sys
import json
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import RESULTS_DIR
from src.result_validator import SearchResultValidator


def main():
    print("=" * 80)
    print("PHASE 5: RUN NUMPY VS QDRANT SEARCH VALIDATION")
    print("=" * 80)

    numpy_results_path = RESULTS_DIR / "numpy_results.json"
    qdrant_results_path = RESULTS_DIR / "qdrant_results.json"

    if not numpy_results_path.exists():
        raise FileNotFoundError(f"Missing {numpy_results_path}. Run scripts/run_numpy_search.py first.")
    if not qdrant_results_path.exists():
        raise FileNotFoundError(f"Missing {qdrant_results_path}. Run scripts/run_qdrant_search.py first.")

    with open(numpy_results_path, "r", encoding="utf-8") as f:
        numpy_results = json.load(f)
    with open(qdrant_results_path, "r", encoding="utf-8") as f:
        qdrant_results = json.load(f)

    print(f"NumPy queries loaded:  {numpy_results['query_count']}")
    print(f"Qdrant queries loaded: {qdrant_results['query_count']}")
    print("-" * 80)

    # Perform validation
    comparison = SearchResultValidator.validate_all_queries(
        numpy_results_all=numpy_results,
        qdrant_results_all=qdrant_results,
        metrics=["cosine", "euclidean", "dot"]
    )

    print("\n[SUCCESS] Phase 5 validation completed.")
    print(f"Comparison report saved to: {RESULTS_DIR / 'comparison_results.json'}")


if __name__ == "__main__":
    main()
