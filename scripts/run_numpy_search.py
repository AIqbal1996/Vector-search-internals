"""Phase 3: Run NumPy Brute-Force Vector Search.

Executes 10 representative semantic queries against the 1,000 banking dataset vectors:
- Metric 1: Euclidean Distance (ascending rank)
- Metric 2: Cosine Similarity (descending rank)
- Metric 3: Dot Product Similarity (descending rank)
- Evaluations at k = 1, 5, 10
- Exports detailed search rankings and scores to results/numpy_results.json
"""

import sys
import json
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from tabulate import tabulate

from config.settings import EMBEDDINGS_DIR, RAW_DATA_DIR, RESULTS_DIR
from src.brute_force_search import BruteForceVectorSearch


def main():
    print("=" * 80)
    print("PHASE 3: NUMPY BRUTE-FORCE VECTOR SEARCH EXECUTION")
    print("=" * 80)

    # 1. Load Embeddings
    npy_path = EMBEDDINGS_DIR / "banking_embeddings.npy"
    query_npy_path = EMBEDDINGS_DIR / "query_embeddings.npy"

    if not npy_path.exists() or not query_npy_path.exists():
        raise FileNotFoundError(
            f"Embedding files not found. Please run 'python scripts/generate_embeddings.py' first."
        )

    vectors = np.load(npy_path)
    query_vectors = np.load(query_npy_path)

    # Load metadata and queries
    with open(RAW_DATA_DIR / "banking_dataset.json", "r", encoding="utf-8") as f:
        records = json.load(f)
    with open(RAW_DATA_DIR / "representative_queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)

    num_records, dim = vectors.shape
    print(f"Dataset Size: {num_records} vectors (dim={dim})")
    print(f"Query Count:  {len(query_vectors)} queries")
    print("-" * 80)

    # Initialize Engine
    engine = BruteForceVectorSearch(vectors, ids=[r["id"] for r in records], use_argpartition=True)

    results_payload = {
        "dataset_size": num_records,
        "dimension": dim,
        "query_count": len(query_vectors),
        "cosine": {},
        "euclidean": {},
        "dot": {},
    }

    # Evaluate for k = 10 (covering 1, 5, 10)
    for q_idx, q_vec in enumerate(query_vectors):
        q_info = queries[q_idx]
        q_text = q_info["text"]

        cos_results = engine.search_cosine(q_vec, k=10)
        euc_results = engine.search_euclidean(q_vec, k=10)
        dot_results = engine.search_dot_product(q_vec, k=10)

        # Attach snippet text to results
        for r in cos_results:
            r["text_snippet"] = records[r["id"]]["text"][:65] + "..."
            r["domain"] = records[r["id"]]["domain"]
        for r in euc_results:
            r["text_snippet"] = records[r["id"]]["text"][:65] + "..."
            r["domain"] = records[r["id"]]["domain"]
        for r in dot_results:
            r["text_snippet"] = records[r["id"]]["text"][:65] + "..."
            r["domain"] = records[r["id"]]["domain"]

        results_payload["cosine"][str(q_idx)] = cos_results
        results_payload["euclidean"][str(q_idx)] = euc_results
        results_payload["dot"][str(q_idx)] = dot_results

        print(f"\nQuery {q_idx}: \"{q_text}\" (Target: {q_info['target_domain']})")
        print("  Top-3 Cosine Similarity:")
        for r in cos_results[:3]:
            print(f"    Rank {r['rank']}: ID={r['id']:<4} | Cosine={r['score']:.4f} | [{r['domain']}] {r['text_snippet']}")
        print("  Top-3 Euclidean Distance:")
        for r in euc_results[:3]:
            print(f"    Rank {r['rank']}: ID={r['id']:<4} | Euclid={r['score']:.4f} | [{r['domain']}] {r['text_snippet']}")

    # Save to results/numpy_results.json
    output_path = RESULTS_DIR / "numpy_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print("\n" + "=" * 80)
    print(f"[SUCCESS] NumPy search completed for {len(query_vectors)} queries.")
    print(f"Results saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
