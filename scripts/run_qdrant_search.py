"""Phase 4: Run Qdrant Vector Search.

1. Connects to Qdrant (local Docker or Qdrant Cloud)
2. Initializes collections:
   - week5_cosine (Distance.COSINE)
   - week5_euclidean (Distance.EUCLID)
   - week5_dot (Distance.DOT)
3. Uploads 1,000 vectors with identical IDs and metadata payloads
4. Executes exact vector search for the 10 representative queries
5. Saves results to results/qdrant_results.json
"""

import sys
import json
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from config.settings import EMBEDDINGS_DIR, RAW_DATA_DIR, RESULTS_DIR, settings
from src.qdrant_manager import QdrantManager


def main():
    print("=" * 80)
    print("PHASE 4: QDRANT VECTOR SEARCH & INGESTION")
    print("=" * 80)

    # 1. Load Embeddings
    npy_path = EMBEDDINGS_DIR / "banking_embeddings.npy"
    query_npy_path = EMBEDDINGS_DIR / "query_embeddings.npy"

    if not npy_path.exists() or not query_npy_path.exists():
        raise FileNotFoundError(
            "Embedding files not found. Please run 'python scripts/generate_embeddings.py' first."
        )

    vectors = np.load(npy_path)
    query_vectors = np.load(query_npy_path)

    with open(RAW_DATA_DIR / "banking_dataset.json", "r", encoding="utf-8") as f:
        records = json.load(f)
    with open(RAW_DATA_DIR / "representative_queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)

    num_records, dim = vectors.shape
    print(f"Connecting to Qdrant at: {settings.qdrant_url}")
    manager = QdrantManager()
    manager.verify_connection()
    print("[OK] Connected to Qdrant successfully.")

    # 2. Initialize Collections
    print("\n[INIT] Initializing Qdrant collections...")
    manager.init_all_collections(dimension=dim, recreate=True)

    # 3. Upload Datasets
    ids = [r["id"] for r in records]
    payloads = [{"domain": r["domain"], "text": r["text"], "metadata": r.get("metadata", {})} for r in records]

    print("\n[UPLOAD] Upserting vectors into all three collections...")
    manager.upload_dataset(manager.collection_cosine, vectors, ids, payloads=payloads)
    manager.upload_dataset(manager.collection_euclidean, vectors, ids, payloads=payloads)
    manager.upload_dataset(manager.collection_dot, vectors, ids, payloads=payloads)

    # 4. Execute Exact Search for the 10 Queries
    print("\n[SEARCH] Executing exact search for 10 representative queries...")
    results_payload = {
        "dataset_size": num_records,
        "dimension": dim,
        "query_count": len(query_vectors),
        "cosine": {},
        "euclidean": {},
        "dot": {},
    }

    for q_idx, q_vec in enumerate(query_vectors):
        cos_res = manager.search(manager.collection_cosine, q_vec, k=10, exact=True)
        euc_res = manager.search(manager.collection_euclidean, q_vec, k=10, exact=True)
        dot_res = manager.search(manager.collection_dot, q_vec, k=10, exact=True)

        results_payload["cosine"][str(q_idx)] = cos_res
        results_payload["euclidean"][str(q_idx)] = euc_res
        results_payload["dot"][str(q_idx)] = dot_res

    # Save to results/qdrant_results.json
    output_path = RESULTS_DIR / "qdrant_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print("\n" + "=" * 80)
    print(f"[SUCCESS] Qdrant search completed. Results saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
