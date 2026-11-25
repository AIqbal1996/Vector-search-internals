"""Script to generate and cache real Gemini Embedding 2 vectors for the banking dataset."""

import sys
import json
import time
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from config.settings import RAW_DATA_DIR, EMBEDDINGS_DIR, settings
from src.embedding_client import EmbeddingClient


def main():
    print("=" * 80)
    print("PHASE 2: REAL GEMINI EMBEDDING 2 GENERATION VIA EURI API")
    print("=" * 80)

    # Load 1,000 banking records
    raw_dataset_path = RAW_DATA_DIR / "banking_dataset.json"
    if not raw_dataset_path.exists():
        print("[INFO] Generating fresh banking dataset...")
        from src.dataset_manager import save_banking_dataset
        save_banking_dataset()

    with open(raw_dataset_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"Loaded {len(records)} banking records.")

    # Load 10 representative queries
    queries_path = RAW_DATA_DIR / "representative_queries.json"
    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)
    print(f"Loaded {len(queries)} representative search queries.")

    # Initialize Embedding Client
    client = EmbeddingClient(
        api_key=settings.euri_api_key,
        base_url=settings.euri_base_url,
        model=settings.embedding_model,
        dimension=settings.embedding_dimension,
    )
    print(f"Embedding Provider: EURI REST API")
    print(f"Target Model: {client.model}")
    print(f"Target Dimension: {client.dimension}")
    print("-" * 80)

    # 1. Generate / Cache Dataset Embeddings
    dataset_vectors, dataset_meta = client.generate_and_cache_dataset_embeddings(
        records=records,
        batch_size=25,
        cache_name="banking_embeddings"
    )
    print(f"\n[OK] Dataset vectors shape: {dataset_vectors.shape}, dtype: {dataset_vectors.dtype}")

    # 2. Generate / Cache Query Embeddings
    query_texts = [q["text"] for q in queries]
    query_npy_path = EMBEDDINGS_DIR / "query_embeddings.npy"
    query_meta_path = EMBEDDINGS_DIR / "query_embeddings_meta.json"

    if query_npy_path.exists() and query_meta_path.exists():
        print(f"[CACHE HIT] Loading existing query embeddings: {query_npy_path}")
        query_vectors = np.load(query_npy_path)
    else:
        print(f"[GENERATION] Generating embeddings for {len(query_texts)} queries...")
        query_vectors = client.get_embeddings_batch(query_texts)
        np.save(query_npy_path, query_vectors)
        query_meta = {
            "model": client.model,
            "dimension": int(query_vectors.shape[1]),
            "count": len(query_vectors),
            "created_at": time.time(),
            "source": "EURI_GEMINI_API",
        }
        with open(query_meta_path, "w", encoding="utf-8") as f:
            json.dump(query_meta, f, indent=2)
        print(f"[OK] Saved query embeddings to: {query_npy_path}")

    print(f"[OK] Query vectors shape: {query_vectors.shape}, dtype: {query_vectors.dtype}")

    # 3. Generate Seeded Synthetic Benchmark Datasets (1,000 & 10,000 scale)
    # The 1,000 synthetic dataset is an exact subset of the 10,000 dataset for fair scaling
    dim = dataset_vectors.shape[1]
    print("\n" + "-" * 80)
    print(f"[BENCHMARK DATASET] Preparing seeded synthetic datasets (dim={dim})...")
    synth_10k, _ = client.generate_synthetic_benchmark_dataset(num_vectors=10000, dimension=dim, seed=42)
    synth_1k = synth_10k[:1000]
    np.save(EMBEDDINGS_DIR / f"synthetic_benchmark_1000.npy", synth_1k)
    with open(EMBEDDINGS_DIR / f"synthetic_benchmark_1000_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "dataset_type": "SYNTHETIC_BENCHMARK",
            "model_identifier": "synthetic_gaussian_unit_sphere",
            "dimension": dim,
            "count": 1000,
            "seed": 42,
            "is_real_gemini": False,
            "note": "Exact subset (first 1,000 vectors) of the 10,000 benchmark dataset."
        }, f, indent=2)
    print(f"[OK] Synthetic 1,000 vector benchmark dataset created: shape {synth_1k.shape}")
    print(f"[OK] Synthetic 10,000 vector benchmark dataset created: shape {synth_10k.shape}")

    print("\n" + "=" * 80)
    print("[SUCCESS] Phase 2 Embedding Generation Completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
