"""Phase 7: Performance Benchmarking Engine.

Benchmarks:
1. NumPy brute-force Euclidean
2. NumPy brute-force Cosine
3. Qdrant exact Euclidean
4. Qdrant exact Cosine
5. Qdrant HNSW approximate Euclidean
6. Qdrant HNSW approximate Cosine

Evaluated across:
- 1,000 vectors
- 10,000 vectors

Methodology:
- Precise time.perf_counter_ns()
- 10 warm-up queries
- 100 measured queries per method and scale
- Top-k = 10
- Computes Mean, Median, Min, Max, P95, P99, StdDev, QPS, Recall@10
- Records system telemetry (CPU, RAM, OS, Versions, Network location)
- Exports results/benchmark_results.csv and results/benchmark_summary.json
"""

import os
import csv
import time
import json
import platform
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from tabulate import tabulate

from config.settings import RESULTS_DIR, EMBEDDINGS_DIR, settings
from src.brute_force_search import BruteForceVectorSearch
from src.qdrant_manager import QdrantManager
from qdrant_client.http import models


class BenchmarkRunner:
    """Orchestrates comprehensive vector search benchmarking."""

    def __init__(self, qdrant_manager: Optional[QdrantManager] = None):
        self.qdrant_manager = qdrant_manager or QdrantManager()
        self.system_info = self._collect_system_info()

    @staticmethod
    def _collect_system_info() -> Dict[str, Any]:
        """Collect hardware, OS, and software environment specifications."""
        return {
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "cpu": platform.processor() or "x86_64",
            "cpu_cores": os.cpu_count() or 4,
            "qdrant_target": "Qdrant Cloud (AWS sa-east-1)" if "cloud.qdrant.io" in settings.qdrant_url else "Local Docker",
            "qdrant_url": settings.qdrant_url,
            "precomputed_norms": True,
        }

    @staticmethod
    def _calculate_statistics(latencies_ns: List[int]) -> Dict[str, float]:
        """Convert nanosecond latencies to comprehensive millisecond statistics."""
        lat_ms = np.array(latencies_ns) / 1_000_000.0  # Convert to ms
        mean_ms = float(np.mean(lat_ms))
        median_ms = float(np.median(lat_ms))
        min_ms = float(np.min(lat_ms))
        max_ms = float(np.max(lat_ms))
        p95_ms = float(np.percentile(lat_ms, 95))
        p99_ms = float(np.percentile(lat_ms, 99))
        std_ms = float(np.std(lat_ms))
        qps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

        return {
            "mean_ms": round(mean_ms, 4),
            "median_ms": round(median_ms, 4),
            "min_ms": round(min_ms, 4),
            "max_ms": round(max_ms, 4),
            "p95_ms": round(p95_ms, 4),
            "p99_ms": round(p99_ms, 4),
            "std_ms": round(std_ms, 4),
            "qps": round(qps, 2),
        }

    @staticmethod
    def calculate_recall(ground_truth_ids: List[int], test_ids: List[int]) -> float:
        """Calculate Recall@k given ground truth IDs and test IDs."""
        k = len(ground_truth_ids)
        if k == 0:
            return 1.0
        return len(set(ground_truth_ids).intersection(set(test_ids))) / k

    def run_benchmark_suite(
        self,
        dataset_sizes: List[int] = [1000, 10000],
        num_warmup: int = 10,
        num_queries: int = 100,
        k: int = 10,
        seed: int = 42
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Execute the full benchmark matrix."""
        print("=" * 90)
        print("PHASE 7: VECTOR SEARCH PERFORMANCE BENCHMARKING SUITE")
        print("=" * 90)
        print(f"System: {self.system_info['os']} | CPU Cores: {self.system_info['cpu_cores']}")
        print(f"Python: {self.system_info['python_version']} | NumPy: {self.system_info['numpy_version']} | Target: {self.system_info['qdrant_target']}")
        print(f"Parameters: Warm-up={num_warmup}, Measured Queries={num_queries}, Top-k={k}, Seed={seed}")
        print("-" * 90)

        rng = np.random.default_rng(seed)
        all_results = []
        summary_payload = {
            "system_info": self.system_info,
            "benchmark_parameters": {
                "num_warmup": num_warmup,
                "num_queries": num_queries,
                "top_k": k,
                "seed": seed,
            },
            "results": []
        }

        for size in dataset_sizes:
            print(f"\n>>> BENCHMARKING DATASET SIZE: {size:,} VECTORS <<<")
            npy_path = EMBEDDINGS_DIR / f"synthetic_benchmark_{size}.npy"
            if not npy_path.exists():
                raise FileNotFoundError(f"Benchmark dataset not found: {npy_path}")

            vectors = np.load(npy_path)
            dim = vectors.shape[1]
            ids = list(range(size))

            # Generate query vectors for benchmark
            raw_queries = rng.standard_normal((num_queries + num_warmup, dim), dtype=np.float32)
            norm_queries = (raw_queries / np.linalg.norm(raw_queries, axis=1, keepdims=True)).astype(np.float32)
            warmup_queries = norm_queries[:num_warmup]
            eval_queries = norm_queries[num_warmup:]

            # 1. Initialize NumPy Brute-Force Engine
            numpy_engine = BruteForceVectorSearch(vectors, ids=ids, use_argpartition=True)

            # 2. Initialize Qdrant Collections for this scale
            col_cos = f"benchmark_{size}_cosine"
            col_euc = f"benchmark_{size}_euclid"

            print(f"Setting up Qdrant collections for {size:,} vectors...")
            self.qdrant_manager.create_collection(col_cos, dim, models.Distance.COSINE, recreate=True)
            self.qdrant_manager.create_collection(col_euc, dim, models.Distance.EUCLID, recreate=True)

            print(f"Uploading {size:,} vectors to Qdrant (cosine & euclidean)...")
            self.qdrant_manager.upload_dataset(col_cos, vectors, ids, batch_size=250)
            self.qdrant_manager.upload_dataset(col_euc, vectors, ids, batch_size=250)

            # 3. Benchmark Methods Definition
            methods = [
                ("NumPy Brute-Force Euclidean", "NumPy", "euclidean", True, None),
                ("NumPy Brute-Force Cosine", "NumPy", "cosine", True, None),
                ("Qdrant Exact Euclidean", "Qdrant", "euclidean", False, {"exact": True}),
                ("Qdrant Exact Cosine", "Qdrant", "cosine", False, {"exact": True}),
                ("Qdrant HNSW Approximate Euclidean", "Qdrant", "euclidean", False, {"exact": False, "hnsw_ef": 64}),
                ("Qdrant HNSW Approximate Cosine", "Qdrant", "cosine", False, {"exact": False, "hnsw_ef": 64}),
            ]

            ground_truth = {
                "euclidean": [],
                "cosine": []
            }

            for name, engine_type, metric, is_gt, extra_params in methods:
                print(f"  -> Testing: {name} ({size:,} vectors)...")
                col_name = col_cos if metric == "cosine" else col_euc

                # Warm-up phase
                for q in warmup_queries:
                    if engine_type == "NumPy":
                        if metric == "euclidean":
                            _ = numpy_engine.search_euclidean(q, k=k)
                        else:
                            _ = numpy_engine.search_cosine(q, k=k)
                    else:
                        _ = self.qdrant_manager.search(
                            col_name, q, k=k,
                            exact=extra_params["exact"],
                            hnsw_ef=extra_params.get("hnsw_ef")
                        )

                # Measurement phase
                latencies_ns = []
                method_results_ids = []

                for q in eval_queries:
                    t_start = time.perf_counter_ns()
                    if engine_type == "NumPy":
                        if metric == "euclidean":
                            res = numpy_engine.search_euclidean(q, k=k)
                        else:
                            res = numpy_engine.search_cosine(q, k=k)
                    else:
                        res = self.qdrant_manager.search(
                            col_name, q, k=k,
                            exact=extra_params["exact"],
                            hnsw_ef=extra_params.get("hnsw_ef")
                        )
                    t_end = time.perf_counter_ns()

                    latencies_ns.append(t_end - t_start)
                    method_results_ids.append([item["id"] for item in res])

                # Store ground truth for recall calculation
                if is_gt:
                    ground_truth[metric] = method_results_ids

                # Calculate Recall@k against NumPy ground truth
                recalls = [
                    self.calculate_recall(ground_truth[metric][i], method_results_ids[i])
                    for i in range(len(eval_queries))
                ]
                mean_recall = float(np.mean(recalls))

                # Compute statistics
                stats = self._calculate_statistics(latencies_ns)
                record = {
                    "dataset_size": size,
                    "method": name,
                    "engine": engine_type,
                    "metric": metric.upper(),
                    "approximate": not (is_gt or (extra_params and extra_params.get("exact"))),
                    "recall_at_10": round(mean_recall, 4),
                    **stats
                }
                all_results.append(record)
                summary_payload["results"].append(record)

        # Print formatted table
        display_headers = ["Size", "Method", "Metric", "Mean (ms)", "P95 (ms)", "QPS", "Recall@10"]
        display_rows = [
            [
                r["dataset_size"],
                r["method"],
                r["metric"],
                f"{r['mean_ms']:.4f}",
                f"{r['p95_ms']:.4f}",
                f"{r['qps']:.1f}",
                f"{r['recall_at_10'] * 100:.2f}%"
            ]
            for r in all_results
        ]
        print("\n" + "=" * 90)
        print("BENCHMARK EXECUTION SUMMARY TABLE")
        print("=" * 90)
        print(tabulate(display_rows, headers=display_headers, tablefmt="grid"))
        print("=" * 90)

        # Export CSV and JSON
        csv_path = RESULTS_DIR / "benchmark_results.csv"
        json_path = RESULTS_DIR / "benchmark_summary.json"

        fieldnames = list(all_results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)

        print(f"[OK] Benchmark results exported to: {csv_path}")
        print(f"[OK] Benchmark summary exported to: {json_path}")

        return all_results, summary_payload
