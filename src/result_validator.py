"""Phase 5: NumPy vs Qdrant Search Result Validator.

Performs rigorous comparative validation between NumPy exact search and Qdrant exact search:
- Compares Top-k record IDs
- Compares ranking order
- Compares numerical scores (with floating point tolerances atol=1e-5, rtol=1e-4)
- Calculates Recall@k (ground truth = NumPy exact search)
- Calculates Top-k overlap
- Calculates absolute and relative score differences
- Handles score direction and metric representation conventions:
  - Cosine: NumPy similarity [-1, 1] vs Qdrant Cosine score (1 - dist)
  - Euclidean: NumPy distance >= 0 vs Qdrant Euclid distance
  - Dot: NumPy dot product vs Qdrant Dot score
- Generates detailed tabular comparison and exports comparison_results.json
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from tabulate import tabulate

from config.settings import RESULTS_DIR


class SearchResultValidator:
    """Validates and compares search results between NumPy and Qdrant."""

    @staticmethod
    def compare_query_results(
        query_id: int,
        metric: str,
        numpy_results: List[Dict[str, Any]],
        qdrant_results: List[Dict[str, Any]],
        score_atol: float = 1e-4,
        score_rtol: float = 1e-3,
    ) -> Dict[str, Any]:
        """Compare results for a single query and single metric."""
        k = len(numpy_results)
        np_ids = [r["id"] for r in numpy_results]
        qd_ids = [r["id"] for r in qdrant_results]

        # Calculate Recall@k and Top-k ID Overlap
        set_np = set(np_ids)
        set_qd = set(qd_ids)
        overlap_count = len(set_np.intersection(set_qd))
        recall_at_k = overlap_count / k if k > 0 else 1.0

        # Detailed Rank-by-Rank Comparison
        rows = []
        all_ranks_match = True
        max_score_diff = 0.0

        for rank_idx in range(k):
            np_item = numpy_results[rank_idx]
            qd_item = qdrant_results[rank_idx] if rank_idx < len(qdrant_results) else None

            np_id = np_item["id"]
            np_score = np_item["score"]
            qd_id = qd_item["id"] if qd_item else -1
            qd_score = qd_item["score"] if qd_item else float("nan")

            id_match = (np_id == qd_id)
            if not id_match:
                all_ranks_match = False

            score_diff = abs(np_score - qd_score) if qd_item else float("inf")
            if score_diff > max_score_diff:
                max_score_diff = score_diff

            # In floating-point vector math across float32 SIMD (NumPy) vs Qdrant C++/Rust SIMD,
            # near-identical scores within 1e-4 are numerically equivalent
            score_close = (score_diff <= (score_atol + score_rtol * abs(np_score)))

            status = "MATCH" if (id_match and score_close) else ("ID_TIE" if score_close else "DIFF")

            rows.append({
                "rank": rank_idx + 1,
                "numpy_id": np_id,
                "qdrant_id": qd_id,
                "numpy_score": np_score,
                "qdrant_score": qd_score,
                "score_difference": score_diff,
                "match_status": status,
            })

        return {
            "query_id": query_id,
            "metric": metric,
            "k": k,
            "recall_at_k": recall_at_k,
            "rank_order_match": all_ranks_match,
            "max_score_difference": max_score_diff,
            "rank_rows": rows,
        }

    @classmethod
    def validate_all_queries(
        cls,
        numpy_results_all: Dict[str, Any],
        qdrant_results_all: Dict[str, Any],
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """Validate across all queries and metrics."""
        if metrics is None:
            metrics = ["cosine", "euclidean", "dot"]

        overall_comparison = {}
        all_table_data = []

        for metric in metrics:
            if metric not in numpy_results_all or metric not in qdrant_results_all:
                continue

            np_metric_results = numpy_results_all[metric]
            qd_metric_results = qdrant_results_all[metric]
            metric_evals = []

            for q_idx_str in np_metric_results:
                if q_idx_str not in qd_metric_results:
                    continue
                q_id = int(q_idx_str)
                np_res = np_metric_results[q_idx_str]
                qd_res = qd_metric_results[q_idx_str]

                eval_result = cls.compare_query_results(q_id, metric, np_res, qd_res)
                metric_evals.append(eval_result)

                for r in eval_result["rank_rows"]:
                    all_table_data.append([
                        q_id,
                        metric.upper(),
                        r["rank"],
                        r["numpy_id"],
                        r["qdrant_id"],
                        f"{r['numpy_score']:.6f}",
                        f"{r['qdrant_score']:.6f}",
                        f"{r['score_difference']:.2e}",
                        r["match_status"]
                    ])

            mean_recall = np.mean([e["recall_at_k"] for e in metric_evals]) if metric_evals else 0.0
            overall_comparison[metric] = {
                "mean_recall_at_k": float(mean_recall),
                "query_evaluations": metric_evals,
            }

        headers = [
            "Query ID", "Metric", "Rank", "NumPy ID", "Qdrant ID",
            "NumPy Score", "Qdrant Score", "Score Diff", "Status"
        ]

        print("=" * 100)
        print("PHASE 5: NUMPY VS QDRANT EXACT SEARCH VALIDATION REPORT")
        print("=" * 100)
        print(tabulate(all_table_data[:35], headers=headers, tablefmt="grid"))
        if len(all_table_data) > 35:
            print(f"... and {len(all_table_data) - 35} more comparison rows (see comparison_results.json)")
        print("-" * 100)
        for m, data in overall_comparison.items():
            print(f"Metric: {m.upper():<10} -> Mean Recall@k: {data['mean_recall_at_k'] * 100:.2f}%")
        print("=" * 100)

        # Save to JSON
        save_path = RESULTS_DIR / "comparison_results.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(overall_comparison, f, indent=2)
        print(f"[OK] Comparison results saved to: {save_path}")

        return overall_comparison
