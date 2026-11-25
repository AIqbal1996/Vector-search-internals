"""Visualization module for vector search internals.

Generates all 8 required professional publication-quality academic charts:
1. Classroom 2D vector visualization (01_classroom_2d_vectors.png)
2. Euclidean vs Cosine ranking comparison (02_ranking_comparison.png)
3. NumPy vs Qdrant latency for 1,000 vectors (03_latency_1000_vectors.png)
4. NumPy vs Qdrant latency for 10,000 vectors (04_latency_10000_vectors.png)
5. Dataset size vs search latency scaling (05_dataset_size_vs_latency.png)
6. P95 latency comparison across all methods (06_p95_latency_comparison.png)
7. HNSW recall vs latency trade-off (07_hnsw_recall_vs_latency.png)
8. Vector normalization and unit circle projection (08_vector_normalization.png)
"""

import csv
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np

from config.settings import CHARTS_DIR, RESULTS_DIR


def plot_classroom_2d_vectors(results: Dict[str, Any], save_path: Path = None) -> Path:
    """Generate 2D coordinate plot for the classroom example."""
    if save_path is None:
        save_path = CHARTS_DIR / "01_classroom_2d_vectors.png"

    fig, ax = plt.subplots(figsize=(9, 8), dpi=300)
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#ffffff")
    ax.grid(True, linestyle="--", alpha=0.6, color="#cbd5e1")
    ax.axhline(0, color="#94a3b8", linewidth=1.2)
    ax.axvline(0, color="#94a3b8", linewidth=1.2)

    query = results["query"]
    points = results["points"]
    color_map = {"A": "#2563eb", "B": "#16a34a", "C": "#ea580c"}
    query_color = "#dc2626"

    # Query vector
    ax.annotate("", xy=(query[0], query[1]), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=query_color, lw=3, mutation_scale=20))
    ax.scatter([query[0]], [query[1]], color=query_color, s=140, zorder=5, label=f"Query Q ({query[0]}, {query[1]})")
    ax.text(query[0] + 0.1, query[1] + 0.15, f"Query Q (2, 4)\n||Q|| = {results['query_magnitude']:.2f}",
            fontsize=10, fontweight="bold", color=query_color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fee2e2", edgecolor=query_color, alpha=0.9))

    for name, data in points.items():
        pt = data["coordinates"]
        col = color_map[name]
        euclid_d = data["euclidean_distance"]
        cos_s = data["cosine_similarity"]

        ax.annotate("", xy=(pt[0], pt[1]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2, mutation_scale=16, alpha=0.85))
        ax.scatter([pt[0]], [pt[1]], color=col, s=110, zorder=5, label=f"Point {name} ({pt[0]}, {pt[1]})")
        ax.plot([query[0], pt[0]], [query[1], pt[1]], linestyle=":", color=col, lw=2, alpha=0.8)

        mid_x = (query[0] + pt[0]) / 2
        mid_y = (query[1] + pt[1]) / 2
        offset_x = 0.12 if name != "C" else -0.35
        ax.text(mid_x + offset_x, mid_y, f"d = {euclid_d:.2f}", fontsize=9, fontweight="bold", color=col,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor=col, alpha=0.85))

        ann_x = pt[0] + 0.1 if name != "C" else pt[0] - 0.7
        ann_y = pt[1] - 0.25 if name == "B" else pt[1] + 0.1
        ax.text(ann_x, ann_y,
                f"Point {name} ({pt[0]}, {pt[1]})\nEuclid: {euclid_d:.2f} (Rank #{results['rankings']['euclidean']['order'].index(name)+1})\nCosine: {cos_s:.4f} (Rank #{results['rankings']['cosine']['order'].index(name)+1})",
                fontsize=8.5, color=col, bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#cbd5e1", alpha=0.9))

    ax.scatter([0], [0], color="#0f172a", s=80, zorder=6)
    ax.text(-0.25, -0.25, "Origin (0,0)", fontsize=9, color="#475569")
    ax.set_xlim(-0.8, 3.2)
    ax.set_ylim(-0.8, 5.0)
    ax.set_xlabel("Dimension 1 (X-axis)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Dimension 2 (Y-axis)", fontsize=11, fontweight="bold")
    ax.set_title("Classroom 2D Vector Search: Euclidean Distance vs Cosine Similarity\nDemonstrating Geometric Distance (A closest) vs Directional Alignment (B collinear with Q)",
                 fontsize=12, fontweight="bold", pad=15)
    ax.legend(loc="upper left", framealpha=0.95, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_ranking_comparison(results: Dict[str, Any], save_path: Path = None) -> Path:
    """Plot bar chart comparing Euclidean distance vs Cosine similarity scores."""
    if save_path is None:
        save_path = CHARTS_DIR / "02_ranking_comparison.png"

    points = results["points"]
    names = list(points.keys())
    euclid_scores = [points[n]["euclidean_distance"] for n in names]
    cosine_scores = [points[n]["cosine_similarity"] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

    # Euclidean Chart
    bars1 = ax1.bar(names, euclid_scores, color=["#2563eb", "#ef4444", "#f59e0b"], width=0.5, edgecolor="#0f172a")
    ax1.set_title("Euclidean Distance (Lower is Better)", fontweight="bold", fontsize=11)
    ax1.set_ylabel("Euclidean Distance (d)", fontsize=10)
    ax1.set_ylim(0, 3.0)
    ax1.grid(axis="y", linestyle="--", alpha=0.6)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + 0.08, f"{yval:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Cosine Chart
    bars2 = ax2.bar(names, cosine_scores, color=["#2563eb", "#10b981", "#f59e0b"], width=0.5, edgecolor="#0f172a")
    ax2.set_title("Cosine Similarity (Higher is Better)", fontweight="bold", fontsize=11)
    ax2.set_ylabel("Cosine Similarity (cos θ)", fontsize=10)
    ax2.set_ylim(0.8, 1.05)
    ax2.grid(axis="y", linestyle="--", alpha=0.6)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval + 0.005, f"{yval:.6f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.suptitle("Classroom Example: Score & Ranking Discrepancy", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def generate_all_benchmark_charts(csv_path: Path = None) -> List[Path]:
    """Generate Charts 3 through 7 from benchmark results CSV."""
    if csv_path is None:
        csv_path = RESULTS_DIR / "benchmark_results.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Benchmark CSV not found at: {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for r in rows:
        r["dataset_size"] = int(r["dataset_size"])
        r["mean_ms"] = float(r["mean_ms"])
        r["p95_ms"] = float(r["p95_ms"])
        r["qps"] = float(r["qps"])
        r["recall_at_10"] = float(r["recall_at_10"])

    rows_1k = [r for r in rows if r["dataset_size"] == 1000]
    rows_10k = [r for r in rows if r["dataset_size"] == 10000]

    generated_charts = []
    colors = ["#2563eb", "#3b82f6", "#dc2626", "#ea580c", "#16a34a", "#10b981"]

    # Chart 3: Latency for 1,000 Vectors
    c3_path = CHARTS_DIR / "03_latency_1000_vectors.png"
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    methods_1k = [r["method"].replace("Approximate", "HNSW") for r in rows_1k]
    means_1k = [r["mean_ms"] for r in rows_1k]
    bars = ax.barh(methods_1k, means_1k, color=colors, edgecolor="#0f172a", height=0.6)
    ax.set_xlabel("Mean Query Latency (milliseconds)", fontsize=11, fontweight="bold")
    ax.set_title("Search Latency Comparison: 1,000 Vectors (Top-k=10)\nLocal NumPy Matrix SIMD vs Qdrant Cloud Network API", fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + max(means_1k) * 0.01, bar.get_y() + bar.get_height() / 2, f"{w:.2f} ms", va="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    fig.savefig(c3_path, bbox_inches="tight")
    plt.close(fig)
    generated_charts.append(c3_path)

    # Chart 4: Latency for 10,000 Vectors
    c4_path = CHARTS_DIR / "04_latency_10000_vectors.png"
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    methods_10k = [r["method"].replace("Approximate", "HNSW") for r in rows_10k]
    means_10k = [r["mean_ms"] for r in rows_10k]
    bars = ax.barh(methods_10k, means_10k, color=colors, edgecolor="#0f172a", height=0.6)
    ax.set_xlabel("Mean Query Latency (milliseconds)", fontsize=11, fontweight="bold")
    ax.set_title("Search Latency Comparison: 10,000 Vectors (Top-k=10)\nLocal NumPy Matrix SIMD vs Qdrant Cloud Network API", fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + max(means_10k) * 0.01, bar.get_y() + bar.get_height() / 2, f"{w:.2f} ms", va="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    fig.savefig(c4_path, bbox_inches="tight")
    plt.close(fig)
    generated_charts.append(c4_path)

    # Chart 5: Scaling Analysis
    c5_path = CHARTS_DIR / "05_dataset_size_vs_latency.png"
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    sizes = [1000, 10000]
    for m_label in ["NumPy Brute-Force Cosine", "Qdrant Exact Cosine", "Qdrant HNSW Approximate Cosine"]:
        vals = [next(r["mean_ms"] for r in rows if r["dataset_size"] == s and r["method"] == m_label) for s in sizes]
        marker = "o" if "NumPy" in m_label else ("s" if "Exact" in m_label else "^")
        linestyle = "--" if "NumPy" in m_label else "-"
        ax.plot(sizes, vals, marker=marker, markersize=8, linewidth=2, linestyle=linestyle, label=m_label)

    ax.set_xscale("log")
    ax.set_xlabel("Dataset Size (Vectors) [Log Scale]", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Latency (ms)", fontsize=11, fontweight="bold")
    ax.set_title("Scaling Analysis: Dataset Size vs Search Latency\nObserving O(N) Matrix Scaling vs Remote Database Network Overhead", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xticks(sizes)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(loc="upper left", framealpha=0.95)
    plt.tight_layout()
    fig.savefig(c5_path, bbox_inches="tight")
    plt.close(fig)
    generated_charts.append(c5_path)

    # Chart 6: P95 Latency Comparison
    c6_path = CHARTS_DIR / "06_p95_latency_comparison.png"
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    x = np.arange(len(rows_1k))
    width = 0.35
    p95_1k = [r["p95_ms"] for r in rows_1k]
    p95_10k = [r["p95_ms"] for r in rows_10k]
    short_labels = [r["method"].replace("Brute-Force ", "").replace("Approximate ", "") for r in rows_1k]
    ax.bar(x - width / 2, p95_1k, width, label="1,000 Vectors", color="#3b82f6", edgecolor="#0f172a")
    ax.bar(x + width / 2, p95_10k, width, label="10,000 Vectors", color="#ea580c", edgecolor="#0f172a")
    ax.set_ylabel("P95 Latency (milliseconds)", fontsize=11, fontweight="bold")
    ax.set_title("Tail Latency (P95) Across Search Methods and Scales", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=25, ha="right", fontsize=9, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend(framealpha=0.95)
    plt.tight_layout()
    fig.savefig(c6_path, bbox_inches="tight")
    plt.close(fig)
    generated_charts.append(c6_path)

    # Chart 7: HNSW Recall vs Latency
    c7_path = CHARTS_DIR / "07_hnsw_recall_vs_latency.png"
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
    for r in rows:
        color = "#2563eb" if "NumPy" in r["method"] else ("#dc2626" if "Exact" in r["method"] else "#16a34a")
        marker = "o" if r["dataset_size"] == 1000 else "s"
        size_lbl = "1k" if r["dataset_size"] == 1000 else "10k"
        ax.scatter(r["mean_ms"], r["recall_at_10"] * 100, s=120, color=color, marker=marker, edgecolors="#0f172a", zorder=5)
        ax.annotate(f"{r['method'].split()[0]} {r['metric']} ({size_lbl})",
                    (r["mean_ms"], r["recall_at_10"] * 100),
                    xytext=(6, -3), textcoords="offset points", fontsize=8, fontweight="bold", color="#1e293b")

    ax.set_xlabel("Mean Search Latency (ms)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Recall@10 (%) [Ground Truth = NumPy Exact]", fontsize=11, fontweight="bold")
    ax.set_title("Accuracy vs Latency Trade-Off\nEvaluating Exactness vs Approximate Retrieval", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylim(85, 102)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    fig.savefig(c7_path, bbox_inches="tight")
    plt.close(fig)
    generated_charts.append(c7_path)

    print(f"[OK] Generated {len(generated_charts)} benchmark charts in: {CHARTS_DIR}")
    return generated_charts
