"""Phase 6: Vector Normalization & Equivalence Proof.

Demonstrates and mathematically proves that:
1. Cosine similarity of arbitrary vectors equals the Dot Product of their L2-normalized versions.
2. For unit vectors (||a|| = 1, ||b|| = 1):
      cosine_similarity(a, b) = dot_product(a, b)
3. Squared Euclidean distance and Cosine similarity have an exact monotonic relationship on unit vectors:
      ||a - b||² = ||a||² + ||b||² - 2(a · b) = 1 + 1 - 2*cos(a, b) = 2 - 2*cos(a, b)
      d_euclidean(a, b) = sqrt(2 - 2 * cosine_similarity(a, b))
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

from src.vector_math import np_l2_normalize, np_cosine_similarity, np_dot_product, np_euclidean_distance
from src.brute_force_search import BruteForceVectorSearch
from config.settings import RESULTS_DIR, CHARTS_DIR


def demonstrate_normalization_on_classroom() -> Dict[str, Any]:
    """Demonstrate L2 normalization and cosine-dot equivalence on classroom 2D vectors."""
    print("=" * 80)
    print("PHASE 6: VECTOR NORMALIZATION PROOF (CLASSROOM 2D VECTORS)")
    print("=" * 80)

    # 1. Classroom Vectors
    names = ["A", "B", "C"]
    raw_vectors = np.array([
        [2.0, 3.0],  # A
        [1.0, 2.0],  # B
        [0.0, 4.0],  # C
    ], dtype=np.float32)
    raw_query = np.array([2.0, 4.0], dtype=np.float32)

    # 2. Magnitudes
    raw_mags = np.linalg.norm(raw_vectors, axis=1)
    raw_query_mag = np.linalg.norm(raw_query)

    # 3. L2 Normalization: v / ||v||
    norm_vectors = np_l2_normalize(raw_vectors)
    norm_query = np_l2_normalize(raw_query)

    norm_mags = np.linalg.norm(norm_vectors, axis=1)
    norm_query_mag = np.linalg.norm(norm_query)

    print(f"Raw Query Q: {raw_query.tolist()} | Magnitude: {raw_query_mag:.6f}")
    print(f"Normalized Query Q_hat: {norm_query.tolist()} | Magnitude: {norm_query_mag:.6f}\n")

    # 4. Compute Scores:
    # - Cosine similarity on raw vectors
    cosine_sims = np_cosine_similarity(raw_query, raw_vectors)
    # - Raw dot product on unnormalized vectors
    raw_dots = np_dot_product(raw_query, raw_vectors)
    # - Dot product on normalized vectors
    norm_dots = np_dot_product(norm_query, norm_vectors)
    # - Euclidean distance on normalized vectors
    norm_euclid = np_euclidean_distance(norm_query, norm_vectors)
    # - Theoretical Euclidean from Cosine: sqrt(2 - 2 * cos)
    theoretical_euclid = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * cosine_sims))

    table_data = []
    for i, name in enumerate(names):
        table_data.append([
            name,
            raw_vectors[i].tolist(),
            f"{raw_mags[i]:.4f}",
            norm_vectors[i].tolist(),
            f"{norm_mags[i]:.4f}",
            f"{raw_dots[i]:.2f}",
            f"{cosine_sims[i]:.6f}",
            f"{norm_dots[i]:.6f}",
            f"{abs(cosine_sims[i] - norm_dots[i]):.2e}",
            f"{norm_euclid[i]:.6f}",
            f"{theoretical_euclid[i]:.6f}",
        ])

    headers = [
        "Point", "Raw Coords", "Raw Mag", "Norm Coords", "Norm Mag",
        "Raw Dot", "Cosine Sim", "Norm Dot", "Abs Diff (Cos vs NormDot)",
        "Norm Euclid", "sqrt(2 - 2*Cos)"
    ]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print()

    # 5. Rankings
    # Raw Dot Ranking (descending)
    raw_dot_rank = [names[i] for i in np.argsort(-raw_dots)]
    # Cosine Ranking (descending)
    cosine_rank = [names[i] for i in np.argsort(-cosine_sims)]
    # Normalized Dot Ranking (descending)
    norm_dot_rank = [names[i] for i in np.argsort(-norm_dots)]
    # Normalized Euclidean Ranking (ascending)
    norm_euclid_rank = [names[i] for i in np.argsort(norm_euclid)]

    print(f"Raw Dot Product Ranking:        {' -> '.join(raw_dot_rank)}  (biased by large magnitude)")
    print(f"Cosine Similarity Ranking:      {' -> '.join(cosine_rank)}")
    print(f"Normalized Dot Product Ranking: {' -> '.join(norm_dot_rank)}  (EXACT MATCH with Cosine!)")
    print(f"Normalized Euclidean Ranking:   {' -> '.join(norm_euclid_rank)}  (EXACT MATCH with Cosine!)")

    # Mathematical Verification
    max_score_diff = float(np.max(np.abs(cosine_sims - norm_dots)))
    max_euclid_formula_diff = float(np.max(np.abs(norm_euclid - theoretical_euclid)))
    rankings_match = (cosine_rank == norm_dot_rank == norm_euclid_rank)

    assert max_score_diff < 1e-6, f"Cosine and Normalized Dot differ by {max_score_diff}"
    assert max_euclid_formula_diff < 1e-6, f"Normalized Euclid formula mismatch: {max_euclid_formula_diff}"
    assert rankings_match, "Rankings did not match after normalization!"

    print("\n" + "=" * 80)
    print("MATHEMATICAL PROOF SUMMARY:")
    print("=" * 80)
    print("1. Cosine similarity formula:")
    print("   cosine_sim(a, b) = (a · b) / (||a|| * ||b||)")
    print("2. When a and b are L2-normalized: ||a|| = 1 and ||b|| = 1:")
    print("   cosine_sim(a_hat, b_hat) = (a_hat · b_hat) / (1 * 1) = a_hat · b_hat = dot(a_hat, b_hat)")
    print("3. Relationship between Euclidean distance and Cosine similarity for unit vectors:")
    print("   ||a_hat - b_hat||² = ||a_hat||² + ||b_hat||² - 2(a_hat · b_hat)")
    print("                      = 1 + 1 - 2*cos(a, b) = 2 - 2*cos(a, b)")
    print("   Therefore, d_euclid(a_hat, b_hat) = sqrt(2 - 2 * cos(a, b))")
    print("   Because sqrt(2 - 2x) is strictly monotonically decreasing for x in [-1, 1],")
    print("   minimizing Euclidean distance on unit vectors is MATHEMATICALLY IDENTICAL")
    print("   to maximizing Cosine similarity! Both yield the identical ranking order: B -> A -> C.")
    print("=" * 80)

    results_dict = {
        "dataset_type": "classroom_2d",
        "raw_query": raw_query.tolist(),
        "normalized_query": norm_query.tolist(),
        "points": {
            names[i]: {
                "raw_coords": raw_vectors[i].tolist(),
                "raw_magnitude": float(raw_mags[i]),
                "normalized_coords": norm_vectors[i].tolist(),
                "normalized_magnitude": float(norm_mags[i]),
                "raw_dot_product": float(raw_dots[i]),
                "cosine_similarity": float(cosine_sims[i]),
                "normalized_dot_product": float(norm_dots[i]),
                "normalized_euclidean": float(norm_euclid[i]),
                "theoretical_euclidean": float(theoretical_euclid[i]),
            } for i in range(len(names))
        },
        "rankings": {
            "raw_dot": raw_dot_rank,
            "cosine": cosine_rank,
            "normalized_dot": norm_dot_rank,
            "normalized_euclidean": norm_euclid_rank,
            "rankings_equivalent": rankings_match,
        },
        "max_score_difference": max_score_diff,
        "max_euclid_formula_difference": max_euclid_formula_diff,
    }

    # Save results
    save_path = RESULTS_DIR / "normalization_results.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)
    print(f"[OK] Normalization results saved to: {save_path}")

    # Generate academic visualization
    plot_normalization_proof(results_dict)

    return results_dict


def plot_normalization_proof(results: Dict[str, Any], save_path: Path = None) -> Path:
    """Generate professional chart illustrating vector normalization and unit sphere projection."""
    if save_path is None:
        save_path = CHARTS_DIR / "08_vector_normalization.png"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), dpi=300)

    # Subplot 1: Geometric Unit Circle Projection
    ax1.set_facecolor("#f8f9fa")
    ax1.grid(True, linestyle="--", alpha=0.6, color="#cbd5e1")
    ax1.axhline(0, color="#94a3b8", linewidth=1.2)
    ax1.axvline(0, color="#94a3b8", linewidth=1.2)

    # Unit circle
    circle = plt.Circle((0, 0), 1.0, color="#64748b", fill=False, linestyle="--", linewidth=1.5, label="Unit Circle (||v|| = 1)")
    ax1.add_patch(circle)

    colors = {"A": "#2563eb", "B": "#16a34a", "C": "#ea580c"}
    pts = results["points"]

    # Raw Query & Normalized Query
    q_raw = results["raw_query"]
    q_norm = results["normalized_query"]
    ax1.scatter([q_raw[0]], [q_raw[1]], color="#dc2626", s=110, label=f"Raw Query ({q_raw[0]}, {q_raw[1]})", zorder=5)
    ax1.scatter([q_norm[0]], [q_norm[1]], color="#dc2626", marker="X", s=130, label=f"Norm Query ({q_norm[0]:.2f}, {q_norm[1]:.2f})", zorder=6)
    # Projection line
    ax1.plot([0, q_raw[0]], [0, q_raw[1]], color="#dc2626", linestyle=":", alpha=0.7)

    for name, data in pts.items():
        raw = data["raw_coords"]
        norm = data["normalized_coords"]
        col = colors[name]
        ax1.scatter([raw[0]], [raw[1]], color=col, s=90, label=f"Raw Point {name}", zorder=5)
        ax1.scatter([norm[0]], [norm[1]], color=col, marker="X", s=110, label=f"Norm Point {name}", zorder=6)
        # Ray from origin to raw showing normalization project onto circle
        ax1.plot([0, raw[0]], [0, raw[1]], color=col, linestyle=":", alpha=0.6)

    ax1.set_xlim(-0.5, 2.5)
    ax1.set_ylim(-0.5, 4.5)
    ax1.set_aspect("equal")
    ax1.set_title("Geometric L2 Normalization onto Unit Circle", fontweight="bold", fontsize=11)
    ax1.set_xlabel("X-coordinate", fontsize=10)
    ax1.set_ylabel("Y-coordinate", fontsize=10)
    ax1.legend(loc="upper left", fontsize=7.5, framealpha=0.9)

    # Subplot 2: Score Equivalence Bar Chart
    names = list(pts.keys())
    x = np.arange(len(names))
    width = 0.25

    cos_scores = [pts[n]["cosine_similarity"] for n in names]
    norm_dot_scores = [pts[n]["normalized_dot_product"] for n in names]
    raw_dot_scores = [pts[n]["raw_dot_product"] / 20.0 for n in names]  # Scaled for visual comparison

    rects1 = ax2.bar(x - width, cos_scores, width, label="Cosine Sim (Raw Vectors)", color="#2563eb", edgecolor="#0f172a")
    rects2 = ax2.bar(x, norm_dot_scores, width, label="Dot Product (Normalized Vectors)", color="#10b981", edgecolor="#0f172a", hatch="//")
    rects3 = ax2.bar(x + width, raw_dot_scores, width, label="Raw Dot Product (Scaled /20)", color="#f59e0b", edgecolor="#0f172a")

    ax2.set_ylabel("Score", fontsize=10)
    ax2.set_title("Equivalence: Cosine Similarity == Normalized Dot Product", fontweight="bold", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Point {n}" for n in names], fontweight="bold")
    ax2.set_ylim(0.7, 1.05)
    ax2.grid(axis="y", linestyle="--", alpha=0.6)
    ax2.legend(loc="lower right", fontsize=8, framealpha=0.95)

    # Value labels
    for rect in rects2:
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width() / 2, h + 0.005, f"{h:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Normalization proof chart saved to: {save_path}")
    return save_path


if __name__ == "__main__":
    demonstrate_normalization_on_classroom()
