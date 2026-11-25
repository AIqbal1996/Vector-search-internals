"""Phase 1: Classroom Mathematical Example.

Points:
    A: [2, 3]
    B: [1, 2]
    C: [0, 4]
Query:
    Q: [2, 4]

Demonstrates step-by-step intermediate calculations for Euclidean distance
and Cosine similarity from scratch, and explains why their rankings differ.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from tabulate import tabulate

from src.vector_math import (
    dot_product,
    vector_magnitude,
    coordinate_differences,
    squared_euclidean_distance,
    euclidean_distance,
    cosine_similarity,
)
from config.settings import RESULTS_DIR


# Define classroom points
POINTS = {
    "A": [2, 3],
    "B": [1, 2],
    "C": [0, 4],
}
QUERY = [2, 4]


def run_classroom_calculations() -> Dict[str, Any]:
    """Execute complete classroom calculations and print intermediate steps."""
    print("=" * 80)
    print("PHASE 1: CLASSROOM MATHEMATICAL EXAMPLE (FROM SCRATCH)")
    print("=" * 80)
    print(f"Query Vector Q: {QUERY}")
    query_mag = vector_magnitude(QUERY)
    print(f"Query Vector Magnitude ||Q|| = sqrt(2² + 4²) = sqrt(20) = {query_mag:.6f}")
    print("-" * 80)

    detailed_results = {}
    table_rows = []

    for name, point in POINTS.items():
        # Step-by-step intermediate calculations
        point_mag = vector_magnitude(point)
        coord_diff = coordinate_differences(QUERY, point)
        sq_diff = [d ** 2 for d in coord_diff]
        sq_euclid = squared_euclidean_distance(QUERY, point)
        euclid_dist = euclidean_distance(QUERY, point)
        dot_prod = dot_product(QUERY, point)
        cos_sim = cosine_similarity(QUERY, point)

        detailed_results[name] = {
            "coordinates": point,
            "magnitude": point_mag,
            "coord_difference": coord_diff,
            "squared_differences": sq_diff,
            "squared_euclidean_distance": sq_euclid,
            "euclidean_distance": euclid_dist,
            "dot_product": dot_prod,
            "cosine_similarity": cos_sim,
        }

        table_rows.append([
            f"{name} {point}",
            f"{point_mag:.6f}",
            f"{coord_diff}",
            f"{sq_diff}",
            f"{dot_prod:.1f}",
            f"{euclid_dist:.6f}",
            f"{cos_sim:.6f}",
        ])

    headers = [
        "Point",
        "Magnitude",
        "Diff (Q - V)",
        "Sq Diff",
        "Dot Product",
        "Euclidean Dist",
        "Cosine Sim",
    ]
    print(tabulate(table_rows, headers=headers, tablefmt="grid"))
    print()

    # Determine Rankings
    # Euclidean: smaller distance is better (ascending)
    euclidean_ranking = sorted(
        POINTS.keys(),
        key=lambda k: (detailed_results[k]["euclidean_distance"], k)
    )

    # Cosine: higher similarity is better (descending)
    cosine_ranking = sorted(
        POINTS.keys(),
        key=lambda k: (-detailed_results[k]["cosine_similarity"], k)
    )

    print("=" * 80)
    print("RANKING RESULTS:")
    print("=" * 80)
    print(f"1. Euclidean Distance Ranking (Ascending - Closest in Space):")
    for rank, name in enumerate(euclidean_ranking, start=1):
        print(f"   Rank {rank}: Point {name} {POINTS[name]} -> Distance = {detailed_results[name]['euclidean_distance']:.6f}")
    print(f"   -> Ranking Order: {' -> '.join(euclidean_ranking)}")

    print()
    print(f"2. Cosine Similarity Ranking (Descending - Most Aligned in Direction):")
    for rank, name in enumerate(cosine_ranking, start=1):
        print(f"   Rank {rank}: Point {name} {POINTS[name]} -> Cosine Sim = {detailed_results[name]['cosine_similarity']:.6f}")
    print(f"   -> Ranking Order: {' -> '.join(cosine_ranking)}")

    # Detailed Explanation of Ranking Divergence
    explanation = (
        "WHY DO THE RANKINGS DIFFER?\n"
        "1. Euclidean Distance measures STRAIGHT-LINE GEOMETRIC DISTANCE in space.\n"
        "   - Point A (2, 3) differs from Query (2, 4) by only 1 unit along the y-axis.\n"
        "     Distance = sqrt((2-2)² + (4-3)²) = sqrt(1) = 1.000000. It is physically closest!\n"
        "   - Point C (0, 4) differs by 2 units along the x-axis: sqrt((2-0)² + (4-4)²) = 2.000000.\n"
        "   - Point B (1, 2) has Euclidean distance sqrt((2-1)² + (4-2)²) = sqrt(5) ≈ 2.236068, making it physically furthest.\n"
        "\n"
        "2. Cosine Similarity measures THE ANGLE (DIRECTION) between vectors, ignoring magnitude.\n"
        "   - Notice that Query (2, 4) is an exact scalar multiple of Point B (1, 2): (2, 4) = 2 * (1, 2).\n"
        "   - The angle θ between vector B and Query Q is exactly 0°! cos(0°) = 1.000000.\n"
        "   - Because Cosine Similarity measures directional alignment, Point B achieves a PERFECT score of 1.000000,\n"
        "     even though Point B is the furthest in Euclidean distance!\n"
        "   - Point A (2, 3) is at a small angle θ ≈ 7.125° from (2, 4), yielding cos(θ) ≈ 0.992278.\n"
        "   - Point C (0, 4) lies along the y-axis at θ ≈ 26.565° from (2, 4), yielding cos(θ) ≈ 0.894427.\n"
        "\n"
        "KEY TAKEAWAY:\n"
        "- Use Euclidean distance when the magnitude of values represents physical scale, intensity, or magnitude.\n"
        "- Use Cosine similarity when semantic direction matters regardless of document/vector length."
    )
    print("=" * 80)
    print(explanation)
    print("=" * 80)

    # Prepare Output Payload
    output_payload = {
        "query": QUERY,
        "query_magnitude": query_mag,
        "points": detailed_results,
        "rankings": {
            "euclidean": {
                "order": euclidean_ranking,
                "ranking_string": " -> ".join(euclidean_ranking),
                "expected": ["A", "C", "B"],
                "matches_expected": euclidean_ranking == ["A", "C", "B"],
            },
            "cosine": {
                "order": cosine_ranking,
                "ranking_string": " -> ".join(cosine_ranking),
                "expected": ["B", "A", "C"],
                "matches_expected": cosine_ranking == ["B", "A", "C"],
            },
        },
        "explanation": explanation,
    }

    # Save to classroom_results.json
    results_path = RESULTS_DIR / "classroom_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\n[OK] Results successfully saved to: {results_path}")

    return output_payload


if __name__ == "__main__":
    run_classroom_calculations()
