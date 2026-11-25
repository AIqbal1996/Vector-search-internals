"""Script to execute Phase 1 classroom example and generate figures."""

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.classroom_example import run_classroom_calculations
from src.visualization import plot_classroom_2d_vectors, plot_ranking_comparison


def main():
    print("Executing Phase 1: Classroom Mathematical Example...")
    results = run_classroom_calculations()
    
    print("\nGenerating Phase 1 Academic Visualizations...")
    chart1 = plot_classroom_2d_vectors(results)
    chart2 = plot_ranking_comparison(results)
    
    print("\n[SUCCESS] Phase 1 completed successfully.")
    print(f"Results: results/classroom_results.json")
    print(f"Chart 1: {chart1}")
    print(f"Chart 2: {chart2}")


if __name__ == "__main__":
    main()
