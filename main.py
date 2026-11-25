"""Master Execution CLI for Week 5 Vector Search Internals.

Usage:
    python main.py --stage classroom
    python main.py --stage embeddings
    python main.py --stage numpy
    python main.py --stage qdrant
    python main.py --stage normalization
    python main.py --stage benchmark
    python main.py --stage all
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))


def run_stage_classroom():
    from scripts.run_classroom_example import main as classroom_main
    classroom_main()


def run_stage_embeddings():
    from scripts.generate_embeddings import main as embeddings_main
    embeddings_main()


def run_stage_numpy():
    from scripts.run_numpy_search import main as numpy_main
    numpy_main()


def run_stage_qdrant():
    from scripts.run_qdrant_search import main as qdrant_main
    from scripts.run_validation import main as validation_main
    qdrant_main()
    validation_main()


def run_stage_normalization():
    from src.normalization import demonstrate_normalization_on_classroom
    demonstrate_normalization_on_classroom()


def run_stage_benchmark():
    from scripts.run_benchmark import main as benchmark_main
    from src.visualization import generate_all_benchmark_charts
    benchmark_main()
    print("\nGenerating Benchmark Visualizations...")
    generate_all_benchmark_charts()


def run_stage_all():
    print("=" * 80)
    print("EXECUTING COMPLETE END-TO-END ASSIGNMENT WORKFLOW")
    print("=" * 80)
    
    print("\n>>> STAGE 1: CLASSROOM MATHEMATICAL EXAMPLE <<<")
    run_stage_classroom()

    print("\n>>> STAGE 2: DATASET & EMBEDDING GENERATION <<<")
    run_stage_embeddings()

    print("\n>>> STAGE 3: NUMPY BRUTE-FORCE SEARCH <<<")
    run_stage_numpy()

    print("\n>>> STAGE 4 & 5: QDRANT SEARCH & VALIDATION <<<")
    run_stage_qdrant()

    print("\n>>> STAGE 6: VECTOR NORMALIZATION PROOF <<<")
    run_stage_normalization()

    print("\n>>> STAGE 7: PERFORMANCE BENCHMARKING <<<")
    run_stage_benchmark()

    print("\n" + "=" * 80)
    print("[ALL STAGES COMPLETED SUCCESSFULLY]")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Week 5: Vector Search Internals — Master Execution CLI",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["classroom", "embeddings", "numpy", "qdrant", "normalization", "benchmark", "all"],
        help=(
            "Stage to execute:\n"
            "  classroom     : Phase 1 classroom 2D manual mathematical calculations & plot\n"
            "  embeddings    : Phase 2 Gemini Embedding 2 generation via EURI API\n"
            "  numpy         : Phase 3 NumPy brute-force search across 10 queries\n"
            "  qdrant        : Phase 4 & 5 Qdrant multi-metric search & exact validation\n"
            "  normalization : Phase 6 L2 normalization & cosine-dot equivalence proof\n"
            "  benchmark     : Phase 7 1k vs 10k vector search performance benchmark\n"
            "  all           : Run complete workflow end-to-end\n"
        )
    )

    args = parser.parse_args()

    stages = {
        "classroom": run_stage_classroom,
        "embeddings": run_stage_embeddings,
        "numpy": run_stage_numpy,
        "qdrant": run_stage_qdrant,
        "normalization": run_stage_normalization,
        "benchmark": run_stage_benchmark,
        "all": run_stage_all,
    }

    stages[args.stage]()


if __name__ == "__main__":
    main()
