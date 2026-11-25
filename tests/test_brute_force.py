"""Unit tests for Phase 3 NumPy brute-force search engine."""

import math
import pytest
import numpy as np

from src.brute_force_search import (
    BruteForceVectorSearch,
    top_k_euclidean,
    top_k_cosine,
    top_k_dot_product,
)


class TestBruteForceSearch:
    """Test brute force search functionality, rankings, and edge cases."""

    @pytest.fixture
    def classroom_engine(self):
        vectors = np.array([
            [2.0, 3.0],  # ID 0: A
            [1.0, 2.0],  # ID 1: B
            [0.0, 4.0],  # ID 2: C
        ], dtype=np.float32)
        return BruteForceVectorSearch(vectors, ids=[0, 1, 2])

    def test_euclidean_ranking_classroom(self, classroom_engine):
        query = np.array([2.0, 4.0], dtype=np.float32)
        results = classroom_engine.search_euclidean(query, k=3)

        assert len(results) == 3
        # Expected: A (ID 0) -> C (ID 2) -> B (ID 1)
        assert results[0]["id"] == 0
        assert math.isclose(results[0]["score"], 1.0, abs_tol=1e-5)

        assert results[1]["id"] == 2
        assert math.isclose(results[1]["score"], 2.0, abs_tol=1e-5)

        assert results[2]["id"] == 1
        assert math.isclose(results[2]["score"], math.sqrt(5), abs_tol=1e-5)

    def test_cosine_ranking_classroom(self, classroom_engine):
        query = np.array([2.0, 4.0], dtype=np.float32)
        results = classroom_engine.search_cosine(query, k=3)

        assert len(results) == 3
        # Expected: B (ID 1) -> A (ID 0) -> C (ID 2)
        assert results[0]["id"] == 1
        assert math.isclose(results[0]["score"], 1.000000, abs_tol=1e-5)

        assert results[1]["id"] == 0
        assert math.isclose(results[1]["score"], 0.992278, abs_tol=1e-5)

        assert results[2]["id"] == 2
        assert math.isclose(results[2]["score"], 0.894427, abs_tol=1e-5)

    def test_deterministic_tie_breaking(self):
        # Two identical vectors with different IDs
        vectors = np.array([
            [1.0, 1.0],  # ID 10
            [1.0, 1.0],  # ID 5
        ], dtype=np.float32)
        engine = BruteForceVectorSearch(vectors, ids=[10, 5])
        query = np.array([1.0, 1.0], dtype=np.float32)

        results = engine.search_euclidean(query, k=2)
        # ID 5 should be ranked before ID 10 due to deterministic tie breaking
        assert results[0]["id"] == 5
        assert results[1]["id"] == 10

    def test_zero_vector_dataset(self):
        vectors = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
        ], dtype=np.float32)
        engine = BruteForceVectorSearch(vectors, ids=[0, 1])
        query = np.array([1.0, 0.0], dtype=np.float32)

        # Zero vector should have cosine similarity 0.0 safely without exception
        results = engine.search_cosine(query, k=2)
        assert results[0]["id"] == 1
        assert math.isclose(results[0]["score"], 1.0, abs_tol=1e-5)
        assert results[1]["id"] == 0
        assert math.isclose(results[1]["score"], 0.0, abs_tol=1e-5)

    def test_invalid_k_and_dimensions(self, classroom_engine):
        query = np.array([2.0, 4.0], dtype=np.float32)
        with pytest.raises(ValueError, match="k must be a positive integer"):
            classroom_engine.search_euclidean(query, k=0)
        with pytest.raises(ValueError, match="k must be a positive integer"):
            classroom_engine.search_euclidean(query, k=-5)

        wrong_dim_query = np.array([2.0, 4.0, 6.0], dtype=np.float32)
        with pytest.raises(ValueError, match="Query dimension"):
            classroom_engine.search_euclidean(wrong_dim_query, k=3)

    def test_nan_rejection(self):
        nan_vectors = np.array([[float("nan"), 1.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="Dataset contains non-finite"):
            BruteForceVectorSearch(nan_vectors)

    def test_convenience_functions(self):
        vectors = np.array([[2.0, 3.0], [1.0, 2.0]], dtype=np.float32)
        query = np.array([2.0, 4.0], dtype=np.float32)
        euc = top_k_euclidean(query, vectors, k=1)
        cos = top_k_cosine(query, vectors, k=1)
        dot = top_k_dot_product(query, vectors, k=1)
        assert euc[0]["id"] == 0
        assert cos[0]["id"] == 1
        assert dot[0]["id"] == 0  # (2*2 + 4*3) = 16 vs (2*1 + 4*2) = 10
