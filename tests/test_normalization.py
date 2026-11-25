"""Unit tests for Phase 6 Vector Normalization & Equivalence."""

import math
import pytest
import numpy as np

from src.normalization import demonstrate_normalization_on_classroom
from src.vector_math import np_l2_normalize, np_cosine_similarity, np_dot_product, np_euclidean_distance


class TestVectorNormalization:
    """Test normalization properties, dot product equivalence, and Euclidean monotonicity."""

    def test_classroom_normalization_proof(self):
        results = demonstrate_normalization_on_classroom()
        assert results["rankings"]["rankings_equivalent"] is True
        assert results["max_score_difference"] < 1e-6
        assert results["max_euclid_formula_difference"] < 1e-6

    def test_high_dimensional_random_normalization(self):
        np.random.seed(42)
        dim = 768  # Gemini Embedding dimension
        num_vecs = 50

        raw_vectors = np.random.randn(num_vecs, dim).astype(np.float32)
        raw_query = np.random.randn(dim).astype(np.float32)

        norm_vectors = np_l2_normalize(raw_vectors)
        norm_query = np_l2_normalize(raw_query)

        # Magnitudes must be 1.0
        assert np.allclose(np.linalg.norm(norm_vectors, axis=1), 1.0, atol=1e-6)
        assert math.isclose(np.linalg.norm(norm_query), 1.0, abs_tol=1e-6)

        # Cosine on raw == Dot on normalized
        cosine_sims = np_cosine_similarity(raw_query, raw_vectors)
        norm_dots = np_dot_product(norm_query, norm_vectors)
        assert np.allclose(cosine_sims, norm_dots, atol=1e-5)

        # Euclidean on normalized == sqrt(2 - 2 * cos)
        norm_euclid = np_euclidean_distance(norm_query, norm_vectors)
        theoretical_euclid = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * cosine_sims))
        assert np.allclose(norm_euclid, theoretical_euclid, atol=1e-5)

        # Rankings must match exactly
        cos_ranking = np.argsort(-cosine_sims)
        norm_dot_ranking = np.argsort(-norm_dots)
        norm_euclid_ranking = np.argsort(norm_euclid)

        assert np.array_equal(cos_ranking, norm_dot_ranking)
        assert np.array_equal(cos_ranking, norm_euclid_ranking)
