"""Unit tests for scratch mathematical functions and classroom example."""

import math
import pytest
import numpy as np

from src.vector_math import (
    dot_product,
    vector_magnitude,
    squared_euclidean_distance,
    euclidean_distance,
    cosine_similarity,
    l2_normalize_vector,
    np_euclidean_distance,
    np_cosine_similarity,
    np_dot_product,
    np_l2_normalize,
)


class TestClassroomMathematics:
    """Validate classroom example exact values and ranking behavior."""

    def test_classroom_exact_values(self):
        query = [2, 4]
        points = {
            "A": [2, 3],
            "B": [1, 2],
            "C": [0, 4],
        }

        # Point A (2, 3)
        assert math.isclose(euclidean_distance(query, points["A"]), 1.000000, abs_tol=1e-6)
        assert math.isclose(cosine_similarity(query, points["A"]), 0.9922778767136677, abs_tol=1e-6)

        # Point B (1, 2) - Collinear with (2, 4)
        assert math.isclose(euclidean_distance(query, points["B"]), math.sqrt(5), abs_tol=1e-6)
        assert math.isclose(cosine_similarity(query, points["B"]), 1.000000, abs_tol=1e-6)

        # Point C (0, 4)
        assert math.isclose(euclidean_distance(query, points["C"]), 2.000000, abs_tol=1e-6)
        assert math.isclose(cosine_similarity(query, points["C"]), 0.8944271909999159, abs_tol=1e-6)

    def test_classroom_rankings(self):
        query = [2, 4]
        points = {"A": [2, 3], "B": [1, 2], "C": [0, 4]}

        euclid_dists = {k: euclidean_distance(query, v) for k, v in points.items()}
        cosine_sims = {k: cosine_similarity(query, v) for k, v in points.items()}

        euclid_rank = sorted(euclid_dists.keys(), key=lambda k: euclid_dists[k])
        cosine_rank = sorted(cosine_sims.keys(), key=lambda k: -cosine_sims[k])

        assert euclid_rank == ["A", "C", "B"]
        assert cosine_rank == ["B", "A", "C"]


class TestVectorMathEdgeCases:
    """Test vector math behavior on mathematical edge cases."""

    def test_identical_vectors(self):
        v = [3.0, 4.0, 5.0]
        assert math.isclose(euclidean_distance(v, v), 0.0, abs_tol=1e-9)
        assert math.isclose(cosine_similarity(v, v), 1.0, abs_tol=1e-9)

    def test_orthogonal_vectors(self):
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        assert math.isclose(dot_product(v1, v2), 0.0, abs_tol=1e-9)
        assert math.isclose(cosine_similarity(v1, v2), 0.0, abs_tol=1e-9)
        assert math.isclose(euclidean_distance(v1, v2), math.sqrt(2), abs_tol=1e-6)

    def test_opposite_direction_vectors(self):
        v1 = [2.0, 3.0]
        v2 = [-2.0, -3.0]
        assert math.isclose(cosine_similarity(v1, v2), -1.0, abs_tol=1e-9)
        assert math.isclose(euclidean_distance(v1, v2), 2 * math.sqrt(13), abs_tol=1e-6)

    def test_zero_vector_handling(self):
        v_zero = [0.0, 0.0]
        v_normal = [1.0, 2.0]
        assert math.isclose(vector_magnitude(v_zero), 0.0, abs_tol=1e-9)
        assert math.isclose(dot_product(v_zero, v_normal), 0.0, abs_tol=1e-9)
        with pytest.raises(ValueError, match="undefined for zero-magnitude"):
            cosine_similarity(v_zero, v_normal)
        with pytest.raises(ValueError, match="Cannot normalize a zero vector"):
            l2_normalize_vector(v_zero)

    def test_l2_normalization_magnitude(self):
        v = [3.0, 4.0]
        v_norm = l2_normalize_vector(v)
        assert math.isclose(v_norm[0], 0.6, abs_tol=1e-6)
        assert math.isclose(v_norm[1], 0.8, abs_tol=1e-6)
        assert math.isclose(vector_magnitude(v_norm), 1.0, abs_tol=1e-6)

    def test_dimension_mismatch(self):
        with pytest.raises(ValueError, match="Vector dimension mismatch"):
            dot_product([1.0, 2.0], [1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Vector dimension mismatch"):
            euclidean_distance([1.0, 2.0], [1.0])

    def test_non_finite_values(self):
        with pytest.raises(ValueError, match="finite numerical values"):
            dot_product([float("nan"), 1.0], [1.0, 2.0])
        with pytest.raises(ValueError, match="finite numerical values"):
            vector_magnitude([float("inf"), 2.0])


class TestNumPyVectorizedMath:
    """Validate vectorized NumPy implementations against scratch math."""

    def test_numpy_matches_scratch(self):
        query = np.array([2.0, 4.0], dtype=np.float32)
        candidates = np.array([
            [2.0, 3.0],
            [1.0, 2.0],
            [0.0, 4.0]
        ], dtype=np.float32)

        # Euclidean
        np_euclid = np_euclidean_distance(query, candidates)
        scratch_euclid = [euclidean_distance(query.tolist(), c.tolist()) for c in candidates]
        assert np.allclose(np_euclid, scratch_euclid, atol=1e-5)

        # Dot Product
        np_dots = np_dot_product(query, candidates)
        scratch_dots = [dot_product(query.tolist(), c.tolist()) for c in candidates]
        assert np.allclose(np_dots, scratch_dots, atol=1e-5)

        # Cosine Similarity
        np_cos = np_cosine_similarity(query, candidates)
        scratch_cos = [cosine_similarity(query.tolist(), c.tolist()) for c in candidates]
        assert np.allclose(np_cos, scratch_cos, atol=1e-5)

    def test_numpy_normalization(self):
        matrix = np.array([
            [3.0, 4.0],
            [1.0, 2.0],
            [0.0, 5.0]
        ], dtype=np.float32)
        norm_matrix = np_l2_normalize(matrix)
        magnitudes = np.linalg.norm(norm_matrix, axis=1)
        assert np.allclose(magnitudes, 1.0, atol=1e-6)
