"""Core mathematical functions for vector similarity and distance calculations.

Implements pure Python mathematical operations from scratch without relying
on scikit-learn or SciPy distance metrics.
"""

import math
from typing import Sequence, Union, Tuple, List
import numpy as np


# ============================================================================
# Pure Python Scratch Implementations (Classroom & Foundational)
# ============================================================================

def dot_product(v1: Sequence[float], v2: Sequence[float]) -> float:
    """Calculate the dot product (scalar product) of two vectors from scratch.

    Formula:
        dot(v1, v2) = sum(v1[i] * v2[i]) for i in 0..n-1
    """
    if len(v1) != len(v2):
        raise ValueError(f"Vector dimension mismatch: {len(v1)} != {len(v2)}")
    if len(v1) == 0:
        raise ValueError("Cannot compute dot product of empty vectors.")
    
    total = 0.0
    for x, y in zip(v1, v2):
        if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
            raise ValueError("Vectors must contain finite numerical values.")
        total += float(x) * float(y)
    return total


def vector_magnitude(v: Sequence[float]) -> float:
    """Calculate the Euclidean L2 norm (magnitude/length) of a vector from scratch.

    Formula:
        ||v|| = sqrt(sum(v[i]²))
    """
    if len(v) == 0:
        raise ValueError("Cannot compute magnitude of empty vector.")
    
    sum_squares = 0.0
    for x in v:
        if math.isnan(x) or math.isinf(x):
            raise ValueError("Vector must contain finite numerical values.")
        sum_squares += float(x) * float(x)
    return math.sqrt(sum_squares)


def coordinate_differences(v1: Sequence[float], v2: Sequence[float]) -> List[float]:
    """Calculate coordinate-wise difference (v1[i] - v2[i]) for every dimension."""
    if len(v1) != len(v2):
        raise ValueError(f"Vector dimension mismatch: {len(v1)} != {len(v2)}")
    return [float(x) - float(y) for x, y in zip(v1, v2)]


def squared_euclidean_distance(v1: Sequence[float], v2: Sequence[float]) -> float:
    """Calculate squared Euclidean distance from scratch.

    Formula:
        d²(v1, v2) = sum((v1[i] - v2[i])²)
    """
    diffs = coordinate_differences(v1, v2)
    return sum(d * d for d in diffs)


def euclidean_distance(v1: Sequence[float], v2: Sequence[float]) -> float:
    """Calculate Euclidean distance (L2 distance) between two vectors from scratch.

    Formula:
        d(v1, v2) = sqrt(sum((v1[i] - v2[i])²))
    """
    return math.sqrt(squared_euclidean_distance(v1, v2))


def cosine_similarity(v1: Sequence[float], v2: Sequence[float]) -> float:
    """Calculate cosine similarity between two vectors from scratch.

    Formula:
        cosine_sim(v1, v2) = dot(v1, v2) / (||v1|| * ||v2||)
    """
    norm1 = vector_magnitude(v1)
    norm2 = vector_magnitude(v2)
    
    if norm1 == 0.0 or norm2 == 0.0:
        raise ValueError("Cosine similarity is undefined for zero-magnitude vectors.")
    
    dot = dot_product(v1, v2)
    cos_val = dot / (norm1 * norm2)
    
    # Clamp to [-1.0, 1.0] to guard against floating-point imprecision
    return max(-1.0, min(1.0, cos_val))


def l2_normalize_vector(v: Sequence[float]) -> List[float]:
    """Normalize a vector to unit length (L2 norm = 1.0) from scratch."""
    mag = vector_magnitude(v)
    if mag == 0.0:
        raise ValueError("Cannot normalize a zero vector.")
    return [float(x) / mag for x in v]


# ============================================================================
# Vectorized NumPy Implementations (High Performance Brute Force Search)
# ============================================================================

def np_euclidean_distance(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Calculate Euclidean distance between 1D query and 2D matrix of vectors using NumPy.

    Parameters:
        query: 1D array of shape (D,)
        vectors: 2D array of shape (N, D)

    Returns:
        1D array of distances of shape (N,)
    """
    _validate_numpy_inputs(query, vectors)
    diff = vectors - query
    return np.sqrt(np.sum(diff ** 2, axis=1))


def np_squared_euclidean_distance(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Calculate squared Euclidean distance between 1D query and 2D matrix of vectors."""
    _validate_numpy_inputs(query, vectors)
    diff = vectors - query
    return np.sum(diff ** 2, axis=1)


def np_dot_product(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Calculate dot product between 1D query and 2D matrix of vectors.

    Formula:
        scores = vectors @ query
    """
    _validate_numpy_inputs(query, vectors)
    return np.dot(vectors, query)


def np_cosine_similarity(
    query: np.ndarray,
    vectors: np.ndarray,
    vector_norms: np.ndarray = None
) -> np.ndarray:
    """Calculate cosine similarity between 1D query and 2D vectors using NumPy.

    Parameters:
        query: 1D array of shape (D,)
        vectors: 2D array of shape (N, D)
        vector_norms: Optional precomputed L2 norms of vectors (shape (N,))

    Returns:
        1D array of cosine similarities of shape (N,)
    """
    _validate_numpy_inputs(query, vectors)
    
    query_norm = np.linalg.norm(query)
    if query_norm == 0.0:
        raise ValueError("Query vector has zero norm; cosine similarity undefined.")
        
    if vector_norms is None:
        vector_norms = np.linalg.norm(vectors, axis=1)
        
    if np.any(vector_norms == 0.0):
        # Explicit handling of zero norm vectors: set their similarity to 0.0
        zero_mask = (vector_norms == 0.0)
        safe_norms = np.where(zero_mask, 1.0, vector_norms)
        dots = np.dot(vectors, query)
        sims = dots / (safe_norms * query_norm)
        sims[zero_mask] = 0.0
    else:
        dots = np.dot(vectors, query)
        sims = dots / (vector_norms * query_norm)
        
    return np.clip(sims, -1.0, 1.0)


def np_l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2 normalize a 1D vector or 2D matrix of vectors along axis=-1."""
    if not np.all(np.isfinite(vectors)):
        raise ValueError("Vectors contain NaN or Inf values.")
        
    if vectors.ndim == 1:
        norm = np.linalg.norm(vectors)
        if norm == 0.0:
            raise ValueError("Cannot normalize a zero vector.")
        return vectors / norm
    elif vectors.ndim == 2:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero for zero vectors
        zero_mask = (norms == 0.0)
        safe_norms = np.where(zero_mask, 1.0, norms)
        normalized = vectors / safe_norms
        normalized[zero_mask.squeeze(axis=-1)] = 0.0
        return normalized
    else:
        raise ValueError(f"Expected 1D or 2D array, got shape {vectors.shape}")


def _validate_numpy_inputs(query: np.ndarray, vectors: np.ndarray) -> None:
    """Validate query and vectors shape, dimensions, and finite numbers."""
    if not isinstance(query, np.ndarray) or not isinstance(vectors, np.ndarray):
        raise TypeError("query and vectors must be numpy ndarrays.")
    if vectors.size == 0 or query.size == 0:
        raise ValueError("Empty vectors or query provided.")
    if query.ndim != 1:
        raise ValueError(f"Query must be 1D array of shape (D,), got shape {query.shape}")
    if vectors.ndim != 2:
        raise ValueError(f"Vectors must be 2D array of shape (N, D), got shape {vectors.shape}")
    if vectors.shape[1] != query.shape[0]:
        raise ValueError(f"Dimension mismatch: vectors {vectors.shape[1]} vs query {query.shape[0]}")
    if not np.all(np.isfinite(query)):
        raise ValueError("Query contains NaN or Inf values.")
    if not np.all(np.isfinite(vectors)):
        raise ValueError("Vectors contain NaN or Inf values.")
