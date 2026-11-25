"""Phase 3: NumPy Brute-Force Vector Search Engine.

Implements vectorized nearest neighbor search from scratch using NumPy:
- Euclidean Distance (ascending order - nearest first)
- Cosine Similarity (descending order - highest similarity first)
- Dot Product Similarity (descending order - highest dot product first)

Features:
- Deterministic ID-based tie breaking
- Explicit handling of zero-norm vectors
- Strict dimension, k, finite-value, and empty-dataset validation
- Configurable top-k selection via partial sort (np.argpartition) or full sort
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from src.vector_math import (
    np_euclidean_distance,
    np_cosine_similarity,
    np_dot_product,
)


class BruteForceVectorSearch:
    """Vectorized brute-force search engine over NumPy matrices."""

    def __init__(
        self,
        vectors: np.ndarray,
        ids: Optional[List[int]] = None,
        use_argpartition: bool = True
    ):
        """Initialize the brute-force search engine.

        Parameters:
            vectors: 2D array of shape (N, D) representing dataset embeddings
            ids: Optional list of integer IDs (defaults to 0..N-1)
            use_argpartition: If True, uses O(N) np.argpartition + O(k log k) sort.
                              If False, performs full O(N log N) sort.
        """
        self._validate_dataset(vectors)
        self.vectors = vectors.astype(np.float32)
        self.num_vectors, self.dimension = self.vectors.shape

        if ids is None:
            self.ids = np.arange(self.num_vectors, dtype=np.int64)
        else:
            if len(ids) != self.num_vectors:
                raise ValueError(f"IDs count ({len(ids)}) does not match vectors count ({self.num_vectors}).")
            self.ids = np.array(ids, dtype=np.int64)

        # Precompute L2 norms for fast cosine similarity evaluation
        self.vector_norms = np.linalg.norm(self.vectors, axis=1)
        self.use_argpartition = use_argpartition

    @staticmethod
    def _validate_dataset(vectors: np.ndarray) -> None:
        if not isinstance(vectors, np.ndarray):
            raise TypeError("vectors must be a numpy.ndarray.")
        if vectors.size == 0:
            raise ValueError("Dataset cannot be empty.")
        if vectors.ndim != 2:
            raise ValueError(f"Vectors must be 2D array of shape (N, D), got shape {vectors.shape}")
        if not np.all(np.isfinite(vectors)):
            raise ValueError("Dataset contains non-finite values (NaN or Inf).")

    def _validate_query_and_k(self, query: np.ndarray, k: int) -> int:
        if not isinstance(query, np.ndarray):
            raise TypeError("query must be a numpy.ndarray.")
        if query.ndim != 1:
            raise ValueError(f"Query must be 1D array of shape (D,), got shape {query.shape}")
        if query.shape[0] != self.dimension:
            raise ValueError(f"Query dimension {query.shape[0]} does not match dataset dimension {self.dimension}.")
        if not np.all(np.isfinite(query)):
            raise ValueError("Query contains non-finite values (NaN or Inf).")
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")
        return min(k, self.num_vectors)

    def _rank_scores(
        self,
        scores: np.ndarray,
        k: int,
        ascending: bool
    ) -> List[Dict[str, Any]]:
        """Rank scores with deterministic tie-breaking by ID (ascending).

        Parameters:
            scores: 1D array of scores (distances or similarities)
            k: number of top results to return
            ascending: True for Euclidean distance, False for Cosine / Dot product
        """
        n = self.num_vectors
        effective_k = min(k, n)

        # In numpy, structured arrays enable compound sorting:
        # primary key: score (ascending or descending), secondary key: id (ascending)
        if self.use_argpartition and effective_k < n:
            # Step 1: O(N) partial selection of top effective_k elements
            if ascending:
                candidate_indices = np.argpartition(scores, effective_k)[:effective_k]
            else:
                candidate_indices = np.argpartition(-scores, effective_k)[:effective_k]

            candidate_scores = scores[candidate_indices]
            candidate_ids = self.ids[candidate_indices]

            # Step 2: Sort only the top effective_k candidates with tie-breaking
            if ascending:
                order = np.lexsort((candidate_ids, candidate_scores))
            else:
                order = np.lexsort((candidate_ids, -candidate_scores))

            selected_indices = candidate_indices[order]
        else:
            # Full O(N log N) lexicographical sort
            if ascending:
                order = np.lexsort((self.ids, scores))
            else:
                order = np.lexsort((self.ids, -scores))
            selected_indices = order[:effective_k]

        results = []
        for rank, idx in enumerate(selected_indices, start=1):
            results.append({
                "rank": rank,
                "id": int(self.ids[idx]),
                "score": float(scores[idx]),
            })
        return results

    def search_euclidean(self, query: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search nearest neighbors by Euclidean distance (ascending rank)."""
        k = self._validate_query_and_k(query, k)
        scores = np_euclidean_distance(query.astype(np.float32), self.vectors)
        return self._rank_scores(scores, k, ascending=True)

    def search_cosine(self, query: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search nearest neighbors by Cosine similarity (descending rank)."""
        k = self._validate_query_and_k(query, k)
        scores = np_cosine_similarity(
            query.astype(np.float32),
            self.vectors,
            vector_norms=self.vector_norms
        )
        return self._rank_scores(scores, k, ascending=False)

    def search_dot_product(self, query: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search nearest neighbors by Dot product similarity (descending rank)."""
        k = self._validate_query_and_k(query, k)
        scores = np_dot_product(query.astype(np.float32), self.vectors)
        return self._rank_scores(scores, k, ascending=False)


# Standalone function wrappers as requested in Section 8 specification
def euclidean_distance(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Wrapper: calculate Euclidean distance between query and vectors."""
    return np_euclidean_distance(query, vectors)


def cosine_similarity(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Wrapper: calculate Cosine similarity between query and vectors."""
    return np_cosine_similarity(query, vectors)


def dot_product_similarity(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Wrapper: calculate Dot product similarity between query and vectors."""
    return np_dot_product(query, vectors)


def top_k_euclidean(query: np.ndarray, vectors: np.ndarray, k: int) -> List[Dict[str, Any]]:
    """Return top-k nearest neighbors by Euclidean distance (ascending)."""
    engine = BruteForceVectorSearch(vectors)
    return engine.search_euclidean(query, k=k)


def top_k_cosine(query: np.ndarray, vectors: np.ndarray, k: int) -> List[Dict[str, Any]]:
    """Return top-k nearest neighbors by Cosine similarity (descending)."""
    engine = BruteForceVectorSearch(vectors)
    return engine.search_cosine(query, k=k)


def top_k_dot_product(query: np.ndarray, vectors: np.ndarray, k: int) -> List[Dict[str, Any]]:
    """Return top-k nearest neighbors by Dot product similarity (descending)."""
    engine = BruteForceVectorSearch(vectors)
    return engine.search_dot_product(query, k=k)
