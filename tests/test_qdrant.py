"""Integration and unit tests for Phase 4 & 5 Qdrant integration."""

import pytest
import numpy as np
from config.settings import settings
from src.qdrant_manager import QdrantManager


class TestQdrantIntegration:
    """Test Qdrant collection management and exact search."""

    @pytest.fixture
    def qdrant(self):
        manager = QdrantManager()
        try:
            manager.verify_connection()
            return manager
        except Exception as e:
            pytest.skip(f"Qdrant endpoint not accessible ({e}); skipping live Qdrant test.")

    def test_connection_verified(self, qdrant):
        assert qdrant.verify_connection() is True

    def test_collection_existence_and_metrics(self, qdrant):
        cols = [c.name for c in qdrant.client.get_collections().collections]
        assert settings.collection_cosine in cols
        assert settings.collection_euclidean in cols
        assert settings.collection_dot in cols

    def test_exact_search_result_structure(self, qdrant):
        dummy_query = np.zeros(settings.embedding_dimension, dtype=np.float32)
        dummy_query[0] = 1.0

        results = qdrant.search(
            collection_name=settings.collection_cosine,
            query_vector=dummy_query,
            k=3,
            exact=True
        )
        assert len(results) <= 3
        if len(results) > 0:
            top = results[0]
            assert "rank" in top
            assert "id" in top
            assert "score" in top
            assert "payload" in top
            assert top["rank"] == 1
