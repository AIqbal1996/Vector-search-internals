"""Unit tests for Phase 2 Embedding Client & Caching."""

import pytest
import numpy as np

from src.embedding_client import EmbeddingClient


class TestEmbeddingClient:
    """Test embedding client validation, caching, and synthetic data generation."""

    def test_client_initialization(self):
        client = EmbeddingClient(
            api_key="test-key",
            base_url="https://api.euron.one/api/v1/euri",
            model="gemini-embedding-2",
            dimension=3072
        )
        assert client.model == "gemini-embedding-2"
        assert client.dimension == 3072
        assert client.is_api_configured() is True

    def test_synthetic_benchmark_dataset_generation(self, tmp_path):
        num_vecs = 50
        dim = 128
        vectors, meta = EmbeddingClient.generate_synthetic_benchmark_dataset(
            num_vectors=num_vecs,
            dimension=dim,
            seed=42,
            cache_name="test_synth"
        )
        assert vectors.shape == (num_vecs, dim)
        assert vectors.dtype == np.float32
        # Check normalization (unit magnitude)
        mags = np.linalg.norm(vectors, axis=1)
        assert np.allclose(mags, 1.0, atol=1e-5)
        assert meta["count"] == num_vecs
        assert meta["is_real_gemini"] is False

    def test_missing_api_key_raises_error(self):
        client = EmbeddingClient(api_key="", base_url="https://example.com")
        with pytest.raises(ValueError, match="EURI API Key is not configured"):
            client.get_embedding_single("test")
