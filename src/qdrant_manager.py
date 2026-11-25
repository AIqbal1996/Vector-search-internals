"""Phase 4: Qdrant Vector Database Manager.

Manages collections, vectors, payloads, and search operations on Qdrant:
- Multi-metric collections:
  - week5_cosine (Distance.COSINE)
  - week5_euclidean (Distance.EUCLID)
  - week5_dot (Distance.DOT)
- Supports Qdrant Cloud and local Docker endpoints
- Batch ingestion with verification
- Configurable exact search (for correctness testing) vs approximate HNSW search (for benchmarking)
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models

from config.settings import settings

logger = logging.getLogger(__name__)


class QdrantManager:
    """Manages Qdrant collections, data ingestion, and search queries."""

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0
    ):
        self.url = (url or settings.qdrant_url).rstrip("/")
        self.api_key = api_key if api_key is not None else settings.qdrant_api_key
        
        # Initialize official QdrantClient
        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key if self.api_key else None,
            timeout=timeout,
        )
        self.collection_cosine = settings.collection_cosine
        self.collection_euclidean = settings.collection_euclidean
        self.collection_dot = settings.collection_dot

    def verify_connection(self) -> bool:
        """Check connection to Qdrant cluster."""
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant at {self.url}: {e}")
            raise

    def create_collection(
        self,
        collection_name: str,
        dimension: int,
        distance: models.Distance,
        recreate: bool = False
    ) -> None:
        """Create or recreate a single Qdrant collection with the specified metric."""
        existing_collections = [c.name for c in self.client.get_collections().collections]
        if collection_name in existing_collections:
            if recreate:
                print(f"[RECREATE] Deleting existing collection '{collection_name}'...")
                self.client.delete_collection(collection_name)
            else:
                info = self.client.get_collection(collection_name)
                print(f"[EXISTS] Collection '{collection_name}' already exists with {info.points_count} points.")
                return

        print(f"[CREATE] Creating collection '{collection_name}' (dim={dimension}, metric={distance.name})...")
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=dimension,
                distance=distance,
            ),
            hnsw_config=models.HnswConfigDiff(
                m=16,
                ef_construct=100,
                full_scan_threshold=1000,
            ),
        )
        print(f"[OK] Collection '{collection_name}' created successfully.")

    def init_all_collections(self, dimension: int, recreate: bool = False) -> None:
        """Initialize all three week5 collections: cosine, euclidean, dot."""
        self.create_collection(self.collection_cosine, dimension, models.Distance.COSINE, recreate)
        self.create_collection(self.collection_euclidean, dimension, models.Distance.EUCLID, recreate)
        self.create_collection(self.collection_dot, dimension, models.Distance.DOT, recreate)

    def upload_dataset(
        self,
        collection_name: str,
        vectors: np.ndarray,
        ids: List[int],
        payloads: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 100,
        wait: bool = True
    ) -> None:
        """Batch upload vectors and metadata payloads into a Qdrant collection."""
        num_points = len(vectors)
        if len(ids) != num_points:
            raise ValueError(f"IDs count ({len(ids)}) does not match vectors count ({num_points}).")

        print(f"[UPLOAD] Uploading {num_points} vectors to '{collection_name}' in batches of {batch_size}...")
        
        num_batches = (num_points + batch_size - 1) // batch_size
        for b in range(num_batches):
            start = b * batch_size
            end = min(start + batch_size, num_points)
            batch_vectors = vectors[start:end].tolist()
            batch_ids = ids[start:end]
            batch_payloads = payloads[start:end] if payloads else [{} for _ in range(start, end)]

            points = [
                models.PointStruct(
                    id=int(batch_ids[i]),
                    vector=batch_vectors[i],
                    payload=batch_payloads[i]
                )
                for i in range(len(batch_ids))
            ]

            self.client.upsert(
                collection_name=collection_name,
                points=points,
                wait=(wait if b == num_batches - 1 else False)
            )

        info = self.client.get_collection(collection_name)
        print(f"[OK] Upload complete for '{collection_name}'. Current points count: {info.points_count}")

    def search(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        k: int = 5,
        exact: bool = False,
        hnsw_ef: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Execute a vector search query against Qdrant.

        Parameters:
            collection_name: target collection
            query_vector: 1D query array
            k: top-k results
            exact: if True, disables HNSW and performs exact brute-force search
            hnsw_ef: search-time HNSW exploration factor
        """
        search_params = models.SearchParams(
            exact=exact,
            hnsw_ef=hnsw_ef if hnsw_ef is not None else None
        )

        query_list = query_vector.astype(float).tolist()

        # Handle qdrant-client search / query API
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=collection_name,
                query=query_list,
                limit=k,
                search_params=search_params,
                with_payload=True
            )
            points = response.points
        else:
            points = self.client.search(
                collection_name=collection_name,
                query_vector=query_list,
                limit=k,
                search_params=search_params,
                with_payload=True
            )

        results = []
        for rank, pt in enumerate(points, start=1):
            results.append({
                "rank": rank,
                "id": int(pt.id),
                "score": float(pt.score),
                "payload": pt.payload or {}
            })
        return results
