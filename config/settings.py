"""Configuration settings for Week 5 Vector Search Internals."""

import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# Base Directories
ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "config"
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
METADATA_DIR = DATA_DIR / "metadata"
RESULTS_DIR = ROOT_DIR / "results"
CHARTS_DIR = RESULTS_DIR / "charts"
SCREENSHOTS_DIR = RESULTS_DIR / "screenshots"
DOCS_DIR = ROOT_DIR / "docs"

# Ensure runtime directories exist
for directory in [RAW_DATA_DIR, EMBEDDINGS_DIR, METADATA_DIR, CHARTS_DIR, SCREENSHOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Load environment variables
load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    """Application configuration parameters."""
    
    # EURI / Gemini Embedding Configuration
    euri_api_key: str = os.getenv("EURI_API_KEY", "")
    euri_base_url: str = os.getenv("EURI_BASE_URL", "https://api.euri.ai/v1")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "768"))
    
    # Qdrant Vector Database
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    
    # Search & Benchmark
    top_k: int = int(os.getenv("TOP_K", "5"))
    random_seed: int = int(os.getenv("RANDOM_SEED", "42"))
    benchmark_query_count: int = int(os.getenv("BENCHMARK_QUERY_COUNT", "100"))
    
    # Collection names in Qdrant
    collection_cosine: str = "week5_cosine"
    collection_euclidean: str = "week5_euclidean"
    collection_dot: str = "week5_dot"


settings = Settings()
