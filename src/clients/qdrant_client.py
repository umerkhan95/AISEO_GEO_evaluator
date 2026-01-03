"""
Centralized Qdrant client factory.

Provides a singleton QdrantClient instance and collection management utilities.
"""

import logging
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from config import config

logger = logging.getLogger(__name__)

# Singleton instance
_qdrant_client: Optional[QdrantClient] = None

# Single source of truth for collection mappings
# Keys are category names, values are Qdrant collection names
COLLECTIONS = {
    "universal_seo_geo": "geo_seo_universal",
    "universal": "geo_seo_universal",  # Alias
    "industry_specific": "geo_seo_industry",
    "industry": "geo_seo_industry",  # Alias
    "technical": "geo_seo_technical",
    "citation_optimization": "geo_seo_citation",
    "citation": "geo_seo_citation",  # Alias
    "metrics": "geo_seo_metrics",
}

# Unique collection names (for iteration)
COLLECTION_NAMES = list(set(COLLECTIONS.values()))


def get_qdrant_client() -> QdrantClient:
    """
    Get or create a singleton QdrantClient instance.

    Returns:
        QdrantClient: Configured Qdrant client instance.
    """
    global _qdrant_client

    if _qdrant_client is None:
        host = config.qdrant.host
        api_key = config.qdrant.api_key if config.qdrant.api_key else None

        # Use URL for cloud instances (contain 'cloud.qdrant.io')
        if "cloud.qdrant.io" in host or api_key:
            url = f"https://{host}" if not host.startswith("http") else host
            _qdrant_client = QdrantClient(
                url=url,
                api_key=api_key,
            )
            logger.info(f"Initialized Qdrant Cloud client: {url}")
        else:
            # Local instance
            _qdrant_client = QdrantClient(
                host=host,
                port=config.qdrant.port,
            )
            logger.info(f"Initialized Qdrant client: {host}:{config.qdrant.port}")

    return _qdrant_client


def ensure_collections_exist(
    client: Optional[QdrantClient] = None,
    embedding_dim: int = None,
) -> None:
    """
    Ensure all required collections exist in Qdrant.

    Args:
        client: Optional QdrantClient. If None, uses singleton.
        embedding_dim: Embedding dimension. Defaults to config value.
    """
    if client is None:
        client = get_qdrant_client()

    if embedding_dim is None:
        embedding_dim = config.models.embedding_dimensions

    for collection_name in COLLECTION_NAMES:
        try:
            if not client.collection_exists(collection_name):
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=embedding_dim,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Created collection: {collection_name}")
            else:
                logger.debug(f"Collection exists: {collection_name}")
        except Exception as e:
            logger.warning(f"Error ensuring collection {collection_name}: {e}")


def get_collection_name(category: str) -> str:
    """
    Get the Qdrant collection name for a category.

    Args:
        category: Category key (e.g., 'universal_seo_geo', 'technical').

    Returns:
        Collection name string.

    Raises:
        KeyError: If category is not found.
    """
    if category not in COLLECTIONS:
        raise KeyError(f"Unknown category: {category}. Valid: {list(COLLECTIONS.keys())}")
    return COLLECTIONS[category]


def reset_client() -> None:
    """Reset the singleton client (useful for testing)."""
    global _qdrant_client
    _qdrant_client = None
