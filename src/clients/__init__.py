"""
Centralized client factories for AISEO.

This module provides singleton instances of external service clients
to avoid code duplication and ensure consistent configuration.
"""

from src.clients.qdrant_client import (
    get_qdrant_client,
    ensure_collections_exist,
    get_collection_name,
    COLLECTIONS,
    COLLECTION_NAMES,
)
from src.clients.embeddings import (
    get_embeddings,
    get_openai_client,
    generate_embedding,
)
from src.clients.llm import (
    get_llm,
    get_classifier_llm,
    get_optimizer_llm,
    get_humanizer_llm,
    get_extractor_llm,
    get_scorer_llm,
    LLMPreset,
)
from src.clients.utils import (
    calculate_priority,
    calculate_complexity,
    format_search_result,
)

__all__ = [
    # Qdrant
    "get_qdrant_client",
    "ensure_collections_exist",
    "get_collection_name",
    "COLLECTIONS",
    "COLLECTION_NAMES",
    # Embeddings
    "get_embeddings",
    "get_openai_client",
    "generate_embedding",
    # LLM
    "get_llm",
    "get_classifier_llm",
    "get_optimizer_llm",
    "get_humanizer_llm",
    "get_extractor_llm",
    "get_scorer_llm",
    "LLMPreset",
    # Utilities
    "calculate_priority",
    "calculate_complexity",
    "format_search_result",
]
