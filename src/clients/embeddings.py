"""
Centralized embeddings client factory.

Provides singleton instances for generating embeddings consistently.
"""

import logging
from typing import Optional

from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

from config import config

logger = logging.getLogger(__name__)

# Singleton instances
_embeddings: Optional[OpenAIEmbeddings] = None
_openai_client: Optional[OpenAI] = None


def get_embeddings() -> OpenAIEmbeddings:
    """
    Get or create a singleton OpenAIEmbeddings instance (LangChain).

    Returns:
        OpenAIEmbeddings: Configured embeddings instance.
    """
    global _embeddings

    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model=config.models.embedding_model,
            openai_api_key=config.models.openai_api_key,
        )
        logger.info(f"Initialized OpenAIEmbeddings: {config.models.embedding_model}")

    return _embeddings


def get_openai_client() -> OpenAI:
    """
    Get or create a singleton OpenAI client (for direct API calls).

    Returns:
        OpenAI: Configured OpenAI client instance.
    """
    global _openai_client

    if _openai_client is None:
        _openai_client = OpenAI(api_key=config.models.openai_api_key)
        logger.info("Initialized OpenAI client")

    return _openai_client


def generate_embedding(text: str, max_length: int = 8000) -> list[float]:
    """
    Generate an embedding vector for the given text.

    Args:
        text: Text to embed.
        max_length: Maximum text length to embed (truncates if longer).

    Returns:
        List of floats representing the embedding vector.
    """
    client = get_openai_client()
    truncated_text = text[:max_length] if len(text) > max_length else text

    response = client.embeddings.create(
        model=config.models.embedding_model,
        input=truncated_text,
    )

    return response.data[0].embedding


def reset_clients() -> None:
    """Reset singleton clients (useful for testing)."""
    global _embeddings, _openai_client
    _embeddings = None
    _openai_client = None
