"""
Configuration management for GEO/SEO Knowledge Base system.

Centralizes all API keys, model configs, and system parameters.
"""

import os
from dataclasses import dataclass
from typing import Literal


@dataclass
class ModelConfig:
    """Configuration for AI models."""

    # OpenAI GPT-4o for PDF processing (multimodal)
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_temperature: float = 0.1  # Low temp for extraction accuracy

    # Gemini as fallback (optional)
    gemini_model: str = "gemini-2.0-flash-exp"
    gemini_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    gemini_temperature: float = 0.1

    # OpenAI for embeddings
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536


@dataclass
class QdrantConfig:
    """Configuration for Qdrant vector database."""

    host: str = os.getenv("QDRANT_HOST", "localhost")
    port: int = int(os.getenv("QDRANT_PORT", "6333"))
    api_key: str = os.getenv("QDRANT_API_KEY", "")
    grpc_port: int = int(os.getenv("QDRANT_GRPC_PORT", "6334"))

    # Collection names for 5 categories
    collections: dict[str, str] = None

    def __post_init__(self):
        if self.collections is None:
            self.collections = {
                "universal_seo_geo": "geo_seo_universal",
                "industry_specific": "geo_seo_industry",
                "technical": "geo_seo_technical",
                "citation_optimization": "geo_seo_citation",
                "metrics": "geo_seo_metrics",
            }

    @property
    def collection_names(self) -> list[str]:
        """Get all collection names."""
        return list(self.collections.values())


@dataclass
class ProcessingConfig:
    """Configuration for document processing behavior."""

    # Deduplication threshold
    similarity_threshold: float = 0.85

    # Memory management
    max_concurrent_documents: int = 3
    max_memory_mb: int = 2048
    chunk_size: int = 10  # Process PDFs in chunks

    # Retry configuration
    max_retries: int = 3
    retry_delay_seconds: int = 5

    # PDF processing limits
    max_pdf_pages: int = 500
    max_pdf_size_mb: int = 50

    # Guideline extraction
    min_guideline_length: int = 20
    max_guidelines_per_document: int = 500


@dataclass
class SystemConfig:
    """Master configuration combining all sub-configs."""

    models: ModelConfig
    qdrant: QdrantConfig
    processing: ProcessingConfig

    @classmethod
    def from_env(cls) -> "SystemConfig":
        """Create configuration from environment variables."""
        return cls(
            models=ModelConfig(),
            qdrant=QdrantConfig(),
            processing=ProcessingConfig(),
        )


# Singleton instance
config = SystemConfig.from_env()


# Category mapping for routing
CATEGORY_DESCRIPTIONS = {
    "universal_seo_geo": "Guidelines applicable to all industries and contexts for both SEO and GEO",
    "industry_specific": "Guidelines specific to particular industries (e.g., healthcare, finance, e-commerce)",
    "technical": "Technical implementation guidelines (schema markup, site architecture, performance)",
    "citation_optimization": "Guidelines for citation formatting, source credibility, and reference optimization",
    "metrics": "Measurement, analytics, and performance tracking guidelines",
}


# Priority scoring rules
PRIORITY_RULES = {
    "critical": {"min_confidence": 0.9, "min_impact_score": 0.8},
    "high": {"min_confidence": 0.75, "min_impact_score": 0.6},
    "medium": {"min_confidence": 0.6, "min_impact_score": 0.4},
    "low": {"min_confidence": 0.0, "min_impact_score": 0.0},
}


# Complexity indicators (keywords that suggest implementation difficulty)
COMPLEXITY_INDICATORS = {
    "easy": ["add", "include", "use", "write", "simple", "basic"],
    "moderate": ["optimize", "implement", "configure", "structure", "organize"],
    "complex": [
        "develop",
        "architect",
        "integrate",
        "automate",
        "system",
        "infrastructure",
        "algorithm",
    ],
}
