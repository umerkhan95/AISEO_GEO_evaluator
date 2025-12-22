"""
Configuration settings using Pydantic Settings.
Loads from environment variables and .env file.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys
    openai_api_key: str = Field(default="", description="OpenAI API key for embeddings")
    google_api_key: str = Field(default="", description="Google/Gemini API key")
    tavily_api_key: str = Field(default="", description="Tavily API key for web search")

    # Qdrant Configuration
    qdrant_url: str = Field(default="http://localhost:6333", description="Qdrant server URL")
    qdrant_api_key: str = Field(default="", description="Qdrant API key (optional)")

    # LangSmith Tracing
    langsmith_api_key: str = Field(default="", description="LangSmith API key")
    langsmith_project: str = Field(default="geo-seo-knowledge-base", description="LangSmith project name")
    langsmith_tracing: bool = Field(default=False, description="Enable LangSmith tracing")

    # Processing Configuration
    max_pdfs_per_batch: int = Field(default=10, description="Maximum PDFs per batch")
    similarity_threshold: float = Field(default=0.85, description="Similarity threshold for deduplication")
    embedding_model: str = Field(default="text-embedding-3-small", description="OpenAI embedding model")
    gemini_model: str = Field(default="gemini-2.0-flash", description="Gemini model for PDF analysis")

    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")

    # Paths
    pdfs_dir: str = Field(default="./pdfs", description="Directory for PDF storage")
    data_dir: str = Field(default="./data", description="Directory for data storage")

    @property
    def qdrant_collections(self) -> dict[str, str]:
        """Qdrant collection names and descriptions."""
        return {
            "universal_seo_geo_framework": "General SEO+GEO strategies applicable to all industries",
            "industry_specific_strategies": "Industry-tailored tactics (B2B SaaS, Healthcare, E-commerce, etc.)",
            "technical_implementation": "Schema markup, structured data, entity linking procedures",
            "citation_optimization_tactics": "Content structuring for AI citations, statistical integration",
            "measurement_metrics": "KPIs, ROI tracking, benchmark data",
        }

    @property
    def industry_tags(self) -> list[str]:
        """Supported industry tags."""
        return [
            "Universal",
            "B2B_SaaS",
            "Healthcare",
            "Ecommerce",
            "Finance",
            "Professional_Services",
            "Education",
            "Technology",
            "Media",
            "Travel",
        ]

    @property
    def guideline_categories(self) -> list[str]:
        """Guideline category types."""
        return [
            "Universal_SEO_GEO",
            "Industry_Specific",
            "Technical_Implementation",
            "Citation_Optimization",
            "Measurement_Metrics",
        ]


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
