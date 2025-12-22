"""
Centralized LLM client factory.

Provides pre-configured LLM instances for different use cases.
"""

import logging
from enum import Enum
from typing import Optional

from langchain_openai import ChatOpenAI

from config import config

logger = logging.getLogger(__name__)


class LLMPreset(Enum):
    """Predefined LLM configurations for different use cases."""

    # Default - balanced for general use
    DEFAULT = "default"

    # For classification tasks (low temperature, fast model)
    CLASSIFIER = "classifier"

    # For content optimization (moderate temperature)
    OPTIMIZER = "optimizer"

    # For humanization (higher temperature for creativity)
    HUMANIZER = "humanizer"

    # For extraction tasks (very low temperature)
    EXTRACTOR = "extractor"

    # For scoring/evaluation (low temperature, precise)
    SCORER = "scorer"


# Preset configurations
_PRESET_CONFIGS = {
    LLMPreset.DEFAULT: {
        "model": "gpt-4o",
        "temperature": 0.3,
    },
    LLMPreset.CLASSIFIER: {
        "model": "gpt-4o-mini",
        "temperature": 0.0,
    },
    LLMPreset.OPTIMIZER: {
        "model": "gpt-4o",
        "temperature": 0.3,
    },
    LLMPreset.HUMANIZER: {
        "model": "gpt-4o",
        "temperature": 0.5,
    },
    LLMPreset.EXTRACTOR: {
        "model": "gpt-4o",
        "temperature": 0.1,
    },
    LLMPreset.SCORER: {
        "model": "gpt-4o-mini",
        "temperature": 0.0,
    },
}

# Cache for LLM instances
_llm_cache: dict[str, ChatOpenAI] = {}


def get_llm(
    preset: LLMPreset = LLMPreset.DEFAULT,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    use_cache: bool = True,
) -> ChatOpenAI:
    """
    Get a configured ChatOpenAI instance.

    Args:
        preset: Predefined configuration preset.
        model: Override model name (optional).
        temperature: Override temperature (optional).
        use_cache: Whether to cache and reuse instances.

    Returns:
        ChatOpenAI: Configured LLM instance.
    """
    # Get preset config
    preset_config = _PRESET_CONFIGS.get(preset, _PRESET_CONFIGS[LLMPreset.DEFAULT])

    # Apply overrides
    final_model = model or preset_config["model"]
    final_temperature = temperature if temperature is not None else preset_config["temperature"]

    # Create cache key
    cache_key = f"{final_model}:{final_temperature}"

    # Check cache
    if use_cache and cache_key in _llm_cache:
        return _llm_cache[cache_key]

    # Create new instance
    llm = ChatOpenAI(
        model=final_model,
        temperature=final_temperature,
        api_key=config.models.openai_api_key,
    )

    logger.debug(f"Created LLM: model={final_model}, temp={final_temperature}")

    # Cache if requested
    if use_cache:
        _llm_cache[cache_key] = llm

    return llm


def get_classifier_llm() -> ChatOpenAI:
    """Get LLM configured for classification tasks."""
    return get_llm(LLMPreset.CLASSIFIER)


def get_optimizer_llm() -> ChatOpenAI:
    """Get LLM configured for content optimization."""
    return get_llm(LLMPreset.OPTIMIZER)


def get_humanizer_llm() -> ChatOpenAI:
    """Get LLM configured for humanization tasks."""
    return get_llm(LLMPreset.HUMANIZER)


def get_extractor_llm() -> ChatOpenAI:
    """Get LLM configured for extraction tasks."""
    return get_llm(LLMPreset.EXTRACTOR)


def get_scorer_llm() -> ChatOpenAI:
    """Get LLM configured for scoring/evaluation."""
    return get_llm(LLMPreset.SCORER)


def clear_cache() -> None:
    """Clear the LLM cache (useful for testing)."""
    global _llm_cache
    _llm_cache = {}
