"""
Shared utility functions for AISEO.

Consolidates priority calculation, complexity analysis, and other common utilities.
"""

from config import PRIORITY_RULES, COMPLEXITY_INDICATORS


def calculate_priority(confidence: float, impact_score: float = 0.5) -> str:
    """
    Calculate priority level based on confidence and impact scores.

    Args:
        confidence: Confidence score (0.0 to 1.0).
        impact_score: Impact score (0.0 to 1.0).

    Returns:
        Priority level: 'critical', 'high', 'medium', or 'low'.
    """
    if confidence >= PRIORITY_RULES["critical"]["min_confidence"] and impact_score >= PRIORITY_RULES["critical"]["min_impact_score"]:
        return "critical"
    elif confidence >= PRIORITY_RULES["high"]["min_confidence"] and impact_score >= PRIORITY_RULES["high"]["min_impact_score"]:
        return "high"
    elif confidence >= PRIORITY_RULES["medium"]["min_confidence"] and impact_score >= PRIORITY_RULES["medium"]["min_impact_score"]:
        return "medium"
    else:
        return "low"


def calculate_complexity(content: str) -> str:
    """
    Calculate implementation complexity based on content keywords.

    Args:
        content: Guideline or content text.

    Returns:
        Complexity level: 'easy', 'moderate', or 'complex'.
    """
    content_lower = content.lower()

    # Count keyword matches for each level
    complex_count = sum(1 for word in COMPLEXITY_INDICATORS["complex"] if word in content_lower)
    moderate_count = sum(1 for word in COMPLEXITY_INDICATORS["moderate"] if word in content_lower)
    easy_count = sum(1 for word in COMPLEXITY_INDICATORS["easy"] if word in content_lower)

    # Determine complexity based on weighted counts
    if complex_count >= 2 or (complex_count >= 1 and moderate_count >= 2):
        return "complex"
    elif moderate_count >= 2 or (moderate_count >= 1 and complex_count >= 1):
        return "moderate"
    else:
        return "easy"


def format_search_result(
    title: str,
    url: str,
    snippet: str,
    score: float = 1.0,
    source: str = "web",
    max_snippet_length: int = 500,
) -> dict:
    """
    Format a search result consistently.

    Args:
        title: Result title.
        url: Result URL.
        snippet: Result snippet/description.
        score: Relevance score.
        source: Source identifier.
        max_snippet_length: Maximum snippet length.

    Returns:
        Formatted search result dictionary.
    """
    return {
        "title": title or "",
        "url": url or "",
        "snippet": (snippet or "")[:max_snippet_length],
        "score": score,
        "source": source,
    }
