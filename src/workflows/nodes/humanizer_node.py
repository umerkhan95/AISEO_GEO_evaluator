"""
Humanizer Node: Makes optimized content more natural and human-like.

Adds conversational elements, varies sentence structure, and reduces
AI detection signals while maintaining GEO optimization.
"""

from typing import Tuple
from pydantic import BaseModel, Field

# Use centralized clients
from src.clients import get_humanizer_llm


class HumanizedContent(BaseModel):
    """Structured output for humanized content."""
    content: str = Field(description="The humanized content")
    naturalness_score: float = Field(description="How natural the content sounds 1-10")
    changes_summary: str = Field(description="Summary of humanization changes")


def humanize_content(
    optimized_content: str,
    industry: str = None,
    tone: str = "professional"
) -> Tuple[str, float]:
    """
    Humanize optimized content to reduce AI detection.

    Args:
        optimized_content: The GEO-optimized content
        industry: Industry context for appropriate tone
        tone: Desired tone (professional, conversational, technical)

    Returns:
        Tuple of (humanized_content, naturalness_score)
    """
    # Use centralized humanizer LLM (higher temperature for variation)
    llm = get_humanizer_llm()

    prompt = f"""You are a content editor making AI-optimized text sound more human and natural.

## INPUT CONTENT
{optimized_content}

## INDUSTRY
{industry or "General"}

## DESIRED TONE
{tone}

## HUMANIZATION TECHNIQUES TO APPLY

1. **Vary Sentence Structure**
   - Mix short and long sentences
   - Start sentences differently (not always "The", "This", "It")
   - Use occasional sentence fragments for emphasis

2. **Add Conversational Elements**
   - Include transitional phrases ("Here's the thing...", "What this means is...")
   - Add rhetorical questions sparingly
   - Use contractions naturally (it's, you'll, we're)

3. **Inject Personality**
   - Add mild opinions where appropriate ("Frankly, this is essential...")
   - Include subtle humor if fits the brand
   - Use analogies or metaphors

4. **Vary Word Choice**
   - Avoid repetitive words/phrases
   - Use synonyms and variations
   - Include industry-specific casual terms

5. **Maintain Structure**
   - Keep bullet points and headers
   - Preserve statistics and citations
   - Don't remove GEO optimization elements

## CRITICAL RULES
- Keep all factual information accurate
- Preserve citations and statistics
- Maintain clear, scannable structure
- Don't over-humanize (keep it professional)
- The goal is 80% professional, 20% conversational

Output the humanized content that maintains GEO optimization while sounding natural."""

    structured_llm = llm.with_structured_output(HumanizedContent)
    result = structured_llm.invoke(prompt)

    return result.content, result.naturalness_score


def quick_humanize(content: str) -> str:
    """
    Quick humanization pass without LLM call.

    Good for light touch-ups or when speed is priority.
    """
    import re

    # Replace common AI patterns
    replacements = [
        (r'\bIn conclusion,\b', 'So,'),
        (r'\bFurthermore,\b', 'Plus,'),
        (r'\bMoreover,\b', 'And'),
        (r'\bIt is important to note that\b', 'Worth noting:'),
        (r'\bIt should be noted that\b', 'Note that'),
        (r'\bIn order to\b', 'To'),
        (r'\bDue to the fact that\b', 'Because'),
        (r'\bAt this point in time\b', 'Now'),
        (r'\bIn the event that\b', 'If'),
        (r'\bWith regard to\b', 'About'),
        (r'\bIn terms of\b', 'For'),
        (r'\bAs a matter of fact,\b', 'Actually,'),
        (r'\bIt goes without saying that\b', 'Obviously,'),
    ]

    result = content
    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    return result


def estimate_ai_detection_risk(content: str) -> float:
    """
    Estimate AI detection risk (0-1).

    Higher = more likely to be flagged as AI.
    This is a heuristic, not actual detection.
    """
    risk = 0.0

    # Check for AI-typical patterns
    ai_patterns = [
        "it is important to",
        "it should be noted",
        "in conclusion",
        "furthermore",
        "moreover",
        "in order to",
        "due to the fact",
        "delve into",
        "leverage",
        "utilize",
        "facilitate",
        "comprehensive",
        "robust",
        "seamless",
        "cutting-edge",
    ]

    content_lower = content.lower()
    pattern_count = sum(1 for p in ai_patterns if p in content_lower)
    risk += min(pattern_count * 0.05, 0.3)

    # Check sentence uniformity
    sentences = content.split('.')
    if sentences:
        lengths = [len(s.split()) for s in sentences if s.strip()]
        if lengths:
            avg_len = sum(lengths) / len(lengths)
            variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
            # Low variance = more uniform = more AI-like
            if variance < 20:
                risk += 0.2

    # Check for excessive bullet points
    bullet_count = content.count('- ') + content.count('• ')
    word_count = len(content.split())
    if word_count > 0 and bullet_count / word_count > 0.1:
        risk += 0.1

    return min(risk, 1.0)


if __name__ == "__main__":
    test_content = """
## What is Workflow Automation?

Workflow automation is the process of using technology to perform repetitive tasks automatically. It is important to note that this can significantly improve efficiency.

### Key Benefits

- **Increased Productivity**: Teams can focus on high-value work
- **Reduced Errors**: Automated processes are more consistent
- **Cost Savings**: Less manual labor required

Furthermore, workflow automation enables better scalability for growing businesses.
"""

    humanized, score = humanize_content(test_content, "B2B_SaaS")
    print(f"Naturalness Score: {score}/10")
    print(f"\nHumanized content:\n{humanized}")

    # Test quick humanize
    quick = quick_humanize(test_content)
    print(f"\n\nQuick humanized:\n{quick}")
