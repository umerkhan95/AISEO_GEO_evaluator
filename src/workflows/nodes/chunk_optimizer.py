"""
Chunk Optimizer Node: Optimizes content chunks for GEO.

Takes a content chunk and applies retrieved guidelines to create
GEO-optimized content with citations, structure, and authority signals.
"""

import os
import time
from typing import Dict, List, Tuple
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


class OptimizedChunk(BaseModel):
    """Structured output for optimized content."""
    optimized_content: str = Field(description="The GEO-optimized content in markdown")
    geo_score: float = Field(description="Estimated GEO score 1-10")
    changes_made: List[str] = Field(default_factory=list, description="List of changes made")
    guidelines_used: List[str] = Field(default_factory=list, description="IDs of guidelines applied")


class GuidelineApplication(BaseModel):
    """How a guideline was applied."""
    guideline_id: str
    how_applied: str


def optimize_chunk(
    original_content: str,
    section_title: str,
    guidelines: List[Dict],
    industry: str = None,
    context: str = ""
) -> Tuple[str, float, List[Dict], int]:
    """
    Optimize a content chunk using retrieved guidelines.

    Args:
        original_content: The original chunk content
        section_title: Title of the section
        guidelines: Retrieved guidelines to apply
        industry: Detected industry for context
        context: Additional context (e.g., page title, URL)

    Returns:
        Tuple of (optimized_content, geo_score, guidelines_applied, processing_time_ms)
    """
    start_time = time.time()

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.3,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Format guidelines for prompt
    guidelines_text = format_guidelines_for_prompt(guidelines)

    prompt = f"""You are a GEO (Generative Engine Optimization) expert. Your task is to restructure content so AI systems like ChatGPT, Perplexity, and Google Gemini will cite it.

## CONTEXT
- Section: {section_title}
- Industry: {industry or "General"}
- Additional context: {context}

## ORIGINAL CONTENT
{original_content}

## GUIDELINES TO APPLY
{guidelines_text}

## YOUR TASK
Restructure the content to be GEO-optimized while STRICTLY preserving all original facts.

### 1. **Structure for AI Citability**
   - Add a TL;DR summary at the top (using ONLY facts from the original)
   - Use clear H2/H3 headings and bullet points
   - Make sections standalone-quotable
   - Add FAQ section at the end (answer questions using ONLY original content)

### 2. **Enhance Readability**
   - Use clear, declarative statements
   - Break long paragraphs into structured lists
   - Add descriptive subheadings

### 3. **Preserve Contact Info & Links**
   - Keep ALL original email addresses exactly as written
   - Preserve ALL URLs and links
   - Keep ALL company names exactly as written

## ABSOLUTE RULES - VIOLATIONS ARE UNACCEPTABLE

❌ **NEVER INVENT STATISTICS OR PERCENTAGES**
   - Do NOT add percentages like "increased by 40%" unless it's in the original
   - Do NOT add numbers like "saved $X" or "reduced by Y%" unless it's in the original

❌ **NEVER FABRICATE QUOTES**
   - Do NOT add expert quotes that don't exist in the original
   - Do NOT create fake attributions like "[Expert Name]"

❌ **NEVER MISDEFINE TERMS**
   - RAG = Retrieval-Augmented Generation (NOT Red-Amber-Green)
   - LLM = Large Language Model
   - If unsure about a definition, don't include it

❌ **NEVER CHANGE CONTACT INFO**
   - Keep original email addresses exactly (e.g., hello@example.eu stays as-is)
   - Don't invent or modify URLs

✅ **YOU CAN ONLY:**
   - Restructure existing content into better format
   - Add headings and bullet points
   - Create TL;DR summaries from existing facts
   - Create FAQ Q&A from existing information
   - Preserve statistics that ARE in the original content
   - Add markdown formatting

## OUTPUT
- Clean markdown format
- Natural, human-readable tone
- ALL facts must come from the original content

Return the optimized content, estimate its GEO score (1-10), and list which guidelines you applied."""

    structured_llm = llm.with_structured_output(OptimizedChunk)
    result = structured_llm.invoke(prompt)

    # Map applied guidelines
    applied = []
    for g in guidelines:
        if g["guideline_id"] in result.guidelines_used:
            applied.append({
                "guideline_id": g["guideline_id"],
                "guideline_content": g["content"],
                "guideline_category": g["category"],
                "how_applied": "Applied during optimization"
            })

    # If no specific guidelines tracked, try to infer from changes
    if not applied and guidelines:
        for change in result.changes_made[:3]:
            applied.append({
                "guideline_id": guidelines[0]["guideline_id"],
                "guideline_content": guidelines[0]["content"],
                "guideline_category": guidelines[0]["category"],
                "how_applied": change
            })

    processing_time = int((time.time() - start_time) * 1000)

    return result.optimized_content, result.geo_score, applied, processing_time


def format_guidelines_for_prompt(guidelines: List[Dict]) -> str:
    """Format guidelines into a structured prompt section."""
    if not guidelines:
        return "No specific guidelines - use general GEO best practices."

    formatted = []
    for i, g in enumerate(guidelines, 1):
        priority_emoji = "🔴" if g["priority"] == "critical" else "🟡" if g["priority"] == "high" else "⚪"
        formatted.append(
            f"{i}. [{priority_emoji} {g['priority'].upper()}] {g['content']}\n"
            f"   - Category: {g['category']}\n"
            f"   - Complexity: {g['implementation_complexity']}\n"
            f"   - ID: {g['guideline_id']}"
        )

    return "\n\n".join(formatted)


def score_original_content(content: str, section_title: str) -> float:
    """
    Score original content for GEO (before optimization).

    Quick heuristic-based scoring for speed.
    """
    score = 3.0  # Base score

    # Check for structure indicators
    if "##" in content or "**" in content:
        score += 0.5
    if "-" in content or "•" in content or "1." in content:
        score += 0.5

    # Check for authority indicators
    if "%" in content or any(c.isdigit() for c in content):
        score += 0.5
    if "according to" in content.lower() or "research" in content.lower():
        score += 0.5
    if '"' in content:  # Has quotes
        score += 0.3

    # Check for clarity indicators
    word_count = len(content.split())
    if 100 <= word_count <= 500:
        score += 0.5
    if "?" in content:  # Has questions (good for FAQ)
        score += 0.3

    # Check for GEO-specific patterns
    if any(phrase in content.lower() for phrase in ["what is", "how to", "why", "benefits"]):
        score += 0.4

    return min(score, 10.0)


if __name__ == "__main__":
    # Test optimization
    test_content = """
    Our platform helps businesses automate their workflows.
    You can integrate with popular tools.
    Contact us for more information.
    """

    test_guidelines = [
        {
            "guideline_id": "test123",
            "content": "Lead with direct answers using 'What is' frameworks",
            "category": "universal_seo_geo",
            "priority": "high",
            "implementation_complexity": "easy"
        }
    ]

    optimized, score, applied, time_ms = optimize_chunk(
        test_content,
        "Features",
        test_guidelines,
        "B2B_SaaS"
    )

    print(f"GEO Score: {score}/10")
    print(f"Processing time: {time_ms}ms")
    print(f"Guidelines applied: {len(applied)}")
    print(f"\nOptimized content:\n{optimized[:500]}...")
