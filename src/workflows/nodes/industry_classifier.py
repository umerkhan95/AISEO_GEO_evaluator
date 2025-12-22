"""
Industry Classifier Node: Detects website industry from content.

Uses LLM to analyze content and classify into predefined industries
for better guideline matching.
"""

from typing import Tuple, List
from pydantic import BaseModel, Field

# Use centralized clients
from src.clients import get_classifier_llm

# Predefined industries for classification
INDUSTRIES = [
    "B2B_SaaS",
    "Healthcare",
    "Ecommerce",
    "Finance",
    "Education",
    "Technology",
    "Marketing",
    "Legal",
    "Real_Estate",
    "Manufacturing",
    "Travel",
    "Food_Beverage",
    "Entertainment",
    "Nonprofit",
    "Government",
    "Other"
]


class IndustryClassification(BaseModel):
    """Structured output for industry classification."""
    primary_industry: str = Field(description="Primary industry classification")
    confidence: float = Field(description="Confidence score 0-1")
    secondary_industries: List[str] = Field(default_factory=list, description="Secondary industries if applicable")
    reasoning: str = Field(description="Brief explanation for classification")


def classify_industry(content: str, url: str = "") -> Tuple[str, float, List[str]]:
    """
    Classify content into an industry.

    Args:
        content: Page content (can be truncated for efficiency)
        url: Optional URL for additional context

    Returns:
        Tuple of (primary_industry, confidence, secondary_industries)
    """
    # Use centralized classifier LLM (gpt-4o-mini, temp=0)
    llm = get_classifier_llm()

    # Truncate content to first 2000 words for efficiency
    words = content.split()
    truncated = " ".join(words[:2000])

    prompt = f"""Analyze this website content and classify it into ONE primary industry.

AVAILABLE INDUSTRIES:
{', '.join(INDUSTRIES)}

URL: {url}

CONTENT (truncated):
{truncated}

Classify the primary industry, provide confidence (0-1), and list any secondary industries.
Consider:
- Products/services mentioned
- Target audience
- Terminology and jargon used
- Business model indicators
"""

    structured_llm = llm.with_structured_output(IndustryClassification)
    result = structured_llm.invoke(prompt)

    return result.primary_industry, result.confidence, result.secondary_industries


def get_industry_keywords(industry: str) -> List[str]:
    """Get relevant keywords for an industry to enhance guideline retrieval."""
    keywords = {
        "B2B_SaaS": ["enterprise", "API", "integration", "workflow", "automation", "SaaS", "subscription"],
        "Healthcare": ["patient", "medical", "HIPAA", "clinical", "health", "treatment", "diagnosis"],
        "Ecommerce": ["product", "cart", "checkout", "shipping", "catalog", "price", "buy"],
        "Finance": ["investment", "banking", "loan", "credit", "financial", "trading", "compliance"],
        "Education": ["course", "learning", "student", "curriculum", "certification", "training"],
        "Technology": ["software", "hardware", "tech", "digital", "innovation", "platform"],
        "Marketing": ["campaign", "brand", "advertising", "SEO", "content", "social media"],
        "Legal": ["attorney", "law", "legal", "compliance", "contract", "litigation"],
        "Real_Estate": ["property", "listing", "mortgage", "real estate", "home", "rental"],
        "Manufacturing": ["production", "supply chain", "manufacturing", "industrial", "factory"],
        "Travel": ["booking", "hotel", "flight", "destination", "travel", "vacation"],
        "Food_Beverage": ["restaurant", "menu", "food", "beverage", "dining", "recipe"],
        "Entertainment": ["media", "streaming", "content", "entertainment", "video", "music"],
        "Nonprofit": ["donate", "charity", "nonprofit", "mission", "volunteer", "cause"],
        "Government": ["government", "public", "policy", "citizen", "agency", "regulation"],
    }

    return keywords.get(industry, [])


if __name__ == "__main__":
    # Test classification
    test_content = """
    Our enterprise SaaS platform helps B2B companies automate their workflow.
    Integrate with your existing tools via our REST API.
    Start your 14-day free trial today.
    """

    industry, confidence, secondary = classify_industry(test_content, "https://example-saas.com")
    print(f"Industry: {industry} ({confidence:.0%} confidence)")
    print(f"Secondary: {secondary}")
