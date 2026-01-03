#!/usr/bin/env python3
"""
Script to add GEO (Generative Engine Optimization) citations to Qdrant.
Based on research from Princeton, Georgia Tech, Allen AI, and IIT Delhi studies.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.qdrant_tools import store_guidelines

# Source papers metadata
SOURCES = {
    "princeton_geo": {
        "title": "GEO: Generative Engine Optimization",
        "authors": ["Pranjal Aggarwal", "Vishvak Murahari", "Tanmay Rajpurohit", "Ashwin Kalyan", "Karthik Narasimhan", "Ameet Deshpande"],
        "url": "https://arxiv.org/abs/2311.09735",
        "venue": "KDD 2024",
    },
    "ai_search_domination": {
        "title": "Generative Engine Optimization: How to Dominate AI Search",
        "authors": ["Chen et al."],
        "url": "https://arxiv.org/abs/2509.08919",
        "venue": "arXiv 2025",
    },
    "seo_ai_guide": {
        "title": "Generative Engine Optimization (GEO) Research Guide",
        "authors": ["SEO.ai Research Team"],
        "url": "https://seo.ai/blog/generative-engine-optimization-geo",
        "venue": "SEO.ai",
    },
    "allaboutai_geo": {
        "title": "Generative Engine Optimization: The Definitive Research-Backed Guide",
        "authors": ["AllAboutAI Research"],
        "url": "https://www.allaboutai.com/geo/what-is-generative-engine-optimization/",
        "venue": "AllAboutAI",
    },
}

# GEO Citations organized by category
GEO_CITATIONS = [
    # ===== CITATION OPTIMIZATION TACTICS =====
    {
        "content": "Statistics Addition improves AI search visibility by 41% compared to baseline. Modify content to include quantitative statistics instead of qualitative discussion wherever possible, adding data-driven evidence to support claims.",
        "category": "citation_optimization",
        "industries": [],
        "source_section": "Top Performing GEO Strategies",
        "confidence": 0.95,
        "has_quantitative_data": True,
        "has_case_study": False,
        "source": "princeton_geo",
    },
    {
        "content": "Quotation Addition boosts AI search visibility by up to 40%. Incorporate direct quotes from relevant, authoritative sources to enhance content authenticity and depth, making it more likely to be cited by generative engines.",
        "category": "citation_optimization",
        "industries": [],
        "source_section": "Top Performing GEO Strategies",
        "confidence": 0.95,
        "has_quantitative_data": True,
        "has_case_study": False,
        "source": "princeton_geo",
    },
    {
        "content": "Cite Sources strategy improves visibility by 30-40% with minimal content changes. Add relevant citations from credible sources throughout content to support claims and provide attribution, which generative engines strongly prefer.",
        "category": "citation_optimization",
        "industries": [],
        "source_section": "Top Performing GEO Strategies",
        "confidence": 0.95,
        "has_quantitative_data": True,
        "has_case_study": False,
        "source": "princeton_geo",
    },
    {
        "content": "Minimum 3 authoritative citations per 500 words is recommended for GEO-optimized content. Back all quantitative claims with sources and ensure working hyperlinks to increase AI citation likelihood.",
        "category": "citation_optimization",
        "industries": [],
        "source_section": "Tactical Implementation",
        "confidence": 0.85,
        "has_quantitative_data": True,
        "has_case_study": False,
        "source": "allaboutai_geo",
    },
    {
        "content": "The combination of Fluency Optimization and Statistics Addition outperforms any single GEO strategy by more than 5.5%. Combining multiple GEO methods yields better results than isolated tactics.",
        "category": "citation_optimization",
        "industries": [],
        "source_section": "Combination Strategies",
        "confidence": 0.90,
        "has_quantitative_data": True,
        "has_case_study": False,
        "source": "princeton_geo",
    },
    {
        "content": "Cite Sources significantly boosts performance when used in conjunction with other methods, with an average improvement of 31.4% when combined with other optimization strategies.",
        "category": "citation_optimization",
        "industries": [],
        "source_section": "Combination Strategies",
        "confidence": 0.90,
        "has_quantitative_data": True,
        "has_case_study": False,
        "source": "princeton_geo",
    },
    {
        "content": "Structured Query-Response Alignment achieves 28% higher inclusion rates in generative responses. Use clear headings that mirror common search queries, front-load definitive answers, include FAQ sections, and organize content hierarchically.",
        "category": "citation_optimization",
        "industries": [],
        "source_section": "Core Optimization Strategies",
        "confidence": 0.85,
        "has_quantitative_data": True,
        "has_case_study": False,
        "source": "allaboutai_geo",
    },
    {
        "content": "GEO-optimized content shows 4.4x higher conversions versus traditional SEO-only content. Implementing citation optimization and statistical evidence integration significantly improves both visibility and conversion rates.",
        "category": "citation_optimization",
        "industries": [],
        "source_section": "Key Statistics",
        "confidence": 0.85,
        "has_quantitative_data": True,
        "has_case_study": False,
        "source": "allaboutai_geo",
    },

    # ===== UNIVERSAL SEO/GEO STRATEGIES =====
    {
        "content": "Keyword stuffing performs 10% WORSE than baseline in generative engine optimization. Traditional SEO-based keyword strategies are not applicable to Generative Engine settings and can harm visibility.",
        "category": "universal_seo_geo",
        "industries": [],
        "source_section": "What Doesn't Work",
        "confidence": 0.95,
        "has_quantitative_data": True,
        "has_case_study": False,
        "source": "princeton_geo",
    },
    {
        "content": "AI search engines exhibit a systematic and overwhelming bias towards Earned media (third-party, authoritative sources) over Brand-owned and Social content. This contrasts sharply with Google's more balanced mix.",
        "category": "universal_seo_geo",
        "industries": [],
        "source_section": "AI Search Bias",
        "confidence": 0.90,
        "has_quantitative_data": False,
        "has_case_study": True,
        "source": "ai_search_domination",
    },
    {
        "content": "Domain-level authority matters more than specific page rankings for AI visibility. Most ChatGPT citations come from URLs ranking beyond position 21+ on traditional Google search.",
        "category": "universal_seo_geo",
        "industries": [],
        "source_section": "Key Optimization Strategies",
        "confidence": 0.85,
        "has_quantitative_data": True,
        "has_case_study": False,
        "source": "allaboutai_geo",
    },
    {
        "content": "Top content formats for LLM visibility include: Q&A style content, lists with bullet points or numbered steps, how-to guides with clear instructions, TL;DR summaries, and FAQs answering common questions.",
        "category": "universal_seo_geo",
        "industries": [],
        "source_section": "Content Formats",
        "confidence": 0.90,
        "has_quantitative_data": False,
        "has_case_study": False,
        "source": "allaboutai_geo",
    },
    {
        "content": "Platforms like Google AI Overviews and Perplexity favor sites that answer specific questions clearly, especially those starting with 'how,' 'what,' or 'best.' Structure content to directly address these query patterns.",
        "category": "universal_seo_geo",
        "industries": [],
        "source_section": "Content Optimization",
        "confidence": 0.85,
        "has_quantitative_data": False,
        "has_case_study": False,
        "source": "allaboutai_geo",
    },
    {
        "content": "79% of consumers are expected to use AI-enhanced search within one year. Gartner projects a 25% decline in traditional search volume by 2026, making GEO increasingly critical for digital visibility.",
        "category": "universal_seo_geo",
        "industries": [],
        "source_section": "Market Trends",
        "confidence": 0.85,
        "has_quantitative_data": True,
        "has_case_study": False,
        "source": "allaboutai_geo",
    },

    # ===== INDUSTRY-SPECIFIC STRATEGIES =====
    {
        "content": "Authoritative tone optimization works best for Historical domain content. Use persuasive, confident language with clear expertise signals when writing about historical topics for generative engine visibility.",
        "category": "industry_specific",
        "industries": ["Education", "Publishing", "Media"],
        "source_section": "Domain-Specific Effectiveness",
        "confidence": 0.85,
        "has_quantitative_data": False,
        "has_case_study": True,
        "source": "princeton_geo",
    },
    {
        "content": "Citation optimization is most effective for factual search queries. Content answering factual questions should prioritize adding relevant citations from authoritative sources to maximize AI citation probability.",
        "category": "industry_specific",
        "industries": ["Education", "B2B_SaaS", "Healthcare"],
        "source_section": "Domain-Specific Effectiveness",
        "confidence": 0.85,
        "has_quantitative_data": False,
        "has_case_study": True,
        "source": "princeton_geo",
    },
    {
        "content": "Statistics addition proves most beneficial for Law and Government-related questions. Legal and governmental content should include quantitative data, percentages, and specific metrics to improve generative engine visibility.",
        "category": "industry_specific",
        "industries": ["Legal", "Government", "Finance"],
        "source_section": "Domain-Specific Effectiveness",
        "confidence": 0.85,
        "has_quantitative_data": False,
        "has_case_study": True,
        "source": "princeton_geo",
    },
    {
        "content": "Technical domains require 47% more statistical backing than other content types. Health and science content demands peer-reviewed citations for optimal AI search visibility.",
        "category": "industry_specific",
        "industries": ["Healthcare", "Technology", "B2B_SaaS"],
        "source_section": "Multi-Domain Semantic Optimization",
        "confidence": 0.85,
        "has_quantitative_data": True,
        "has_case_study": False,
        "source": "allaboutai_geo",
    },

    # ===== TECHNICAL IMPLEMENTATION =====
    {
        "content": "ChatGPT relies heavily on Wikipedia (47.9% of top 10 cited sources) and Reddit (11.3%). Ensure your content is referenced or discussed on these platforms to increase AI citation likelihood.",
        "category": "technical",
        "industries": [],
        "source_section": "Platform-Specific Citation Patterns",
        "confidence": 0.90,
        "has_quantitative_data": True,
        "has_case_study": False,
        "source": "allaboutai_geo",
    },
    {
        "content": "Perplexity has a strong preference for community-generated content, with Reddit dominating at 46.7% of citations. Participating in relevant Reddit communities can significantly boost Perplexity visibility.",
        "category": "technical",
        "industries": [],
        "source_section": "Platform-Specific Citation Patterns",
        "confidence": 0.90,
        "has_quantitative_data": True,
        "has_case_study": False,
        "source": "allaboutai_geo",
    },
    {
        "content": "Google AI Overviews shows diverse citation distribution: Reddit (21%), YouTube (18.8%), Quora (14.3%), and LinkedIn (13%). YouTube citations have surged 310% since August 2024.",
        "category": "technical",
        "industries": [],
        "source_section": "Platform-Specific Citation Patterns",
        "confidence": 0.90,
        "has_quantitative_data": True,
        "has_case_study": False,
        "source": "allaboutai_geo",
    },
    {
        "content": "Implement schema markup, ensure AI crawler accessibility, optimize site speed and mobile responsiveness, and create AI-focused XML sitemaps for technical GEO optimization.",
        "category": "technical",
        "industries": [],
        "source_section": "Technical Requirements",
        "confidence": 0.80,
        "has_quantitative_data": False,
        "has_case_study": False,
        "source": "allaboutai_geo",
    },
    {
        "content": "Engineer content specifically for scannability and justification by AI systems. Structure content so AI can easily extract and cite specific facts, statistics, and claims with clear attribution.",
        "category": "technical",
        "industries": [],
        "source_section": "Machine Optimization",
        "confidence": 0.85,
        "has_quantitative_data": False,
        "has_case_study": False,
        "source": "ai_search_domination",
    },
    {
        "content": "Develop engine-specific strategies accounting for different AI search engine preferences and language variations. Significant variations exist among AI services regarding domain diversity, content freshness, and query sensitivity.",
        "category": "technical",
        "industries": [],
        "source_section": "Engine Differences",
        "confidence": 0.85,
        "has_quantitative_data": False,
        "has_case_study": True,
        "source": "ai_search_domination",
    },

    # ===== METRICS =====
    {
        "content": "Track citation frequency, position-adjusted visibility scores, cross-platform consistency, and multi-platform attribution rates across ChatGPT, Gemini, Perplexity, and Claude for GEO success measurement.",
        "category": "metrics",
        "industries": [],
        "source_section": "Success Metrics",
        "confidence": 0.80,
        "has_quantitative_data": False,
        "has_case_study": False,
        "source": "allaboutai_geo",
    },
    {
        "content": "Position-Adjusted Word Count and Subjective Impression are key metrics for measuring GEO effectiveness. The best methods improve these metrics by 41% and 28% respectively over baseline.",
        "category": "metrics",
        "industries": [],
        "source_section": "Performance Metrics",
        "confidence": 0.90,
        "has_quantitative_data": True,
        "has_case_study": False,
        "source": "princeton_geo",
    },
    {
        "content": "Perplexity delivers both the highest volume of AI referral traffic and the best conversion rates, with ChatGPT following as the second most impactful. For some sites, AI tools account for about 10% of total traffic.",
        "category": "metrics",
        "industries": [],
        "source_section": "Traffic Attribution",
        "confidence": 0.85,
        "has_quantitative_data": True,
        "has_case_study": True,
        "source": "allaboutai_geo",
    },
    {
        "content": "ChatGPT citations increased 556% throughout 2025 (from 0.9% to 5.9%), while Perplexity query processing tripled from 230M to 780M between mid-2024 and May 2025.",
        "category": "metrics",
        "industries": [],
        "source_section": "Growth Trends",
        "confidence": 0.85,
        "has_quantitative_data": True,
        "has_case_study": False,
        "source": "allaboutai_geo",
    },
    {
        "content": "GEO methods demonstrated efficacy on Perplexity.ai, a real-world generative engine, with visibility improvements up to 37% in controlled experiments.",
        "category": "metrics",
        "industries": [],
        "source_section": "Validation Results",
        "confidence": 0.90,
        "has_quantitative_data": True,
        "has_case_study": True,
        "source": "princeton_geo",
    },
]


def main():
    """Add all GEO citations to Qdrant."""
    print("=" * 60)
    print("Adding GEO Citations to Qdrant")
    print("=" * 60)

    # Group citations by source for storage
    for source_key, source_info in SOURCES.items():
        source_citations = [c for c in GEO_CITATIONS if c.get("source") == source_key]

        if not source_citations:
            continue

        print(f"\nProcessing source: {source_info['title']}")
        print(f"  Citations to add: {len(source_citations)}")

        # Remove the source key from citations before storing
        guidelines = []
        for citation in source_citations:
            guideline = {k: v for k, v in citation.items() if k != "source"}
            guidelines.append(guideline)

        # Store guidelines
        result = store_guidelines(guidelines, source_info)
        print(f"  Result: {result}")

    print("\n" + "=" * 60)
    print("GEO Citations Added Successfully!")
    print("=" * 60)

    # Print summary by category
    categories = {}
    for c in GEO_CITATIONS:
        cat = c["category"]
        categories[cat] = categories.get(cat, 0) + 1

    print("\nSummary by Category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count} citations")

    print(f"\nTotal Citations Added: {len(GEO_CITATIONS)}")


if __name__ == "__main__":
    main()
