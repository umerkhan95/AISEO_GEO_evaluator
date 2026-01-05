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
    "ai_visibility_report_2025": {
        "title": "2025 AI Visibility Report: How LLMs Choose What Sources to Mention",
        "authors": ["The Digital Bloom Research Team"],
        "url": "https://thedigitalbloom.com/learn/2025-ai-citation-llm-visibility-report/",
        "venue": "The Digital Bloom",
    },
    "profound_citation_patterns": {
        "title": "AI Platform Citation Patterns: How ChatGPT, Google AI Overviews, and Perplexity Source Information",
        "authors": ["Profound Analytics"],
        "url": "https://www.tryprofound.com/blog/ai-platform-citation-patterns",
        "venue": "Profound",
    },
    "superprompt_ai_traffic": {
        "title": "AI Traffic Surges 527% in 2025: How to Get Your Site Cited",
        "authors": ["Superprompt Research"],
        "url": "https://superprompt.com/blog/ai-traffic-up-527-percent-how-to-get-cited-by-chatgpt-claude-perplexity-2025",
        "venue": "Superprompt",
    },
    "perplexity_ranking_factors": {
        "title": "Perplexity AI Ranking 59+ Factors Revealed (2025)",
        "authors": ["Metehan AI Research"],
        "url": "https://metehan.ai/blog/perplexity-ai-seo-59-ranking-patterns/",
        "venue": "Metehan.ai",
    },
    "storychief_llm_content": {
        "title": "How to Structure Your Content So LLMs Are More Likely to Cite You",
        "authors": ["StoryChief Research"],
        "url": "https://storychief.io/blog/how-to-structure-your-content-so-llms-are-more-likely-to-cite-you",
        "venue": "StoryChief",
    },
    "omnius_geo_report": {
        "title": "GEO Industry Report 2025: Trends in AI & LLM Optimization",
        "authors": ["Omnius Research Team"],
        "url": "https://www.omnius.so/blog/geo-industry-report",
        "venue": "Omnius",
    },
    "averi_llm_guide": {
        "title": "The Definitive Guide to LLM-Optimized Content: How to Win in the AI Search Era",
        "authors": ["Averi AI Research"],
        "url": "https://www.averi.ai/breakdowns/the-definitive-guide-to-llm-optimized-content",
        "venue": "Averi AI",
    },
    "passionfruit_serp": {
        "title": "Why AI Citations Come from Top 10 Rankings | SERP Data Analysis",
        "authors": ["Passionfruit Analytics"],
        "url": "https://www.getpassionfruit.com/blog/why-ai-citations-lean-on-the-top-10",
        "venue": "Passionfruit",
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

    # ===== NEW 2025 RESEARCH: CONTENT STRUCTURE =====
    {
        "content": "Sites with H2→H3→bullet point structures are 40% more likely to be cited by AI platforms. Use clear heading hierarchy with nested bullet points for maximum AI scannability.",
        "category": "citation_optimization",
        "industries": [],
        "source_section": "Content Structure Analysis",
        "confidence": 0.92,
        "has_quantitative_data": True,
        "has_case_study": True,
        "source": "superprompt_ai_traffic",
    },
    {
        "content": "Answer each FAQ in 2-4 crisp sentences that can stand alone as a citation. Use 'Question → Short Answer → Deeper Explanation' structure so LLMs can easily extract a compact answer and additional context.",
        "category": "citation_optimization",
        "industries": [],
        "source_section": "FAQ Optimization",
        "confidence": 0.90,
        "has_quantitative_data": False,
        "has_case_study": True,
        "source": "storychief_llm_content",
    },
    {
        "content": "Content with clear questions and direct answers is 40% more likely to be rephrased and cited by AI tools like ChatGPT. Structure content to anticipate user queries.",
        "category": "citation_optimization",
        "industries": [],
        "source_section": "Query-Response Patterns",
        "confidence": 0.92,
        "has_quantitative_data": True,
        "has_case_study": False,
        "source": "storychief_llm_content",
    },
    {
        "content": "Listicles with embedded data tables earn 2.3x more AI citations than narrative articles. Use comparative list formats with structured data presentation.",
        "category": "citation_optimization",
        "industries": [],
        "source_section": "Content Format Analysis",
        "confidence": 0.90,
        "has_quantitative_data": True,
        "has_case_study": True,
        "source": "averi_llm_guide",
    },
    {
        "content": "Listicle and comparative content dominates AI citations, representing 25.37% of all AI citations across 2.6 billion analyzed citations. Prioritize comparison and list-based formats.",
        "category": "citation_optimization",
        "industries": [],
        "source_section": "Citation Format Analysis",
        "confidence": 0.95,
        "has_quantitative_data": True,
        "has_case_study": True,
        "source": "profound_citation_patterns",
    },

    # ===== NEW 2025 RESEARCH: PLATFORM-SPECIFIC OPTIMIZATION =====
    {
        "content": "Only 12% of sources cited across ChatGPT, Perplexity, and Google AI features match each other. 86% of top-mentioned sources aren't shared across platforms - platform-specific strategies are essential.",
        "category": "technical",
        "industries": [],
        "source_section": "Cross-Platform Analysis",
        "confidence": 0.95,
        "has_quantitative_data": True,
        "has_case_study": True,
        "source": "profound_citation_patterns",
    },
    {
        "content": "ChatGPT favors content depth and domain rating (0.161 correlation), Perplexity prioritizes content freshness and length, while Claude emphasizes technical accuracy. Optimize differently for each platform.",
        "category": "technical",
        "industries": [],
        "source_section": "Platform Preferences",
        "confidence": 0.90,
        "has_quantitative_data": True,
        "has_case_study": True,
        "source": "superprompt_ai_traffic",
    },
    {
        "content": "Wikipedia serves as ChatGPT's most cited source at 7.8% of total citations, followed by G2 (196K mentions), Forbes (181K), and Amazon (133K). Demonstrate expertise similar to these authoritative sources.",
        "category": "technical",
        "industries": [],
        "source_section": "ChatGPT Citation Sources",
        "confidence": 0.95,
        "has_quantitative_data": True,
        "has_case_study": True,
        "source": "profound_citation_patterns",
    },
    {
        "content": "Reddit dominates Perplexity citations at 6.6% (3.2 million mentions), followed by YouTube (906K) and LinkedIn (553K). Community engagement and user-generated content significantly boost Perplexity visibility.",
        "category": "technical",
        "industries": [],
        "source_section": "Perplexity Citation Sources",
        "confidence": 0.95,
        "has_quantitative_data": True,
        "has_case_study": True,
        "source": "profound_citation_patterns",
    },
    {
        "content": "Only 11% of domains appear across both ChatGPT and Perplexity results. Build platform-specific content strategies rather than one-size-fits-all approaches.",
        "category": "technical",
        "industries": [],
        "source_section": "Platform Overlap Analysis",
        "confidence": 0.92,
        "has_quantitative_data": True,
        "has_case_study": True,
        "source": "perplexity_ranking_factors",
    },

    # ===== NEW 2025 RESEARCH: CONTENT FRESHNESS =====
    {
        "content": "AI platforms cite content that's 25.7% fresher than what appears in organic results. Content freshness plays a bigger role in AI search than traditional SEO.",
        "category": "universal_seo_geo",
        "industries": [],
        "source_section": "Freshness Analysis",
        "confidence": 0.92,
        "has_quantitative_data": True,
        "has_case_study": True,
        "source": "superprompt_ai_traffic",
    },
    {
        "content": "ChatGPT shows the strongest recency bias with 76.4% of its most-cited pages updated in the last 30 days. Regularly update content to maintain AI visibility.",
        "category": "universal_seo_geo",
        "industries": [],
        "source_section": "Recency Analysis",
        "confidence": 0.95,
        "has_quantitative_data": True,
        "has_case_study": True,
        "source": "superprompt_ai_traffic",
    },
    {
        "content": "Pages with original data get 4.1x more AI citations than pages without original research. Prioritize creating original studies, surveys, and data-driven content.",
        "category": "citation_optimization",
        "industries": [],
        "source_section": "Original Research Impact",
        "confidence": 0.95,
        "has_quantitative_data": True,
        "has_case_study": True,
        "source": "superprompt_ai_traffic",
    },

    # ===== NEW 2025 RESEARCH: BRAND AUTHORITY =====
    {
        "content": "Brand search volume—not backlinks—is the strongest predictor of AI citations with 0.334 correlation. Traditional SEO metrics like backlinks show weak or negative correlations with AI visibility.",
        "category": "universal_seo_geo",
        "industries": [],
        "source_section": "Ranking Factor Analysis",
        "confidence": 0.95,
        "has_quantitative_data": True,
        "has_case_study": True,
        "source": "ai_visibility_report_2025",
    },
    {
        "content": "Brands in the top 25% for web mentions earn over 10x more AI Overview citations than the next quartile. Brand authority signals (mentions, branded anchors, search volume) correlate more strongly with AI visibility than backlinks.",
        "category": "universal_seo_geo",
        "industries": [],
        "source_section": "Brand Authority Impact",
        "confidence": 0.95,
        "has_quantitative_data": True,
        "has_case_study": True,
        "source": "passionfruit_serp",
    },
    {
        "content": "Content depth matters most for AI citations—word count and sentence count show the strongest positive correlations across all AI platforms. Prioritize comprehensive, in-depth content over thin pages.",
        "category": "citation_optimization",
        "industries": [],
        "source_section": "Content Depth Analysis",
        "confidence": 0.92,
        "has_quantitative_data": True,
        "has_case_study": True,
        "source": "superprompt_ai_traffic",
    },

    # ===== NEW 2025 RESEARCH: TECHNICAL IMPLEMENTATION =====
    {
        "content": "Sites loading under 2 seconds get preferential treatment from Perplexity. Page speed is a significant ranking factor for AI search visibility.",
        "category": "technical",
        "industries": [],
        "source_section": "Technical Performance",
        "confidence": 0.88,
        "has_quantitative_data": True,
        "has_case_study": False,
        "source": "perplexity_ranking_factors",
    },
    {
        "content": "Use schema markup (HowTo, FAQ, QAPage), bulleted lists, and clear H2s. Implement JSON-LD structured data and Semantic HTML5 markup to make content easier for LLMs to parse and cite.",
        "category": "technical",
        "industries": [],
        "source_section": "Schema Implementation",
        "confidence": 0.88,
        "has_quantitative_data": False,
        "has_case_study": True,
        "source": "averi_llm_guide",
    },
    {
        "content": "Implementing LLMs.txt (a proposed protocol similar to robots.txt for LLM crawlers) improves factual accuracy, relevance, and completeness of AI responses about your content.",
        "category": "technical",
        "industries": [],
        "source_section": "LLM Crawler Protocol",
        "confidence": 0.82,
        "has_quantitative_data": False,
        "has_case_study": True,
        "source": "averi_llm_guide",
    },
    {
        "content": "Mobile-first optimization is critical—majority of AI Overview citations come from mobile-indexed pages. Ensure responsive design and fast mobile load times.",
        "category": "technical",
        "industries": [],
        "source_section": "Mobile Optimization",
        "confidence": 0.85,
        "has_quantitative_data": False,
        "has_case_study": True,
        "source": "averi_llm_guide",
    },
    {
        "content": "Perplexity uses an L3 XGBoost reranker with BERT-based entity linking and disambiguation, promoting answers from authoritative domains. Build topical authority through comprehensive entity coverage.",
        "category": "technical",
        "industries": [],
        "source_section": "Perplexity Algorithm",
        "confidence": 0.88,
        "has_quantitative_data": False,
        "has_case_study": True,
        "source": "perplexity_ranking_factors",
    },

    # ===== NEW 2025 RESEARCH: MARKET TRENDS =====
    {
        "content": "Generative AI traffic has grown by 1,200% between July 2024 and February 2025 according to Adobe Analytics. AI-referred traffic to retail sites jumped 1,300% during 2024.",
        "category": "metrics",
        "industries": ["E-commerce", "Retail"],
        "source_section": "Traffic Growth Analysis",
        "confidence": 0.95,
        "has_quantitative_data": True,
        "has_case_study": True,
        "source": "omnius_geo_report",
    },
    {
        "content": "In 2025, 58% of consumers now rely on AI for product recommendations, more than double the 25% from just two years ago. AI search is becoming the primary discovery channel.",
        "category": "metrics",
        "industries": ["E-commerce", "Retail", "B2B_SaaS"],
        "source_section": "Consumer Behavior",
        "confidence": 0.92,
        "has_quantitative_data": True,
        "has_case_study": False,
        "source": "omnius_geo_report",
    },
    {
        "content": "AI search traffic converts at 4.4x the rate of traditional organic search. Users coming from AI platforms have higher intent and engagement.",
        "category": "metrics",
        "industries": [],
        "source_section": "Conversion Analysis",
        "confidence": 0.90,
        "has_quantitative_data": True,
        "has_case_study": True,
        "source": "omnius_geo_report",
    },

    # ===== NEW 2025 RESEARCH: EXPERT CONTENT =====
    {
        "content": "LLMs heavily favor content that includes expert commentary and professional insights. Expert quotes signal credibility, particularly when offering unique perspectives not found elsewhere.",
        "category": "citation_optimization",
        "industries": [],
        "source_section": "Expert Content Analysis",
        "confidence": 0.88,
        "has_quantitative_data": False,
        "has_case_study": True,
        "source": "averi_llm_guide",
    },
    {
        "content": "Including specific metrics rather than general observations increases LLM citations by 27%. When publishing original research, always include concrete data points and statistics.",
        "category": "citation_optimization",
        "industries": ["B2B_SaaS", "Technology"],
        "source_section": "Original Research Impact",
        "confidence": 0.90,
        "has_quantitative_data": True,
        "has_case_study": True,
        "source": "averi_llm_guide",
    },
    {
        "content": "Use author bylines with credentials, link to author profiles, and include 'About the Author' sections. E-E-A-T signals (Experience, Expertise, Authoritativeness, Trustworthiness) influence AI citation likelihood.",
        "category": "citation_optimization",
        "industries": ["Healthcare", "Finance", "Legal"],
        "source_section": "E-E-A-T for AI",
        "confidence": 0.85,
        "has_quantitative_data": False,
        "has_case_study": True,
        "source": "storychief_llm_content",
    },

    # ===== NEW 2025 RESEARCH: INDUSTRY-SPECIFIC =====
    {
        "content": "For B2B SaaS content, include product comparison tables, feature matrices, and pricing benchmarks. AI tools frequently cite structured comparison content for software recommendations.",
        "category": "industry_specific",
        "industries": ["B2B_SaaS", "Technology"],
        "source_section": "B2B Content Patterns",
        "confidence": 0.88,
        "has_quantitative_data": False,
        "has_case_study": True,
        "source": "omnius_geo_report",
    },
    {
        "content": "Healthcare content requires peer-reviewed citations within the first 500 words. Medical claims must be backed by studies from recognized institutions to be cited by AI platforms.",
        "category": "industry_specific",
        "industries": ["Healthcare"],
        "source_section": "Healthcare Requirements",
        "confidence": 0.92,
        "has_quantitative_data": True,
        "has_case_study": True,
        "source": "omnius_geo_report",
    },
    {
        "content": "E-commerce product pages should include user reviews, ratings data, and price comparisons. AI shopping assistants prioritize pages with comprehensive product information and social proof.",
        "category": "industry_specific",
        "industries": ["E-commerce", "Retail"],
        "source_section": "E-commerce Optimization",
        "confidence": 0.88,
        "has_quantitative_data": False,
        "has_case_study": True,
        "source": "omnius_geo_report",
    },
    {
        "content": "Financial content must include disclaimers, regulatory references, and date stamps. Include specific numbers (interest rates, returns, fees) rather than vague ranges for AI citation eligibility.",
        "category": "industry_specific",
        "industries": ["Finance", "FinTech"],
        "source_section": "Finance Content Guidelines",
        "confidence": 0.90,
        "has_quantitative_data": False,
        "has_case_study": True,
        "source": "omnius_geo_report",
    },
    {
        "content": "Legal content should cite specific statutes, case law, and jurisdictional information. AI platforms prefer legal content with clear jurisdiction scope and recent case references.",
        "category": "industry_specific",
        "industries": ["Legal"],
        "source_section": "Legal Content Guidelines",
        "confidence": 0.88,
        "has_quantitative_data": False,
        "has_case_study": True,
        "source": "omnius_geo_report",
    },

    # ===== NEW 2025 RESEARCH: CONTENT FORMATS =====
    {
        "content": "Create TL;DR summaries at the beginning of long-form content. AI tools frequently extract these summaries as standalone citations, increasing visibility.",
        "category": "citation_optimization",
        "industries": [],
        "source_section": "Summary Optimization",
        "confidence": 0.85,
        "has_quantitative_data": False,
        "has_case_study": True,
        "source": "storychief_llm_content",
    },
    {
        "content": "Include a clear definition paragraph for key terms within the first 200 words. AI tools prioritize content that provides authoritative definitions.",
        "category": "citation_optimization",
        "industries": [],
        "source_section": "Definition Optimization",
        "confidence": 0.85,
        "has_quantitative_data": False,
        "has_case_study": True,
        "source": "storychief_llm_content",
    },
    {
        "content": "Use numbered step-by-step instructions for how-to content. Each step should be self-contained and extractable as a standalone instruction by AI systems.",
        "category": "citation_optimization",
        "industries": [],
        "source_section": "How-To Format",
        "confidence": 0.88,
        "has_quantitative_data": False,
        "has_case_study": True,
        "source": "averi_llm_guide",
    },
    {
        "content": "Tables with comparative data are highly citable. Include tables that compare features, prices, specifications, or options with clear headers and structured rows.",
        "category": "citation_optimization",
        "industries": [],
        "source_section": "Table Optimization",
        "confidence": 0.88,
        "has_quantitative_data": False,
        "has_case_study": True,
        "source": "averi_llm_guide",
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
