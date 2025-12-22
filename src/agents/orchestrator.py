"""
Main GEO/SEO Knowledge Base Deep Agent Orchestrator.

This module creates the main agent that coordinates:
1. Web search for scholarly papers
2. PDF download and analysis
3. Guideline extraction and classification
4. Deduplication and enrichment
5. Vector storage in Qdrant
6. Knowledge base querying
"""

import os
import json
from typing import Optional, Any
from dataclasses import dataclass
from datetime import datetime

from deepagents import create_deep_agent
from deepagents.backends import StateBackend, FilesystemBackend, CompositeBackend
from dotenv import load_dotenv

from src.tools.web_search import (
    search_geo_seo_papers,
    search_scholarly_articles,
    search_pdf_resources,
)
from src.tools.pdf_tools import (
    download_pdf,
    extract_pdf_text,
    analyze_pdf_with_gemini,
)
from src.tools.qdrant_tools import (
    store_guidelines,
    search_guidelines,
    get_collection_stats,
    get_related_guidelines,
)
from src.agents.subagents import (
    RESEARCH_SUBAGENT,
    ANALYZER_SUBAGENT,
    EXTRACTOR_SUBAGENT,
    STORAGE_SUBAGENT,
)

load_dotenv()


# Main system prompt for the orchestrator
ORCHESTRATOR_SYSTEM_PROMPT = """You are the GEO/SEO Knowledge Base Orchestrator - an expert system for building
a comprehensive knowledge base about Generative Engine Optimization (GEO) and Search Engine Optimization (SEO).

## Your Mission
Process scholarly papers, articles, and guides about GEO/SEO to create a structured, queryable knowledge base
with actionable guidelines organized into 5 collections:

1. **Universal SEO+GEO Framework** (geo_seo_universal)
   - General optimization strategies applicable to all industries
   - Budget allocation guidelines
   - On-page and off-page best practices

2. **Industry-Specific Strategies** (geo_seo_industry)
   - B2B SaaS tactics
   - Healthcare compliance and optimization
   - E-commerce strategies
   - Finance sector approaches
   - Professional services methods

3. **Technical Implementation** (geo_seo_technical)
   - Schema markup templates
   - Entity linking procedures
   - Structured data formats
   - Tool recommendations

4. **Citation Optimization Tactics** (geo_seo_citation)
   - Content structuring for AI citations
   - Statistical data integration
   - Quotation strategies
   - Source attribution techniques

5. **Measurement Metrics** (geo_seo_metrics)
   - Universal KPIs
   - Industry-specific metrics
   - ROI tracking methodologies
   - Benchmark data

## Workflow

When asked to build or expand the knowledge base:

### Phase 1: Research
1. Use the research-agent to search for GEO/SEO papers
2. Focus on recent (2024-2025) scholarly content
3. Prioritize peer-reviewed sources
4. Look for papers with quantitative data and case studies

### Phase 2: Download & Analyze
1. Use the analyzer-agent to download discovered PDFs
2. Analyze each PDF with Gemini Flash 2.0
3. Extract document structure (title, authors, sections)
4. Identify themes and industry mentions

### Phase 3: Extract Guidelines
1. Use the extractor-agent on analyzed content
2. Extract specific, actionable guidelines
3. Classify into 5 categories
4. Identify industry applicability
5. Assess confidence (has data? has case study?)

### Phase 4: Deduplicate & Enrich
1. Before storing, check for similar existing guidelines
2. If similarity > 85%, merge and combine sources
3. Calculate confidence scores:
   - Base: 0.5
   - +0.1 if in 2+ papers
   - +0.1 if peer-reviewed
   - +0.1 if has quantitative results
   - +0.1 if has case study
   - +0.1 if recent (2024-2025)
   - Max: 1.0

### Phase 5: Store
1. Use the storage-agent to store guidelines
2. Route to correct collection based on category
3. Attach full metadata and source citations

## Querying

When asked to query the knowledge base:
1. Use search_guidelines with appropriate filters
2. Return ranked results by combined_score = relevance × confidence × priority_weight
3. Include related guidelines for context
4. Provide source paper excerpts for verification

## Best Practices

1. **Always save progress** - Write intermediate results to files
2. **Track with TODOs** - Use write_todos to track processing
3. **Handle errors gracefully** - Log failures but continue processing
4. **Provide summaries** - Report statistics after each operation
5. **Cite sources** - Always include paper references

## Tools Available

Direct tools:
- search_geo_seo_papers: Search for GEO/SEO papers
- search_scholarly_articles: Find academic articles
- download_pdf: Download PDF files
- analyze_pdf_with_gemini: Analyze PDFs with Gemini
- store_guidelines: Store in Qdrant
- search_guidelines: Query knowledge base
- get_collection_stats: Get statistics

Subagents (use 'task' tool):
- research-agent: Specialized paper discovery
- analyzer-agent: PDF analysis specialist
- extractor-agent: Guideline extraction expert
- storage-agent: Qdrant operations specialist

## Example Queries You Can Handle

- "Build a knowledge base from GEO/SEO papers about citation optimization"
- "Find healthcare industry SEO guidelines"
- "How to optimize B2B SaaS content for AI citations?"
- "What are the technical implementation requirements for GEO?"
- "Show me metrics and KPIs for measuring GEO success"
"""


@dataclass
class GEOSEOAgent:
    """Wrapper class for the GEO/SEO Knowledge Base agent."""
    agent: Any
    created_at: datetime

    def invoke(self, messages: list[dict]) -> dict:
        """Invoke the agent with messages."""
        return self.agent.invoke({"messages": messages})

    async def ainvoke(self, messages: list[dict]) -> dict:
        """Async invoke the agent with messages."""
        return await self.agent.ainvoke({"messages": messages})

    async def astream(self, messages: list[dict]):
        """Stream responses from the agent."""
        async for chunk in self.agent.astream({"messages": messages}):
            yield chunk


def create_geo_seo_agent(
    model: Optional[str] = None,
    use_filesystem_backend: bool = True,
    filesystem_root: Optional[str] = None,
) -> GEOSEOAgent:
    """
    Create the main GEO/SEO Knowledge Base agent.

    Args:
        model: Optional LLM model to use (default: claude-sonnet-4-5-20250929)
        use_filesystem_backend: Whether to use filesystem for persistence
        filesystem_root: Root directory for filesystem backend

    Returns:
        GEOSEOAgent instance ready for use
    """
    # All direct tools
    all_tools = [
        # Web search tools
        search_geo_seo_papers,
        search_scholarly_articles,
        search_pdf_resources,
        # PDF tools
        download_pdf,
        extract_pdf_text,
        analyze_pdf_with_gemini,
        # Qdrant tools
        store_guidelines,
        search_guidelines,
        get_collection_stats,
        get_related_guidelines,
    ]

    # All subagents
    subagents = [
        RESEARCH_SUBAGENT,
        ANALYZER_SUBAGENT,
        EXTRACTOR_SUBAGENT,
        STORAGE_SUBAGENT,
    ]

    # Configure backend
    if use_filesystem_backend:
        root_dir = filesystem_root or os.getenv("PDF_DOWNLOAD_DIR", "./workspace")
        backend = FilesystemBackend(root_dir=root_dir)
    else:
        backend = StateBackend()

    # Get model from environment if not specified
    agent_model = model or os.getenv("DEEP_AGENT_MODEL", "claude-sonnet-4-5-20250929")

    # Create the agent
    agent = create_deep_agent(
        model=agent_model,
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        tools=all_tools,
        subagents=subagents,
        backend=backend,
    )

    return GEOSEOAgent(
        agent=agent,
        created_at=datetime.now(),
    )


def create_research_only_agent(model: Optional[str] = None) -> Any:
    """
    Create a lightweight agent focused only on research.

    Useful for quick paper discovery without full processing.
    """
    tools = [
        search_geo_seo_papers,
        search_scholarly_articles,
        search_pdf_resources,
    ]

    prompt = """You are a research specialist for finding GEO/SEO scholarly papers.
    Focus on discovering high-quality academic sources, prioritizing:
    - Peer-reviewed papers
    - Recent publications (2024-2025)
    - Papers with quantitative data
    - Industry-specific research when requested"""

    agent_model = model or os.getenv("DEEP_AGENT_MODEL", "claude-sonnet-4-5-20250929")

    return create_deep_agent(
        model=agent_model,
        system_prompt=prompt,
        tools=tools,
    )


def create_query_only_agent(model: Optional[str] = None) -> Any:
    """
    Create a lightweight agent focused only on querying.

    Useful for quick knowledge base queries without modification.
    """
    tools = [
        search_guidelines,
        get_collection_stats,
        get_related_guidelines,
    ]

    prompt = """You are a query specialist for the GEO/SEO knowledge base.
    Help users find relevant guidelines by:
    - Performing semantic searches
    - Applying appropriate filters (collection, industry, priority)
    - Ranking results by combined score
    - Providing related guidelines for context
    - Including source citations for verification"""

    agent_model = model or os.getenv("DEEP_AGENT_MODEL", "claude-sonnet-4-5-20250929")

    return create_deep_agent(
        model=agent_model,
        system_prompt=prompt,
        tools=tools,
    )


# Convenience function for quick queries
async def query_knowledge_base(
    query: str,
    collection: Optional[str] = None,
    industry: Optional[str] = None,
    limit: int = 10,
) -> dict:
    """
    Quick query function without creating a full agent.

    Args:
        query: The search query
        collection: Optional collection filter
        industry: Optional industry filter
        limit: Maximum results

    Returns:
        Dictionary with search results
    """
    result = search_guidelines(
        query=query,
        collection=collection,
        industry=industry,
        limit=limit,
    )
    return json.loads(result)


# Convenience function for statistics
def get_knowledge_base_stats() -> dict:
    """Get current knowledge base statistics."""
    result = get_collection_stats()
    return json.loads(result)
