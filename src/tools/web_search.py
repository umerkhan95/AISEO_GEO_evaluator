"""
Web search and crawling tools for finding GEO/SEO scholarly papers using Crawl4AI.

These tools enable the Deep Agent to:
1. Search for scholarly papers on GEO (Generative Engine Optimization)
2. Search for SEO research papers and guides
3. Crawl and extract content from web pages
4. Find PDFs and downloadable resources
"""

import asyncio
import json
from typing import Optional
from urllib.parse import urlparse, quote_plus

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.content_filter_strategy import BM25ContentFilter, PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator


class WebSearchClient:
    """Crawl4AI-based web search client for GEO/SEO research."""

    def __init__(self):
        self.browser_config = BrowserConfig(
            headless=True,
            verbose=False,
        )

    async def search_google(
        self,
        query: str,
        max_results: int = 10,
        site_filter: Optional[str] = None
    ) -> list[dict]:
        """
        Search Google for scholarly papers and extract results.

        Args:
            query: Search query
            max_results: Maximum results to return
            site_filter: Optional site filter (e.g., "arxiv.org")

        Returns:
            List of search results with title, url, and snippet
        """
        # Build Google search URL
        search_query = query
        if site_filter:
            search_query = f"site:{site_filter} {query}"

        encoded_query = quote_plus(search_query)
        google_url = f"https://www.google.com/search?q={encoded_query}&num={max_results}"

        results = []

        async with AsyncWebCrawler(config=self.browser_config) as crawler:
            run_config = CrawlerRunConfig(
                word_count_threshold=10,
                excluded_tags=["nav", "footer", "header", "aside"],
            )

            result = await crawler.arun(url=google_url, config=run_config)

            if result.success:
                # Parse search results from HTML
                # Google results are in specific div structures
                results = self._parse_google_results(result.html, max_results)

        return results

    def _parse_google_results(self, html: str, max_results: int) -> list[dict]:
        """Parse Google search results from HTML."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, 'html.parser')
        results = []

        # Find search result divs
        for g in soup.find_all('div', class_='g')[:max_results]:
            try:
                # Extract title and URL
                title_elem = g.find('h3')
                link_elem = g.find('a', href=True)
                snippet_elem = g.find('div', class_='VwiC3b') or g.find('span', class_='aCOpRe')

                if title_elem and link_elem:
                    url = link_elem['href']
                    # Skip Google's redirect URLs
                    if url.startswith('/url?'):
                        continue
                    if not url.startswith('http'):
                        continue

                    results.append({
                        'title': title_elem.get_text(strip=True),
                        'url': url,
                        'snippet': snippet_elem.get_text(strip=True) if snippet_elem else '',
                    })
            except Exception:
                continue

        return results

    async def crawl_page(
        self,
        url: str,
        extract_links: bool = True,
        use_fit_markdown: bool = True
    ) -> dict:
        """
        Crawl a web page and extract content.

        Args:
            url: URL to crawl
            extract_links: Whether to extract links from the page
            use_fit_markdown: Whether to use BM25-filtered markdown

        Returns:
            Dictionary with page content, links, and metadata
        """
        async with AsyncWebCrawler(config=self.browser_config) as crawler:
            # Configure markdown generation with optional BM25 filtering
            if use_fit_markdown:
                md_generator = DefaultMarkdownGenerator(
                    content_filter=BM25ContentFilter(
                        user_query=None,  # Will be set per-request
                        bm25_threshold=1.0
                    )
                )
            else:
                md_generator = DefaultMarkdownGenerator(
                    content_filter=PruningContentFilter(
                        threshold=0.5,
                        threshold_type="fixed"
                    )
                )

            run_config = CrawlerRunConfig(
                word_count_threshold=50,
                markdown_generator=md_generator,
                excluded_tags=["nav", "footer", "header", "aside", "script", "style"],
            )

            result = await crawler.arun(url=url, config=run_config)

            if result.success:
                return {
                    'url': url,
                    'title': result.metadata.get('title', ''),
                    'markdown': result.markdown,
                    'fit_markdown': result.fit_markdown if hasattr(result, 'fit_markdown') else result.markdown,
                    'links': result.links if extract_links else [],
                    'media': result.media if hasattr(result, 'media') else {},
                    'success': True,
                }
            else:
                return {
                    'url': url,
                    'error': result.error_message if hasattr(result, 'error_message') else 'Unknown error',
                    'success': False,
                }


# Global client instance
_client: Optional[WebSearchClient] = None


def _get_client() -> WebSearchClient:
    """Get or create the web search client."""
    global _client
    if _client is None:
        _client = WebSearchClient()
    return _client


def search_geo_seo_papers(
    query: str,
    max_results: int = 10,
    include_domains: Optional[list[str]] = None,
    search_depth: str = "advanced"
) -> str:
    """
    Search for scholarly papers and articles about GEO (Generative Engine Optimization)
    and SEO (Search Engine Optimization).

    This tool searches academic sources, research papers, industry guides, and
    authoritative articles about GEO/SEO optimization strategies.

    Args:
        query: Search query (e.g., "GEO optimization strategies for B2B SaaS")
        max_results: Maximum number of results to return (default: 10)
        include_domains: Optional list of domains to prioritize
                        (e.g., ["arxiv.org", "researchgate.net", "scholar.google.com"])
        search_depth: "basic" or "advanced" search depth (default: "advanced")

    Returns:
        JSON string containing search results with titles, URLs, and snippets
    """
    client = _get_client()

    # Default academic/research domains for GEO/SEO papers
    default_domains = [
        "arxiv.org",
        "researchgate.net",
        "semanticscholar.org",
        "papers.ssrn.com",
        "dl.acm.org",
        "searchengineland.com",
        "moz.com",
        "ahrefs.com",
        "backlinko.com",
        "searchenginejournal.com",
    ]

    domains = include_domains or default_domains

    # Enhance query for scholarly focus
    enhanced_query = f"{query} scholarly paper research PDF"

    async def _search():
        all_results = []

        # Search across priority domains
        for domain in domains[:5]:  # Limit to top 5 domains for performance
            try:
                results = await client.search_google(
                    query=enhanced_query,
                    max_results=max_results // 2,
                    site_filter=domain
                )
                for r in results:
                    r['source_domain'] = domain
                all_results.extend(results)
            except Exception:
                continue

        # Also do a general search
        try:
            general_results = await client.search_google(
                query=enhanced_query,
                max_results=max_results
            )
            for r in general_results:
                r['source_domain'] = _extract_domain(r.get('url', ''))
            all_results.extend(general_results)
        except Exception:
            pass

        # Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for r in all_results:
            url = r.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(r)

        return unique_results[:max_results]

    try:
        results = asyncio.run(_search())

        formatted_results = {
            "query": query,
            "enhanced_query": enhanced_query,
            "answer": "",
            "results": []
        }

        for result in results:
            url = result.get('url', '')
            formatted_results["results"].append({
                "title": result.get("title", ""),
                "url": url,
                "snippet": result.get("snippet", "")[:500],
                "score": 1.0,  # Crawl4AI doesn't provide relevance scores
                "source_domain": result.get("source_domain", _extract_domain(url)),
                "is_pdf": url.lower().endswith(".pdf"),
            })

        return json.dumps(formatted_results, indent=2)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "query": query,
            "results": []
        })


def search_scholarly_articles(
    topic: str,
    industry: Optional[str] = None,
    year_from: int = 2023,
    max_results: int = 10
) -> str:
    """
    Search for scholarly articles on a specific GEO/SEO topic, optionally filtered by industry.

    This is a more targeted search specifically for academic and research content.

    Args:
        topic: The main topic to search for (e.g., "citation optimization", "AI search rankings")
        industry: Optional industry filter (e.g., "healthcare", "B2B SaaS", "e-commerce")
        year_from: Only include results from this year onwards (default: 2023)
        max_results: Maximum results to return (default: 10)

    Returns:
        JSON string with scholarly article results including PDF links where available
    """
    client = _get_client()

    # Build query based on topic and industry
    if industry:
        query = f"{topic} {industry} industry GEO SEO research paper {year_from}"
    else:
        query = f"{topic} GEO SEO generative engine optimization research paper {year_from}"

    # Academic-focused domains
    academic_domains = [
        "arxiv.org",
        "researchgate.net",
        "semanticscholar.org",
        "papers.ssrn.com",
        "dl.acm.org",
        "ieee.org",
        "springer.com",
        "sciencedirect.com",
    ]

    async def _search():
        all_results = []

        # Search academic domains
        for domain in academic_domains[:4]:
            try:
                results = await client.search_google(
                    query=query,
                    max_results=max_results // 2,
                    site_filter=domain
                )
                for r in results:
                    r['source_domain'] = domain
                all_results.extend(results)
            except Exception:
                continue

        # Deduplicate
        seen_urls = set()
        unique_results = []
        for r in all_results:
            url = r.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(r)

        return unique_results[:max_results]

    try:
        results = asyncio.run(_search())

        formatted_results = {
            "topic": topic,
            "industry": industry,
            "year_filter": year_from,
            "query_used": query,
            "summary": "",
            "articles": []
        }

        for result in results:
            url = result.get("url", "")
            article = {
                "title": result.get("title", ""),
                "url": url,
                "snippet": result.get("snippet", "")[:500],
                "score": 1.0,
                "is_pdf": url.lower().endswith(".pdf"),
                "has_pdf_link": "pdf" in url.lower() or "arxiv" in url.lower(),
                "source_domain": result.get("source_domain", _extract_domain(url)),
            }
            formatted_results["articles"].append(article)

        # Sort by likely relevance (PDFs and arxiv first)
        formatted_results["articles"].sort(
            key=lambda x: (x["is_pdf"], x["has_pdf_link"]),
            reverse=True
        )

        return json.dumps(formatted_results, indent=2)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "topic": topic,
            "articles": []
        })


def search_pdf_resources(
    query: str,
    max_results: int = 10
) -> str:
    """
    Search specifically for PDF resources and downloadable papers.

    This tool focuses on finding direct PDF links for GEO/SEO research.

    Args:
        query: Search query for PDF resources
        max_results: Maximum results to return

    Returns:
        JSON string with PDF-focused search results
    """
    client = _get_client()

    # Add PDF-specific terms to query
    pdf_query = f"{query} filetype:pdf"

    async def _search():
        all_results = []

        # Search arxiv specifically
        try:
            arxiv_results = await client.search_google(
                query=f"{query} research paper",
                max_results=max_results,
                site_filter="arxiv.org"
            )
            all_results.extend(arxiv_results)
        except Exception:
            pass

        # Search researchgate
        try:
            rg_results = await client.search_google(
                query=query,
                max_results=max_results // 2,
                site_filter="researchgate.net"
            )
            all_results.extend(rg_results)
        except Exception:
            pass

        # General PDF search
        try:
            pdf_results = await client.search_google(
                query=pdf_query,
                max_results=max_results
            )
            all_results.extend(pdf_results)
        except Exception:
            pass

        # Deduplicate
        seen_urls = set()
        unique_results = []
        for r in all_results:
            url = r.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(r)

        return unique_results[:max_results]

    try:
        results = asyncio.run(_search())

        pdf_results = {
            "query": query,
            "pdf_focused_query": pdf_query,
            "pdfs_found": []
        }

        for result in results:
            url = result.get("url", "")

            # Check if it's likely a PDF or leads to a PDF
            is_direct_pdf = url.lower().endswith(".pdf")
            is_arxiv = "arxiv.org" in url.lower()
            is_researchgate = "researchgate.net" in url.lower()

            # For arxiv, convert to PDF URL
            pdf_url = url
            if is_arxiv and "/abs/" in url:
                pdf_url = url.replace("/abs/", "/pdf/") + ".pdf"

            pdf_results["pdfs_found"].append({
                "title": result.get("title", ""),
                "url": url,
                "pdf_url": pdf_url if (is_direct_pdf or is_arxiv) else None,
                "snippet": result.get("snippet", "")[:300],
                "score": 1.0,
                "source_type": "direct_pdf" if is_direct_pdf else
                              "arxiv" if is_arxiv else
                              "researchgate" if is_researchgate else "webpage",
            })

        # Sort PDFs first
        pdf_results["pdfs_found"].sort(
            key=lambda x: x["source_type"] in ["direct_pdf", "arxiv"],
            reverse=True
        )

        return json.dumps(pdf_results, indent=2)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "query": query,
            "pdfs_found": []
        })


async def crawl_and_extract(
    url: str,
    query: Optional[str] = None
) -> str:
    """
    Crawl a specific URL and extract its content as clean markdown.

    This is useful for extracting full content from pages found via search.

    Args:
        url: The URL to crawl
        query: Optional query to use for BM25 filtering (extracts most relevant content)

    Returns:
        JSON string with extracted content
    """
    client = _get_client()

    try:
        result = await client.crawl_page(
            url=url,
            extract_links=True,
            use_fit_markdown=query is not None
        )

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "url": url,
            "success": False
        })


def _extract_domain(url: str) -> str:
    """Extract domain from URL."""
    try:
        parsed = urlparse(url)
        return parsed.netloc
    except Exception:
        return ""
