"""
Crawler Node: Fetches and chunks website content.

Uses Crawl4AI to fetch page content and splits into logical chunks
based on heading structure (H1, H2 boundaries).

Supports deep crawling to fetch multiple pages from a website.
"""

import re
import asyncio
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import logging

logger = logging.getLogger(__name__)

# Crawl4AI imports
try:
    from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False
    print("Warning: Crawl4AI not available, using fallback")


@dataclass
class ContentChunk:
    """A chunk of content from a web page."""
    title: str
    content: str
    heading_level: int
    word_count: int
    order: int


async def crawl_url(url: str) -> Tuple[str, Dict]:
    """
    Crawl a URL and return markdown content with metadata.

    Returns:
        Tuple of (markdown_content, metadata_dict)
    """
    import time
    start_time = time.time()

    if not CRAWL4AI_AVAILABLE:
        # Fallback: use requests + beautifulsoup
        import requests
        from bs4 import BeautifulSoup

        response = requests.get(url, timeout=30, headers={
            "User-Agent": "Mozilla/5.0 (compatible; GEOBot/1.0)"
        })
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        # Get text
        text = soup.get_text(separator="\n")
        title = soup.title.string if soup.title else "Untitled"
        crawl_time_ms = int((time.time() - start_time) * 1000)

        return text, {
            "title": title,
            "url": url,
            "word_count": len(text.split()),
            "crawl_time_ms": crawl_time_ms,
            "pages_crawled": 1,
            "content_length": len(text),
            "crawler": "BeautifulSoup (fallback)"
        }

    # Use Crawl4AI
    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=10,
        exclude_external_links=True,
        remove_overlay_elements=True,
    )

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url, config=config)
        crawl_time_ms = int((time.time() - start_time) * 1000)

        if not result.success:
            raise Exception(f"Failed to crawl {url}: {result.error_message}")

        markdown_content = result.markdown or ""
        word_count = len(markdown_content.split())

        metadata = {
            "title": result.metadata.get("title", "Untitled") if result.metadata else "Untitled",
            "url": url,
            "word_count": word_count,
            "crawl_time_ms": crawl_time_ms,
            "pages_crawled": 1,
            "content_length": len(markdown_content),
            "crawler": "Crawl4AI",
            # Additional Crawl4AI stats if available
            "links_count": len(result.links.get("internal", [])) + len(result.links.get("external", [])) if result.links else 0,
            "images_count": len(result.media.get("images", [])) if result.media else 0,
        }

        return markdown_content, metadata


def extract_internal_links(result, base_url: str) -> List[str]:
    """
    Extract internal links from crawl result.

    Filters to only include links from the same domain.
    Prioritizes important pages (about, services, products, etc.)
    """
    if not result.links:
        return []

    base_domain = urlparse(base_url).netloc
    internal_links = []

    # Priority keywords for important pages
    priority_keywords = [
        'about', 'service', 'product', 'solution', 'feature',
        'pricing', 'contact', 'team', 'blog', 'case-stud',
        'portfolio', 'work', 'project', 'client', 'testimonial',
        'faq', 'help', 'support', 'resource', 'guide'
    ]

    all_internal = result.links.get("internal", [])

    # Handle different link formats from Crawl4AI
    links_to_process = []
    for link in all_internal:
        if isinstance(link, dict):
            href = link.get("href", "")
        else:
            href = str(link)
        if href:
            links_to_process.append(href)

    # Score and sort links by priority
    scored_links = []
    for href in links_to_process:
        # Skip anchors, javascript, mailto, etc.
        if href.startswith('#') or href.startswith('javascript:') or href.startswith('mailto:'):
            continue

        # Make absolute URL
        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)

        # Only same domain
        if parsed.netloc != base_domain:
            continue

        # Skip common non-content pages
        skip_patterns = [
            '/login', '/signup', '/register', '/cart', '/checkout',
            '/account', '/admin', '/api/', '/wp-', '/feed', '.pdf',
            '.jpg', '.png', '.gif', '/tag/', '/category/', '/author/',
            '/page/', '?', '#'
        ]
        if any(pattern in full_url.lower() for pattern in skip_patterns):
            continue

        # Score by priority keywords
        score = 0
        url_lower = full_url.lower()
        for keyword in priority_keywords:
            if keyword in url_lower:
                score += 10

        # Prefer shorter URLs (usually more important)
        score -= len(parsed.path) // 10

        scored_links.append((full_url, score))

    # Sort by score (highest first) and deduplicate
    scored_links.sort(key=lambda x: x[1], reverse=True)
    seen = set()
    for url, _ in scored_links:
        # Normalize URL (remove trailing slash for dedup)
        normalized = url.rstrip('/')
        if normalized not in seen:
            seen.add(normalized)
            internal_links.append(url)

    return internal_links


async def deep_crawl_url(
    url: str,
    max_pages: int = 5,
    progress_callback: callable = None
) -> Tuple[List[Dict], Dict]:
    """
    Deep crawl a website - crawl multiple pages.

    Args:
        url: Starting URL (usually homepage)
        max_pages: Maximum number of pages to crawl (default 5)
        progress_callback: Optional callback(current_page, total_pages, url)

    Returns:
        Tuple of (list of page contents, aggregated metadata)
    """
    import time
    start_time = time.time()

    pages_content = []
    crawled_urls: Set[str] = set()
    urls_to_crawl = [url]
    all_internal_links = []

    # Normalize starting URL
    base_domain = urlparse(url).netloc

    if not CRAWL4AI_AVAILABLE:
        # Fallback: just crawl single page
        logger.warning("Crawl4AI not available, falling back to single page")
        markdown, metadata = await crawl_url(url)
        return [{"url": url, "content": markdown, "title": metadata.get("title", "Untitled")}], metadata

    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=10,
        exclude_external_links=True,
        remove_overlay_elements=True,
    )

    async with AsyncWebCrawler() as crawler:
        page_num = 0

        while urls_to_crawl and page_num < max_pages:
            current_url = urls_to_crawl.pop(0)

            # Normalize and check if already crawled
            normalized_url = current_url.rstrip('/')
            if normalized_url in crawled_urls:
                continue

            crawled_urls.add(normalized_url)
            page_num += 1

            if progress_callback:
                progress_callback(page_num, max_pages, current_url)

            logger.info(f"Deep crawl [{page_num}/{max_pages}]: {current_url}")

            try:
                result = await crawler.arun(url=current_url, config=config)

                if not result.success:
                    logger.warning(f"Failed to crawl {current_url}: {result.error_message}")
                    continue

                markdown_content = result.markdown or ""

                if len(markdown_content.split()) < 50:
                    logger.info(f"Skipping {current_url} - too little content")
                    continue

                page_title = "Untitled"
                if result.metadata:
                    page_title = result.metadata.get("title", "Untitled")

                pages_content.append({
                    "url": current_url,
                    "content": markdown_content,
                    "title": page_title,
                    "word_count": len(markdown_content.split())
                })

                # Extract internal links for further crawling
                if page_num < max_pages:
                    new_links = extract_internal_links(result, current_url)
                    for link in new_links:
                        norm_link = link.rstrip('/')
                        if norm_link not in crawled_urls and link not in urls_to_crawl:
                            urls_to_crawl.append(link)

                    all_internal_links.extend(new_links)

            except Exception as e:
                logger.error(f"Error crawling {current_url}: {e}")
                continue

    crawl_time_ms = int((time.time() - start_time) * 1000)

    # Aggregate metadata
    total_words = sum(p["word_count"] for p in pages_content)
    metadata = {
        "title": pages_content[0]["title"] if pages_content else "Untitled",
        "url": url,
        "word_count": total_words,
        "crawl_time_ms": crawl_time_ms,
        "pages_crawled": len(pages_content),
        "content_length": sum(len(p["content"]) for p in pages_content),
        "crawler": "Crawl4AI (deep)",
        "links_found": len(set(all_internal_links)),
        "pages_urls": [p["url"] for p in pages_content],
    }

    return pages_content, metadata


def chunk_by_headings(markdown: str, min_chunk_words: int = 50) -> List[ContentChunk]:
    """
    Split markdown content into chunks based on heading structure.

    Strategy:
    - Split on H1 and H2 headings
    - Merge very small chunks with previous
    - Preserve heading hierarchy
    """
    chunks = []

    # Pattern to match markdown headings
    heading_pattern = r'^(#{1,2})\s+(.+?)$'

    # Split content by headings
    lines = markdown.split('\n')
    current_chunk = []
    current_title = "Introduction"
    current_level = 0
    chunk_order = 0

    for line in lines:
        match = re.match(heading_pattern, line, re.MULTILINE)

        if match:
            # Save previous chunk if it has content
            if current_chunk:
                content = '\n'.join(current_chunk).strip()
                word_count = len(content.split())

                if word_count >= min_chunk_words:
                    chunks.append(ContentChunk(
                        title=current_title,
                        content=content,
                        heading_level=current_level,
                        word_count=word_count,
                        order=chunk_order
                    ))
                    chunk_order += 1
                elif chunks:
                    # Merge with previous chunk if too small
                    chunks[-1].content += f"\n\n{content}"
                    chunks[-1].word_count += word_count

            # Start new chunk
            current_level = len(match.group(1))
            current_title = match.group(2).strip()
            current_chunk = [line]
        else:
            current_chunk.append(line)

    # Don't forget the last chunk
    if current_chunk:
        content = '\n'.join(current_chunk).strip()
        word_count = len(content.split())

        if word_count >= min_chunk_words:
            chunks.append(ContentChunk(
                title=current_title,
                content=content,
                heading_level=current_level,
                word_count=word_count,
                order=chunk_order
            ))
        elif chunks:
            chunks[-1].content += f"\n\n{content}"
            chunks[-1].word_count += word_count

    # If no chunks created, treat entire content as one chunk
    if not chunks and markdown.strip():
        chunks.append(ContentChunk(
            title="Main Content",
            content=markdown.strip(),
            heading_level=0,
            word_count=len(markdown.split()),
            order=0
        ))

    return chunks


async def crawl_and_chunk(
    url: str,
    max_pages: int = 1,
    progress_callback: callable = None
) -> Tuple[List[Dict], Dict]:
    """
    Main entry point: Crawl URL(s) and return chunks ready for database.

    Args:
        url: Starting URL
        max_pages: Number of pages to crawl (1 = single page, >1 = deep crawl)
        progress_callback: Optional callback for deep crawl progress

    Returns:
        Tuple of (list of chunk dicts, metadata dict)
    """
    if max_pages > 1:
        # Deep crawl multiple pages
        pages, metadata = await deep_crawl_url(url, max_pages, progress_callback)

        # Chunk each page's content
        all_chunks = []
        chunk_order = 0

        for page in pages:
            page_chunks = chunk_by_headings(page["content"])

            # Add page URL context to each chunk
            for chunk in page_chunks:
                all_chunks.append({
                    "title": chunk.title,
                    "content": chunk.content,
                    "heading_level": chunk.heading_level,
                    "word_count": chunk.word_count,
                    "order": chunk_order,
                    "source_url": page["url"],
                    "page_title": page["title"]
                })
                chunk_order += 1

        metadata["chunks_generated"] = len(all_chunks)
        return all_chunks, metadata

    else:
        # Single page crawl (original behavior)
        markdown, metadata = await crawl_url(url)

        # Chunk the content
        chunks = chunk_by_headings(markdown)

        # Convert to dicts for database storage
        chunk_dicts = [
            {
                "title": chunk.title,
                "content": chunk.content,
                "heading_level": chunk.heading_level,
                "word_count": chunk.word_count,
                "order": chunk.order,
                "source_url": url,
                "page_title": metadata.get("title", "Untitled")
            }
            for chunk in chunks
        ]

        metadata["chunks_generated"] = len(chunk_dicts)
        return chunk_dicts, metadata


# Synchronous wrapper for non-async contexts
def crawl_and_chunk_sync(
    url: str,
    max_pages: int = 1,
    progress_callback: callable = None
) -> Tuple[List[Dict], Dict]:
    """Synchronous wrapper for crawl_and_chunk."""
    return asyncio.run(crawl_and_chunk(url, max_pages, progress_callback))


if __name__ == "__main__":
    # Test the crawler
    import sys

    test_url = sys.argv[1] if len(sys.argv) > 1 else "https://example.com"

    async def test():
        chunks, metadata = await crawl_and_chunk(test_url)
        print(f"URL: {metadata.get('title', 'Unknown')}")
        print(f"Chunks found: {len(chunks)}")
        for chunk in chunks:
            print(f"  - {chunk['title']}: {chunk['word_count']} words")

    asyncio.run(test())
