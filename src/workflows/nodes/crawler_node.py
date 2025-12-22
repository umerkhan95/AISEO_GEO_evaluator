"""
Crawler Node: Fetches and chunks website content.

Uses Crawl4AI to fetch page content and splits into logical chunks
based on heading structure (H1, H2 boundaries).
"""

import re
import asyncio
from typing import List, Dict, Tuple
from dataclasses import dataclass

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


async def crawl_and_chunk(url: str) -> Tuple[List[Dict], Dict]:
    """
    Main entry point: Crawl URL and return chunks ready for database.

    Returns:
        Tuple of (list of chunk dicts, metadata dict)
    """
    # Crawl the URL
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
            "order": chunk.order
        }
        for chunk in chunks
    ]

    return chunk_dicts, metadata


# Synchronous wrapper for non-async contexts
def crawl_and_chunk_sync(url: str) -> Tuple[List[Dict], Dict]:
    """Synchronous wrapper for crawl_and_chunk."""
    return asyncio.run(crawl_and_chunk(url))


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
