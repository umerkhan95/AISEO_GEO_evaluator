"""
Website Optimizer Workflow: Main orchestration for GEO optimization.

Coordinates the entire flow:
1. Crawl URL and chunk content
2. Classify industry
3. Process chunks in parallel
4. Assemble final output
"""

import os
import sys
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from database import (
    create_job, update_job, get_job,
    create_chunks_batch, get_job_chunks, get_pending_chunks,
    update_chunk, mark_chunk_completed,
    record_guideline_application,
    save_final_result,
)

from src.workflows.nodes.crawler_node import crawl_and_chunk_sync
from src.workflows.nodes.industry_classifier import classify_industry
from src.workflows.nodes.guideline_retriever import retrieve_guidelines, get_retriever
from src.workflows.nodes.chunk_optimizer import optimize_chunk, score_original_content
from src.workflows.nodes.humanizer_node import humanize_content, quick_humanize


class WebsiteOptimizer:
    """
    Main orchestrator for website GEO optimization.

    Handles the complete workflow from URL input to final output.
    """

    def __init__(self, max_workers: int = 3):
        """
        Initialize optimizer.

        Args:
            max_workers: Max parallel chunk processors (limited by API rate limits)
        """
        self.max_workers = max_workers
        self.retriever = get_retriever()

    def optimize_url(
        self,
        url: str,
        settings: Dict = None,
        progress_callback: callable = None,
        job_id: str = None
    ) -> str:
        """
        Main entry point: Optimize a URL for GEO.

        Args:
            url: The URL to optimize
            settings: Optional settings dict
            progress_callback: Optional callback(job_id, status, progress_pct, message)
            job_id: Optional existing job_id (if not provided, creates new job)

        Returns:
            job_id for tracking
        """
        settings = settings or {}

        # Use existing job_id or create new one
        if job_id is None:
            job_id = create_job(url, settings)
        self._notify(progress_callback, job_id, "started", 0, "Job created")

        try:
            # Step 1: Crawl and chunk
            self._notify(progress_callback, job_id, "crawling", 5, "Crawling website...")
            update_job(job_id, status="crawling")

            chunks, metadata = crawl_and_chunk_sync(url)

            if not chunks:
                raise Exception("No content extracted from URL")

            # Save crawl metadata
            crawl_stats = {
                "title": metadata.get("title", "Unknown"),
                "word_count": metadata.get("word_count", 0),
                "crawl_time_ms": metadata.get("crawl_time_ms", 0),
                "pages_crawled": metadata.get("pages_crawled", 1),
                "content_length": metadata.get("content_length", 0),
                "crawler": metadata.get("crawler", "Unknown"),
                "links_count": metadata.get("links_count", 0),
                "images_count": metadata.get("images_count", 0),
                "chunks_generated": len(chunks),
            }
            update_job(job_id, crawl_metadata=json.dumps(crawl_stats))

            self._notify(progress_callback, job_id, "crawling", 15,
                        f"Extracted {len(chunks)} content sections")

            # Step 2: Classify industry
            self._notify(progress_callback, job_id, "classifying", 20, "Detecting industry...")
            update_job(job_id, status="classifying")

            combined_content = "\n".join([c["content"] for c in chunks[:3]])
            industry, confidence, secondary = classify_industry(combined_content, url)

            update_job(
                job_id,
                industry=industry,
                industry_confidence=confidence,
            )

            self._notify(progress_callback, job_id, "classifying", 25,
                        f"Detected: {industry} ({confidence:.0%} confidence)")

            # Step 3: Save chunks to database
            chunk_ids = create_chunks_batch(job_id, chunks)
            update_job(job_id, total_chunks=len(chunk_ids))

            # Score original content
            original_scores = []
            for chunk in chunks:
                score = score_original_content(chunk["content"], chunk["title"])
                original_scores.append(score)

            avg_original = sum(original_scores) / len(original_scores) if original_scores else 0
            update_job(job_id, original_geo_score=avg_original)

            # Step 4: Process chunks in parallel
            self._notify(progress_callback, job_id, "processing", 30,
                        f"Processing {len(chunks)} chunks...")
            update_job(job_id, status="processing")

            self._process_chunks_parallel(
                job_id, chunk_ids, chunks, industry, progress_callback
            )

            # Step 5: Assemble final output
            self._notify(progress_callback, job_id, "assembling", 90, "Assembling final output...")
            update_job(job_id, status="assembling")

            self._assemble_output(job_id, metadata, url, industry)

            # Done!
            self._notify(progress_callback, job_id, "completed", 100, "Optimization complete!")
            update_job(job_id, status="completed", completed_at=datetime.now().isoformat())

            return job_id

        except Exception as e:
            error_msg = str(e)
            update_job(job_id, status="failed", error_message=error_msg)
            self._notify(progress_callback, job_id, "failed", 0, f"Error: {error_msg}")
            raise

    def _process_chunks_parallel(
        self,
        job_id: str,
        chunk_ids: List[str],
        chunks: List[Dict],
        industry: str,
        progress_callback: callable
    ):
        """Process chunks in parallel with ThreadPoolExecutor."""

        completed = 0
        total = len(chunks)

        def process_single_chunk(chunk_id: str, chunk_data: Dict) -> bool:
            """Process a single chunk."""
            try:
                update_chunk(chunk_id, status="processing")

                # Retrieve guidelines for this chunk
                guidelines = retrieve_guidelines(chunk_data["content"], industry)

                # Optimize chunk
                optimized, geo_score, applied, time_ms = optimize_chunk(
                    original_content=chunk_data["content"],
                    section_title=chunk_data["title"],
                    guidelines=guidelines,
                    industry=industry
                )

                # Humanize the output
                humanized, naturalness = humanize_content(optimized, industry)

                # Record applied guidelines
                for g in applied:
                    record_guideline_application(
                        chunk_id=chunk_id,
                        job_id=job_id,
                        guideline_id=g["guideline_id"],
                        guideline_content=g["guideline_content"],
                        guideline_category=g["guideline_category"],
                        how_applied=g.get("how_applied", "Applied during optimization")
                    )

                # Save results
                mark_chunk_completed(
                    chunk_id=chunk_id,
                    optimized_content=humanized,
                    geo_score_after=geo_score,
                    guidelines_applied=[g["guideline_id"] for g in applied],
                    processing_time_ms=time_ms
                )

                return True

            except Exception as e:
                update_chunk(chunk_id, status="failed", error_message=str(e))
                return False

        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(process_single_chunk, chunk_id, chunk_data): chunk_id
                for chunk_id, chunk_data in zip(chunk_ids, chunks)
            }

            for future in as_completed(futures):
                completed += 1
                progress = 30 + int((completed / total) * 55)  # 30-85%
                self._notify(
                    progress_callback, job_id, "processing", progress,
                    f"Processed {completed}/{total} chunks"
                )

    def _assemble_output(
        self,
        job_id: str,
        metadata: Dict,
        url: str,
        industry: str
    ):
        """Assemble final markdown output from processed chunks."""

        chunks = get_job_chunks(job_id)

        # Build markdown
        md_parts = []

        # Header
        title = metadata.get("title", "Optimized Content")
        md_parts.append(f"# {title}\n")
        md_parts.append(f"> *GEO-optimized content for {industry}*\n")
        md_parts.append(f"> *Source: {url}*\n")
        md_parts.append("---\n")

        # Collect scores
        scores = []

        # Add each chunk
        for chunk in chunks:
            if chunk["status"] == "completed" and chunk["optimized_content"]:
                md_parts.append(chunk["optimized_content"])
                md_parts.append("\n\n")
                scores.append(chunk["geo_score_after"])

        # Calculate final score
        avg_score = sum(scores) / len(scores) if scores else 0

        # Build report
        job_data = get_job(job_id)
        report = {
            "job_id": job_id,
            "url": url,
            "industry": industry,
            "industry_confidence": job_data.get("industry_confidence", 0),
            "scores": {
                "original_geo_score": job_data.get("original_geo_score", 0),
                "optimized_geo_score": avg_score,
                "improvement_pct": ((avg_score - job_data.get("original_geo_score", 0)) /
                                   max(job_data.get("original_geo_score", 1), 1)) * 100,
            },
            "chunks_processed": len([c for c in chunks if c["status"] == "completed"]),
            "total_chunks": len(chunks),
            "timestamp": datetime.now().isoformat(),
        }

        # Update job with final score
        update_job(job_id, optimized_geo_score=avg_score)

        # Build HTML version
        html = self._markdown_to_html("\n".join(md_parts))

        # Save final result
        save_final_result(
            job_id=job_id,
            markdown="\n".join(md_parts),
            html=html,
            report=report,
            guidelines_count=sum(len(json.loads(c.get("guidelines_applied", "[]")))
                                for c in chunks)
        )

    def _markdown_to_html(self, markdown: str) -> str:
        """Convert markdown to HTML (basic conversion)."""
        try:
            import markdown as md
            return md.markdown(markdown, extensions=['tables', 'fenced_code'])
        except ImportError:
            # Basic fallback
            import re
            html = markdown
            html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
            html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
            html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
            html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
            html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
            html = re.sub(r'^- (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
            html = html.replace('\n\n', '</p><p>')
            return f'<div class="geo-content"><p>{html}</p></div>'

    def _notify(self, callback, job_id, status, progress, message):
        """Send progress notification if callback provided."""
        if callback:
            callback(job_id, status, progress, message)
        print(f"[{job_id}] {status} ({progress}%): {message}")


# Convenience function
def optimize_website(url: str, settings: Dict = None) -> str:
    """
    Convenience function to optimize a website URL.

    Returns job_id for tracking.
    """
    optimizer = WebsiteOptimizer()
    return optimizer.optimize_url(url, settings)


if __name__ == "__main__":
    # Test the optimizer
    import sys

    if len(sys.argv) < 2:
        print("Usage: python website_optimizer.py <url>")
        sys.exit(1)

    url = sys.argv[1]
    print(f"Optimizing: {url}")

    job_id = optimize_website(url)
    print(f"\nJob completed: {job_id}")

    # Print results
    from database import get_result, get_job
    result = get_result(job_id)
    job = get_job(job_id)

    if result:
        print(f"\n=== Results ===")
        print(f"Original Score: {job['original_geo_score']:.1f}/10")
        print(f"Optimized Score: {job['optimized_geo_score']:.1f}/10")
        print(f"Improvement: {result['report_json']['scores']['improvement_pct']:.1f}%")
        print(f"\nMarkdown length: {len(result['final_markdown'])} chars")
