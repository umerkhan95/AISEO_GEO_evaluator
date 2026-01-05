"""
API v2: Website Optimization Endpoints.

New endpoints for the GEO Website Optimizer (Phase 2).
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Optional, List
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
import logging

from database import (
    init_database, get_job, get_all_jobs, get_job_chunks,
    get_result, get_job_guidelines_applied, get_stats
)
from src.workflows.website_optimizer import WebsiteOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# URL Validation
# ============================================================================

# Blocked placeholder/test domains that shouldn't be crawled
BLOCKED_DOMAINS = {
    "example.com", "example.org", "example.net",
    "test.com", "test.org", "test.net",
    "localhost", "127.0.0.1", "0.0.0.0",
    "foo.com", "bar.com", "baz.com",
    "sample.com", "demo.test", "fake.com",
    "domain.com", "website.com", "mysite.com",
    "yoursite.com", "placeholder.com",
}


def normalize_url(url: str) -> str:
    """Normalize URL for comparison (lowercase, strip trailing slash, remove www)."""
    url = url.lower().strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    parsed = urlparse(url)
    domain = parsed.netloc.replace("www.", "")
    path = parsed.path.rstrip("/") or ""
    return f"{domain}{path}"


def extract_domain(url: str) -> str:
    """Extract domain from URL (without www prefix)."""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    parsed = urlparse(url)
    return parsed.netloc.replace("www.", "").lower()


def validate_url(url: str) -> tuple[bool, str]:
    """
    Validate URL for crawling.

    Returns (is_valid, error_message).
    """
    if not url or not url.strip():
        return False, "URL is required"

    # Add protocol if missing
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        parsed = urlparse(url)
    except Exception:
        return False, "Invalid URL format"

    # Must have a valid domain
    if not parsed.netloc:
        return False, "Invalid URL: no domain found"

    domain = parsed.netloc.replace("www.", "").lower()

    # Check for blocked placeholder domains
    base_domain = domain.split(":")[0]  # Remove port if present
    if base_domain in BLOCKED_DOMAINS:
        return False, f"Cannot crawl placeholder domain '{base_domain}'. Please provide a real website URL."

    # Check for IP-only addresses (except localhost which is already blocked)
    if base_domain.replace(".", "").isdigit():
        return False, "Cannot crawl IP addresses directly. Please provide a domain name."

    # Must have a valid TLD (at least one dot in domain)
    if "." not in base_domain:
        return False, f"Invalid domain '{base_domain}'. Please provide a valid website URL with a domain extension."

    return True, ""

# Initialize database
init_database()

# Initialize FastAPI
app = FastAPI(
    title="AISEO - GEO Website Optimizer API",
    description="""
## GEO Website Optimization API

Transform any website content into AI-citation-ready content.

### How it works:
1. Submit a URL → We crawl and analyze the content
2. Our AI detects the industry and retrieves relevant GEO guidelines
3. Each content section is optimized in parallel
4. Get back GEO-optimized markdown ready for publishing

### Key Features:
- 🚀 Parallel chunk processing for speed
- 📊 Before/after GEO scoring
- 🎯 Industry-specific optimization
- 📝 Markdown & HTML output
- 💾 SQLite storage for retrieval
    """,
    version="2.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connections for real-time updates
active_connections: dict = {}


# ============================================================================
# Request/Response Models
# ============================================================================

class OptimizeRequest(BaseModel):
    url: str = Field(description="URL to optimize")
    max_pages: int = Field(default=5, ge=1, le=10, description="Max pages to crawl (1=single page, up to 10 for deep crawl)")
    settings: Optional[dict] = Field(default=None, description="Optional settings")


class AuditRequest(BaseModel):
    url: str = Field(description="URL to audit")
    max_pages: int = Field(default=5, ge=1, le=10, description="Max pages to crawl for audit")


class JobResponse(BaseModel):
    job_id: str
    url: str
    status: str
    industry: Optional[str]
    industry_confidence: float
    total_chunks: int
    completed_chunks: int
    original_geo_score: float
    optimized_geo_score: float
    created_at: str
    completed_at: Optional[str]
    error_message: Optional[str]


class ChunkResponse(BaseModel):
    chunk_id: str
    chunk_order: int
    section_title: str
    status: str
    geo_score_before: float
    geo_score_after: float
    processing_time_ms: int


class ResultResponse(BaseModel):
    job_id: str
    final_markdown: str
    final_html: Optional[str]
    report: dict
    total_guidelines_applied: int
    created_at: str


# ============================================================================
# WebSocket for Real-time Updates
# ============================================================================

@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time job updates."""
    await websocket.accept()
    active_connections[job_id] = websocket

    try:
        while True:
            # Keep connection alive, receive any client messages
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        if job_id in active_connections:
            del active_connections[job_id]


async def broadcast_progress(job_id: str, status: str, progress: int, message: str):
    """Broadcast progress to connected WebSocket clients."""
    if job_id in active_connections:
        try:
            await active_connections[job_id].send_json({
                "job_id": job_id,
                "status": status,
                "progress": progress,
                "message": message,
                "timestamp": datetime.now().isoformat()
            })
        except Exception:
            pass


# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/api/v2/optimize", response_model=dict)
async def start_optimization(
    request: OptimizeRequest,
    background_tasks: BackgroundTasks
):
    """
    Start a new website optimization job.

    The job runs in the background. Use the job_id to:
    - Check status: GET /api/v2/jobs/{job_id}
    - Get results: GET /api/v2/results/{job_id}
    - Stream updates: WebSocket /ws/{job_id}
    """
    from database import create_job

    # Validate URL before processing
    is_valid, error_msg = validate_url(request.url)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)

    # Create job first to get the ID
    job_id = create_job(request.url, request.settings)

    # Simple print callback (no async issues)
    def progress_callback(jid: str, status: str, progress: int, message: str):
        logger.info(f"[{jid}] {status} ({progress}%): {message}")

    # Run optimization in background thread
    def run_optimization():
        try:
            optimizer = WebsiteOptimizer(max_workers=3, max_pages=request.max_pages)
            optimizer.optimize_url(
                url=request.url,
                settings=request.settings,
                progress_callback=progress_callback,
                job_id=job_id  # Pass existing job_id
            )
        except Exception as e:
            logger.error(f"Optimization failed for {job_id}: {e}")
            from database import update_job
            update_job(job_id, status="failed", error_message=str(e))

    background_tasks.add_task(run_optimization)

    return {
        "job_id": job_id,
        "status": "started",
        "message": "Optimization started. Poll status endpoint for updates.",
        "status_url": f"/api/v2/jobs/{job_id}",
        "result_url": f"/api/v2/results/{job_id}"
    }


@app.get("/api/v2/jobs", response_model=List[dict])
async def list_jobs(limit: int = 50):
    """List all optimization jobs."""
    jobs = get_all_jobs(limit)
    return jobs


@app.get("/api/v2/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get detailed status of a job."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    chunks = get_job_chunks(job_id)

    # Parse crawl_metadata JSON
    crawl_metadata = {}
    if job.get("crawl_metadata"):
        try:
            crawl_metadata = json.loads(job["crawl_metadata"])
        except (json.JSONDecodeError, TypeError):
            crawl_metadata = {}

    return {
        **job,
        "crawl_stats": crawl_metadata,
        "chunks": [
            {
                "chunk_id": c["chunk_id"],
                "section_title": c["section_title"],
                "status": c["status"],
                "geo_score_after": c["geo_score_after"],
                "processing_time_ms": c.get("processing_time_ms", 0),
            }
            for c in chunks
        ]
    }


@app.get("/api/v2/results/{job_id}")
async def get_job_result(job_id: str):
    """Get the final result of a completed job."""
    result = get_result(job_id)
    if not result:
        job = get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job["status"] != "completed":
            raise HTTPException(status_code=400, detail=f"Job not completed. Status: {job['status']}")
        raise HTTPException(status_code=404, detail="Result not found")

    return result


@app.get("/api/v2/results/{job_id}/markdown")
async def download_markdown(job_id: str):
    """Download the optimized markdown file."""
    result = get_result(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")

    # Return as downloadable file
    return StreamingResponse(
        iter([result["final_markdown"]]),
        media_type="text/markdown",
        headers={
            "Content-Disposition": f"attachment; filename=geo_optimized_{job_id}.md"
        }
    )


@app.get("/api/v2/results/{job_id}/html")
async def download_html(job_id: str):
    """Download the optimized HTML file."""
    result = get_result(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")

    return StreamingResponse(
        iter([result["final_html"] or ""]),
        media_type="text/html",
        headers={
            "Content-Disposition": f"attachment; filename=geo_optimized_{job_id}.html"
        }
    )


@app.get("/api/v2/results/{job_id}/guidelines")
async def get_applied_guidelines(job_id: str):
    """Get all guidelines that were applied for a job."""
    guidelines = get_job_guidelines_applied(job_id)
    return {
        "job_id": job_id,
        "total_applied": len(guidelines),
        "guidelines": guidelines
    }


@app.get("/api/v2/stats")
async def get_optimizer_stats():
    """Get overall optimizer statistics."""
    stats = get_stats()
    return stats


@app.get("/api/v2/guidelines")
async def get_all_guidelines(
    category: Optional[str] = None,
    industry: Optional[str] = None,
    view: Optional[str] = None,
    limit: int = 100
):
    """
    Get all GEO guidelines stored in Qdrant.

    Optional filters:
    - category: Filter by category (citation_optimization, universal_seo_geo, industry_specific, technical, metrics)
    - industry: Filter by industry (B2B_SaaS, Healthcare, E-commerce, etc.)
    - view: Set to "sources" to get unique research papers instead of guidelines
    - limit: Maximum number of results to return (default 100)
    """
    from src.clients.qdrant_client import get_qdrant_client, COLLECTIONS

    client = get_qdrant_client()
    if not client:
        raise HTTPException(status_code=503, detail="Qdrant not available")

    # Map category to collection name
    collection_map = {
        "citation_optimization": "geo_seo_citation",
        "citation": "geo_seo_citation",
        "universal_seo_geo": "geo_seo_universal",
        "universal": "geo_seo_universal",
        "industry_specific": "geo_seo_industry",
        "industry": "geo_seo_industry",
        "technical": "geo_seo_technical",
        "metrics": "geo_seo_metrics",
    }

    # Determine which collections to query
    if category:
        collection_name = collection_map.get(category.lower())
        if not collection_name:
            raise HTTPException(status_code=400, detail=f"Invalid category: {category}")
        collections_to_query = [collection_name]
    else:
        collections_to_query = list(set(collection_map.values()))

    all_guidelines = []

    for collection_name in collections_to_query:
        try:
            # Scroll through all points in the collection
            results, _ = client.scroll(
                collection_name=collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )

            for point in results:
                payload = point.payload
                # Apply industry filter if specified
                if industry:
                    industries = payload.get("industry_tags", [])
                    if industry not in industries and industries:
                        continue

                all_guidelines.append({
                    "id": payload.get("guideline_id", str(point.id)),
                    "content": payload.get("guideline_text", ""),
                    "category": payload.get("category", ""),
                    "industries": payload.get("industry_tags", []),
                    "source": payload.get("source_paper", {}),
                    "confidence": payload.get("confidence_score", 0),
                    "priority": payload.get("priority", "medium"),
                    "has_quantitative_data": payload.get("has_quantitative_data", False),
                    "has_case_study": payload.get("has_case_study", False),
                })
        except Exception as e:
            logger.warning(f"Error querying {collection_name}: {e}")
            continue

    # Get collection counts for summary
    collection_counts = {}
    for coll_name in set(collection_map.values()):
        try:
            info = client.get_collection(coll_name)
            collection_counts[coll_name] = info.points_count
        except Exception:
            collection_counts[coll_name] = 0

    # If view=sources, return unique research papers instead of guidelines
    if view == "sources":
        sources_map = {}
        for g in all_guidelines:
            source = g.get("source", {})
            url = source.get("url", "")
            if url and url not in sources_map:
                sources_map[url] = {
                    "title": source.get("title", "Unknown"),
                    "authors": source.get("authors", []),
                    "url": url,
                    "venue": source.get("venue", source.get("excerpt", "")),
                    "guidelines_count": 1
                }
            elif url:
                sources_map[url]["guidelines_count"] += 1

        sources = sorted(sources_map.values(), key=lambda x: x["guidelines_count"], reverse=True)
        return {
            "total": len(sources),
            "sources": sources[:limit]
        }

    return {
        "total": len(all_guidelines),
        "filters": {
            "category": category,
            "industry": industry,
        },
        "stats": {
            "total_in_db": sum(collection_counts.values()),
            "by_category": collection_counts,
        },
        "guidelines": all_guidelines[:limit]
    }


@app.get("/api/v2/showcase")
async def get_showcase(limit: int = 50):
    """
    Get completed optimization jobs for public showcase.

    Returns URLs that have been optimized with full details:
    - URL, industry, scores, improvement %, timestamp
    - Deduplicated by domain (only shows most recent entry per domain)
    """
    from database import get_db

    with get_db() as conn:
        cursor = conn.cursor()
        # Fetch more than limit to account for deduplication
        cursor.execute("""
            SELECT
                job_id,
                url,
                industry,
                industry_confidence,
                original_geo_score,
                optimized_geo_score,
                CASE
                    WHEN original_geo_score > 0
                    THEN ROUND(((optimized_geo_score - original_geo_score) / original_geo_score) * 100, 1)
                    ELSE 0
                END as improvement_pct,
                total_chunks,
                created_at,
                completed_at
            FROM optimization_jobs
            WHERE status = 'completed'
            ORDER BY completed_at DESC
            LIMIT ?
        """, (limit * 3,))  # Fetch extra to have enough after deduplication

        # Deduplicate by domain - keep only the most recent entry per domain
        seen_domains = set()
        jobs = []

        for row in cursor.fetchall():
            domain = extract_domain(row["url"])

            # Skip if we've already seen this domain
            if domain in seen_domains:
                continue

            seen_domains.add(domain)
            jobs.append({
                "job_id": row["job_id"],
                "url": row["url"],
                "industry": row["industry"],
                "industry_confidence": row["industry_confidence"],
                "original_score": row["original_geo_score"],
                "optimized_score": row["optimized_geo_score"],
                "improvement_pct": row["improvement_pct"] or 0,
                "chunks_processed": row["total_chunks"],
                "optimized_at": row["completed_at"] or row["created_at"]
            })

            # Stop once we have enough unique domains
            if len(jobs) >= limit:
                break

        return {
            "total": len(jobs),
            "websites": jobs
        }


@app.get("/api/v2/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "GEO Website Optimizer API",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/v2/audit")
async def audit_website(request: AuditRequest):
    """
    Audit a website for GEO optimization issues.

    Returns a detailed report showing:
    - Overall GEO score
    - Per-page breakdown with URL identification
    - Issues found (missing citations, statistics, structure, FAQ, etc.)
    - Actionable recommendations

    This helps users understand WHY their content needs optimization.
    """
    import time
    from dataclasses import asdict
    from src.workflows.nodes.crawler_node import crawl_and_chunk
    from src.workflows.nodes.audit_analyzer import generate_audit_report

    # Validate URL before processing
    is_valid, error_msg = validate_url(request.url)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)

    start_time = time.time()

    try:
        # Crawl the website (respecting max_pages limit)
        chunks, metadata = await crawl_and_chunk(request.url, max_pages=request.max_pages)

        if not chunks:
            raise HTTPException(status_code=400, detail="No content extracted from URL")

        # Group chunks by source URL for page-level analysis
        pages_content = {}
        for chunk in chunks:
            source_url = chunk.get("source_url", request.url)
            if source_url not in pages_content:
                pages_content[source_url] = {
                    "url": source_url,
                    "title": chunk.get("page_title", chunk.get("title", "Untitled")),
                    "content": ""
                }
            pages_content[source_url]["content"] += "\n\n" + chunk.get("content", "")

        # Generate audit report
        report = generate_audit_report(list(pages_content.values()))

        elapsed_ms = int((time.time() - start_time) * 1000)

        return {
            "url": request.url,
            "pages_audited": report.total_pages,
            "max_pages": request.max_pages,
            "overall_score": report.overall_score,
            "summary": report.summary,
            "top_issues": report.top_issues,
            "top_recommendations": report.top_recommendations,
            "pages": report.pages,
            "crawl_metadata": {
                "title": metadata.get("title", "Unknown"),
                "total_words": metadata.get("word_count", 0),
                "pages_crawled": metadata.get("pages_crawled", 1),
                "crawl_time_ms": metadata.get("crawl_time_ms", 0),
            },
            "audit_time_ms": elapsed_ms,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audit failed for {request.url}: {e}")
        raise HTTPException(status_code=500, detail=f"Audit failed: {str(e)}")


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
