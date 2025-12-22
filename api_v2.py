"""
API v2: Website Optimization Endpoints.

New endpoints for the GEO Website Optimizer (Phase 2).
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Optional, List
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
    settings: Optional[dict] = Field(default=None, description="Optional settings")


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

    # Create job first to get the ID
    job_id = create_job(request.url, request.settings)

    # Simple print callback (no async issues)
    def progress_callback(jid: str, status: str, progress: int, message: str):
        logger.info(f"[{jid}] {status} ({progress}%): {message}")

    # Run optimization in background thread
    def run_optimization():
        try:
            optimizer = WebsiteOptimizer(max_workers=3)
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


@app.get("/api/v2/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "GEO Website Optimizer API",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
