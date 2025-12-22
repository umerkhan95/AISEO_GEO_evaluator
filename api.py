"""
FastAPI server for GEO/SEO Knowledge Base system with Deep Agents integration.

Provides REST endpoints for:
- PDF upload and processing
- Web search and paper discovery (via Deep Agents)
- Guideline search and retrieval
- Agent-based chat queries
- Batch status monitoring
- Qdrant collection management
- Knowledge base statistics and export
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, AsyncGenerator
from datetime import datetime
import os
import uuid
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables BEFORE importing config
load_dotenv()

# Use centralized clients
from src.clients import (
    get_qdrant_client,
    get_embeddings,
    get_llm,
    LLMPreset,
    COLLECTIONS,
    ensure_collections_exist,
)

# LangGraph imports
from graph import app as langgraph_app
from state import PDFDocument
from config import config

# Deep Agents imports (optional)
DEEP_AGENTS_AVAILABLE = False
try:
    from src.agents.orchestrator import (
        create_geo_seo_agent,
        create_research_only_agent,
        create_query_only_agent,
        query_knowledge_base,
        get_knowledge_base_stats,
    )
    DEEP_AGENTS_AVAILABLE = True
except ImportError:
    create_geo_seo_agent = None
    create_research_only_agent = None
    create_query_only_agent = None
    query_knowledge_base = None
    get_knowledge_base_stats = None

# Tools imports
from src.tools.web_search import search_geo_seo_papers, search_scholarly_articles

try:
    from src.tools.pdf_tools import download_pdf, analyze_pdf_with_gemini
except ImportError:
    download_pdf = None
    analyze_pdf_with_gemini = None

try:
    from src.tools.qdrant_tools import store_guidelines, search_guidelines as qdrant_search_guidelines, get_collection_stats
except ImportError:
    store_guidelines = None
    qdrant_search_guidelines = None
    get_collection_stats = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
api = FastAPI(
    title="GEO/SEO Knowledge Base API",
    description="""
    ## GEO/SEO Knowledge Base with Deep Agents

    A comprehensive API for building and querying a knowledge base of
    Generative Engine Optimization (GEO) and Search Engine Optimization (SEO) guidelines.

    ### Features
    - **Web Search**: Discover scholarly papers using Crawl4AI
    - **PDF Processing**: Download and analyze PDFs with OpenAI GPT-4o
    - **Deep Agent Chat**: Interactive agent for research and queries
    - **Vector Search**: Semantic search across 5 specialized collections
    - **Statistics**: Real-time knowledge base analytics

    ### Collections
    1. Universal SEO+GEO Framework
    2. Industry-Specific Strategies
    3. Technical Implementation
    4. Citation Optimization Tactics
    5. Measurement Metrics
    """,
    version="2.0.0",
)

# Add CORS middleware
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage for uploaded PDFs
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# In-memory batch tracking (use Redis/DB in production)
batch_status_tracker = {}


# ============================================================================
# Request/Response Models
# ============================================================================


class PDFUploadResponse(BaseModel):
    """Response after PDF upload."""

    batch_id: str
    filename: str
    file_size_mb: float
    status: str
    message: str


class BatchStatusResponse(BaseModel):
    """Batch processing status."""

    batch_id: str
    status: str
    documents_processed: int
    guidelines_extracted: int
    guidelines_stored: int
    errors: int
    started_at: datetime
    completed_at: Optional[datetime]


class GuidelineSearchRequest(BaseModel):
    """Request for searching guidelines."""

    query: str = Field(description="Search query")
    category: Optional[str] = Field(
        None,
        description="Filter by category: universal_seo_geo, industry_specific, technical, citation_optimization, metrics",
    )
    industries: Optional[List[str]] = Field(None, description="Filter by industries")
    priority: Optional[str] = Field(
        None, description="Filter by priority: critical, high, medium, low"
    )
    complexity: Optional[str] = Field(
        None, description="Filter by complexity: easy, moderate, complex"
    )
    limit: int = Field(10, description="Number of results to return", ge=1, le=100)


class GuidelineResponse(BaseModel):
    """Single guideline result."""

    guideline_id: str
    content: str
    category: str
    source_section: str
    page_numbers: List[int]
    industries: List[str]
    confidence_score: float
    priority: str
    implementation_complexity: str
    related_guideline_ids: List[str]
    similarity_score: float


class SearchResponse(BaseModel):
    """Search results."""

    query: str
    results: List[GuidelineResponse]
    total_found: int


# ============================================================================
# Helper Functions
# ============================================================================


def initialize_qdrant_client():
    """Initialize Qdrant client using centralized factory."""
    return get_qdrant_client()


def initialize_embeddings():
    """Initialize OpenAI embeddings using centralized factory."""
    return get_embeddings()


async def process_pdf_batch(batch_id: str, documents: List[PDFDocument]):
    """
    Background task to process PDF batch through LangGraph workflow.
    """
    logger.info(f"Starting background processing for batch: {batch_id}")

    # Update status
    batch_status_tracker[batch_id]["status"] = "processing"

    try:
        # Prepare initial state
        initial_state = {
            "documents": documents,
            "analyzed_documents": [],
            "raw_guidelines": [],
            "deduplicated_guidelines": [],
            "routed_guidelines": {},
            "enriched_guidelines": [],
            "storage_results": [],
            "errors": [],
            "batch_id": batch_id,
            "processing_start_time": datetime.now(),
            "current_node": "start",
            "retry_count": 0,
            "documents_processed_count": 0,
            "total_guidelines_extracted": 0,
            "memory_threshold_exceeded": False,
        }

        # Run LangGraph workflow
        final_state = langgraph_app.invoke(initial_state)

        # Update batch status with results
        batch_status_tracker[batch_id].update(
            {
                "status": "completed",
                "documents_processed": final_state.get("documents_processed_count", 0),
                "guidelines_extracted": final_state.get("total_guidelines_extracted", 0),
                "guidelines_stored": len(final_state.get("storage_results", [])),
                "errors": len(final_state.get("errors", [])),
                "completed_at": datetime.now(),
            }
        )

        logger.info(f"Batch {batch_id} processing complete")

    except Exception as e:
        logger.error(f"Error processing batch {batch_id}: {str(e)}")
        batch_status_tracker[batch_id].update(
            {
                "status": "failed",
                "error_message": str(e),
                "completed_at": datetime.now(),
            }
        )


# ============================================================================
# API Endpoints
# ============================================================================


@api.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "GEO/SEO Knowledge Base API",
        "version": "1.0.0",
    }


@api.post("/upload", response_model=PDFUploadResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload PDF for processing.

    The PDF will be processed through the LangGraph workflow:
    1. Document Analysis
    2. Content Extraction & Classification
    3. Deduplication & Merging
    4. Collection Routing
    5. Metadata Enrichment
    6. Vector Storage

    Returns a batch_id for status tracking.
    """
    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Generate batch ID
    batch_id = str(uuid.uuid4())

    # Save uploaded file
    file_path = UPLOAD_DIR / f"{batch_id}_{file.filename}"

    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        file_size_mb = len(content) / (1024 * 1024)

        # Check file size
        if file_size_mb > config.processing.max_pdf_size_mb:
            os.remove(file_path)
            raise HTTPException(
                status_code=400,
                detail=f"PDF exceeds maximum size of {config.processing.max_pdf_size_mb}MB",
            )

        # Create PDFDocument
        pdf_doc = PDFDocument(
            file_path=str(file_path),
            filename=file.filename,
            upload_timestamp=datetime.now(),
            file_size_mb=file_size_mb,
        )

        # Initialize batch status
        batch_status_tracker[batch_id] = {
            "batch_id": batch_id,
            "status": "queued",
            "documents_processed": 0,
            "guidelines_extracted": 0,
            "guidelines_stored": 0,
            "errors": 0,
            "started_at": datetime.now(),
            "completed_at": None,
        }

        # Start background processing
        background_tasks.add_task(process_pdf_batch, batch_id, [pdf_doc])

        logger.info(f"Queued PDF for processing: {file.filename} (batch: {batch_id})")

        return PDFUploadResponse(
            batch_id=batch_id,
            filename=file.filename,
            file_size_mb=file_size_mb,
            status="queued",
            message="PDF uploaded successfully. Processing started in background.",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading PDF: {str(e)}")
        if file_path.exists():
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@api.post("/upload-batch", response_model=PDFUploadResponse)
async def upload_pdf_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
):
    """
    Upload multiple PDFs for batch processing.

    All PDFs will be processed together using the parallel workflow.
    """
    if len(files) > 10:
        raise HTTPException(
            status_code=400, detail="Maximum 10 PDFs per batch upload"
        )

    batch_id = str(uuid.uuid4())
    documents = []

    try:
        for file in files:
            if not file.filename.endswith(".pdf"):
                raise HTTPException(
                    status_code=400, detail=f"File {file.filename} is not a PDF"
                )

            file_path = UPLOAD_DIR / f"{batch_id}_{file.filename}"
            content = await file.read()

            with open(file_path, "wb") as f:
                f.write(content)

            file_size_mb = len(content) / (1024 * 1024)

            pdf_doc = PDFDocument(
                file_path=str(file_path),
                filename=file.filename,
                upload_timestamp=datetime.now(),
                file_size_mb=file_size_mb,
            )

            documents.append(pdf_doc)

        # Initialize batch status
        batch_status_tracker[batch_id] = {
            "batch_id": batch_id,
            "status": "queued",
            "documents_processed": 0,
            "guidelines_extracted": 0,
            "guidelines_stored": 0,
            "errors": 0,
            "started_at": datetime.now(),
            "completed_at": None,
        }

        # Start background processing
        background_tasks.add_task(process_pdf_batch, batch_id, documents)

        logger.info(f"Queued {len(files)} PDFs for batch processing (batch: {batch_id})")

        return PDFUploadResponse(
            batch_id=batch_id,
            filename=f"{len(files)} PDFs",
            file_size_mb=sum(d.file_size_mb for d in documents),
            status="queued",
            message=f"Batch of {len(files)} PDFs uploaded successfully.",
        )

    except Exception as e:
        logger.error(f"Error in batch upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch upload failed: {str(e)}")


@api.get("/batch/{batch_id}", response_model=BatchStatusResponse)
async def get_batch_status(batch_id: str):
    """
    Get processing status for a batch.
    """
    if batch_id not in batch_status_tracker:
        raise HTTPException(status_code=404, detail="Batch not found")

    status = batch_status_tracker[batch_id]
    return BatchStatusResponse(**status)


@api.post("/search", response_model=SearchResponse)
async def search_guidelines_endpoint(request: GuidelineSearchRequest):
    """
    Search for guidelines using semantic search.

    Searches across all Qdrant collections or filtered by category.
    """
    try:
        # Initialize clients
        qdrant_client = initialize_qdrant_client()
        embeddings_model = initialize_embeddings()

        # Generate query embedding
        query_embedding = embeddings_model.embed_query(request.query)

        # Determine which collections to search
        if request.category:
            if request.category not in config.qdrant.collections:
                raise HTTPException(status_code=400, detail="Invalid category")
            collections_to_search = [config.qdrant.collections[request.category]]
        else:
            collections_to_search = config.qdrant.collection_names

        # Search across collections
        all_results = []

        for collection_name in collections_to_search:
            # Build filter
            must_conditions = []

            if request.industries:
                must_conditions.append(
                    {"key": "industries", "match": {"any": request.industries}}
                )

            if request.priority:
                must_conditions.append(
                    {"key": "priority", "match": {"value": request.priority}}
                )

            if request.complexity:
                must_conditions.append(
                    {
                        "key": "implementation_complexity",
                        "match": {"value": request.complexity},
                    }
                )

            query_filter = {"must": must_conditions} if must_conditions else None

            # Search using query_points (newer Qdrant API)
            from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue

            # Build proper filter
            filter_obj = None
            if must_conditions:
                conditions = []
                for cond in must_conditions:
                    if "any" in cond.get("match", {}):
                        conditions.append(FieldCondition(
                            key=cond["key"],
                            match=MatchAny(any=cond["match"]["any"])
                        ))
                    else:
                        conditions.append(FieldCondition(
                            key=cond["key"],
                            match=MatchValue(value=cond["match"]["value"])
                        ))
                filter_obj = Filter(must=conditions)

            results = qdrant_client.query_points(
                collection_name=collection_name,
                query=query_embedding,
                query_filter=filter_obj,
                limit=request.limit,
            )

            for result in results.points:
                all_results.append(
                    GuidelineResponse(
                        guideline_id=result.payload["guideline_id"],
                        content=result.payload["content"],
                        category=result.payload["category"],
                        source_section=result.payload["source_section"],
                        page_numbers=result.payload["page_numbers"],
                        industries=result.payload["industries"],
                        confidence_score=result.payload["confidence_score"],
                        priority=result.payload["priority"],
                        implementation_complexity=result.payload[
                            "implementation_complexity"
                        ],
                        related_guideline_ids=result.payload["related_guideline_ids"],
                        similarity_score=result.score,
                    )
                )

        # Sort by similarity score
        all_results.sort(key=lambda x: x.similarity_score, reverse=True)

        # Limit results
        all_results = all_results[: request.limit]

        return SearchResponse(
            query=request.query, results=all_results, total_found=len(all_results)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@api.get("/collections")
async def list_collections():
    """
    List all Qdrant collections and their statistics.
    """
    try:
        qdrant_client = initialize_qdrant_client()

        collections_info = []
        for category, collection_name in config.qdrant.collections.items():
            if qdrant_client.collection_exists(collection_name):
                info = qdrant_client.get_collection(collection_name)
                # Handle different Qdrant API versions
                vectors_count = getattr(info, 'vectors_count', None)
                if vectors_count is None:
                    # Try to get from points_count for newer Qdrant versions
                    vectors_count = getattr(info, 'points_count', 0)
                collections_info.append(
                    {
                        "category": category,
                        "collection_name": collection_name,
                        "vectors_count": vectors_count,
                        "points_count": info.points_count,
                    }
                )
            else:
                collections_info.append(
                    {
                        "category": category,
                        "collection_name": collection_name,
                        "status": "not_created",
                    }
                )

        return {"collections": collections_info}

    except Exception as e:
        logger.error(f"Error listing collections: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list collections: {str(e)}"
        )


@api.get("/guideline/{guideline_id}")
async def get_guideline(guideline_id: str):
    """
    Get a specific guideline by ID.
    """
    try:
        qdrant_client = initialize_qdrant_client()

        # Search across all collections
        for collection_name in config.qdrant.collection_names:
            if not qdrant_client.collection_exists(collection_name):
                continue

            # Search by guideline_id in payload
            results = qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter={
                    "must": [{"key": "guideline_id", "match": {"value": guideline_id}}]
                },
                limit=1,
            )

            if results[0]:  # Found
                payload = results[0][0].payload
                return GuidelineResponse(
                    guideline_id=payload["guideline_id"],
                    content=payload["content"],
                    category=payload["category"],
                    source_section=payload["source_section"],
                    page_numbers=payload["page_numbers"],
                    industries=payload["industries"],
                    confidence_score=payload["confidence_score"],
                    priority=payload["priority"],
                    implementation_complexity=payload["implementation_complexity"],
                    related_guideline_ids=payload["related_guideline_ids"],
                    similarity_score=1.0,  # Exact match
                )

        raise HTTPException(status_code=404, detail="Guideline not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching guideline: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch guideline: {str(e)}")


# ============================================================================
# Deep Agents Endpoints
# ============================================================================


class ResearchRequest(BaseModel):
    """Request for paper research."""
    topic: str = Field(description="Research topic (e.g., 'citation optimization for healthcare')")
    industry: Optional[str] = Field(None, description="Industry filter")
    max_results: int = Field(10, description="Maximum results", ge=1, le=50)


class AgentChatRequest(BaseModel):
    """Request for agent chat."""
    message: str = Field(description="User message to the agent")
    session_id: Optional[str] = Field(None, description="Optional session ID for context")


class ProcessURLRequest(BaseModel):
    """Request to process a paper from URL."""
    url: str = Field(description="URL of the PDF to process")
    filename: Optional[str] = Field(None, description="Custom filename for the PDF")


# Global agent instances (lazy initialization)
_main_agent = None
_research_agent = None
_query_agent = None


def get_main_agent():
    """Get or create main Deep Agent."""
    global _main_agent
    if _main_agent is None:
        if DEEP_AGENTS_AVAILABLE and create_geo_seo_agent:
            _main_agent = create_geo_seo_agent()
        else:
            # Use centralized LLM factory as fallback
            _main_agent = get_llm(LLMPreset.DEFAULT)
    return _main_agent


@api.post("/api/v1/research", tags=["Deep Agents"])
async def research_papers(request: ResearchRequest):
    """
    Search for GEO/SEO scholarly papers using Tavily.

    This endpoint uses web search to discover relevant papers.
    """
    try:
        if request.industry:
            result = search_scholarly_articles(
                topic=request.topic,
                industry=request.industry,
                max_results=request.max_results,
            )
        else:
            result = search_geo_seo_papers(
                query=request.topic,
                max_results=request.max_results,
            )

        return json.loads(result)

    except Exception as e:
        logger.error(f"Research error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Research failed: {str(e)}")


@api.post("/api/v1/process-url", tags=["Deep Agents"])
async def process_paper_from_url(
    background_tasks: BackgroundTasks,
    request: ProcessURLRequest,
):
    """
    Download and process a paper from URL.

    The paper will be downloaded, analyzed with Gemini, and guidelines stored in Qdrant.
    """
    batch_id = str(uuid.uuid4())

    try:
        # Download PDF
        download_result = download_pdf(
            url=request.url,
            filename=request.filename,
        )
        download_data = json.loads(download_result)

        if download_data.get("status") != "success":
            raise HTTPException(
                status_code=400,
                detail=f"Failed to download PDF: {download_data.get('error', 'Unknown error')}"
            )

        filepath = download_data["filepath"]

        # Initialize batch tracking
        batch_status_tracker[batch_id] = {
            "batch_id": batch_id,
            "status": "processing",
            "url": request.url,
            "filepath": filepath,
            "documents_processed": 0,
            "guidelines_extracted": 0,
            "guidelines_stored": 0,
            "errors": 0,
            "started_at": datetime.now(),
            "completed_at": None,
        }

        # Process in background
        async def process_pdf():
            try:
                # Analyze with Gemini
                analysis_result = analyze_pdf_with_gemini(filepath, "full")
                analysis_data = json.loads(analysis_result)

                if analysis_data.get("status") not in ["success", "partial"]:
                    batch_status_tracker[batch_id]["status"] = "failed"
                    batch_status_tracker[batch_id]["error"] = analysis_data.get("error")
                    return

                batch_status_tracker[batch_id]["documents_processed"] = 1

                # Extract and store guidelines
                analysis = analysis_data.get("analysis", {})
                guidelines = analysis.get("guidelines", [])

                if guidelines:
                    source_paper = {
                        "title": analysis.get("structure", {}).get("title", "Unknown"),
                        "authors": analysis.get("structure", {}).get("authors", []),
                        "url": request.url,
                        "filepath": filepath,
                    }

                    store_result = store_guidelines(guidelines, source_paper)
                    store_data = json.loads(store_result)

                    batch_status_tracker[batch_id]["guidelines_extracted"] = len(guidelines)
                    batch_status_tracker[batch_id]["guidelines_stored"] = store_data.get("stored", 0)
                    batch_status_tracker[batch_id]["errors"] = store_data.get("failed", 0)

                batch_status_tracker[batch_id]["status"] = "completed"
                batch_status_tracker[batch_id]["completed_at"] = datetime.now()

            except Exception as e:
                batch_status_tracker[batch_id]["status"] = "failed"
                batch_status_tracker[batch_id]["error"] = str(e)

        background_tasks.add_task(process_pdf)

        return {
            "batch_id": batch_id,
            "status": "processing",
            "url": request.url,
            "message": "PDF download started. Processing in background.",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Process URL error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@api.post("/api/v1/chat", tags=["Deep Agents"])
async def chat_with_agent(request: AgentChatRequest):
    """
    Chat with the GEO/SEO Deep Agent.

    The agent can:
    - Search for papers
    - Analyze documents
    - Query the knowledge base
    - Provide recommendations

    Example messages:
    - "Find papers about citation optimization for B2B SaaS"
    - "What are the best practices for healthcare GEO?"
    - "Show me technical implementation guidelines"
    """
    try:
        agent = get_main_agent()

        # Prepare messages
        messages = [
            {"role": "user", "content": request.message}
        ]

        # Invoke agent
        result = agent.invoke(messages)

        # Extract response
        response_messages = result.get("messages", [])
        last_message = response_messages[-1] if response_messages else None

        if last_message:
            # Handle different message types
            if hasattr(last_message, "content"):
                content = last_message.content
            elif isinstance(last_message, dict):
                content = last_message.get("content", str(last_message))
            else:
                content = str(last_message)

            return {
                "response": content,
                "session_id": request.session_id,
            }
        else:
            return {
                "response": "No response generated",
                "session_id": request.session_id,
            }

    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@api.get("/api/v1/query", tags=["Deep Agents"])
async def query_guidelines(
    query: str,
    collection: Optional[str] = None,
    industry: Optional[str] = None,
    priority: Optional[str] = None,
    limit: int = 10,
):
    """
    Query the knowledge base for guidelines.

    This is a direct semantic search without agent processing.

    Args:
        query: Search query
        collection: Filter by collection (universal, industry, technical, citation, metrics)
        industry: Filter by industry (B2B_SaaS, Healthcare, Ecommerce, etc.)
        priority: Filter by priority (high, medium, low)
        limit: Maximum results
    """
    try:
        result = search_guidelines(
            query=query,
            collection=collection,
            industry=industry,
            priority=priority,
            limit=limit,
        )
        return json.loads(result)

    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@api.get("/api/v1/stats", tags=["Deep Agents"])
async def get_stats():
    """
    Get knowledge base statistics.

    Returns counts per collection, industry distribution, and average confidence scores.
    """
    try:
        result = get_collection_stats()
        return json.loads(result)

    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Stats failed: {str(e)}")


@api.get("/api/v1/export", tags=["Deep Agents"])
async def export_guidelines(
    collection: Optional[str] = None,
    format: str = "json",
):
    """
    Export all guidelines as JSON.

    Args:
        collection: Optional collection filter
        format: Export format (currently only 'json' supported)
    """
    try:
        # Get stats first to see what we have
        stats_result = get_collection_stats()
        stats = json.loads(stats_result)

        # Collect all guidelines
        all_guidelines = []

        # Use centralized client and COLLECTIONS
        client = get_qdrant_client()

        collections_to_export = (
            [COLLECTIONS.get(collection, collection)] if collection
            else list(COLLECTIONS.values())
        )

        for coll_name in collections_to_export:
            try:
                # Scroll through all points
                offset = None
                while True:
                    results, offset = client.scroll(
                        collection_name=coll_name,
                        limit=100,
                        offset=offset,
                        with_payload=True,
                    )

                    for point in results:
                        guideline = {
                            "collection": coll_name,
                            **point.payload,
                        }
                        all_guidelines.append(guideline)

                    if offset is None:
                        break
            except Exception:
                continue

        return {
            "total_guidelines": len(all_guidelines),
            "collections_exported": collections_to_export,
            "export_timestamp": datetime.now().isoformat(),
            "guidelines": all_guidelines,
        }

    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@api.get("/health", tags=["System"])
async def health_check():
    """Detailed health check."""
    health = {
        "status": "healthy",
        "service": "GEO/SEO Knowledge Base API",
        "version": "2.0.0",
        "components": {
            "api": "healthy",
            "qdrant": "unknown",
            "deep_agents": "unknown",
        },
        "timestamp": datetime.now().isoformat(),
    }

    # Check Qdrant using centralized client
    try:
        client = get_qdrant_client()
        collections = client.get_collections()
        health["components"]["qdrant"] = "healthy"
        health["qdrant_collections"] = len(collections.collections)
    except Exception as e:
        health["components"]["qdrant"] = f"unhealthy: {str(e)}"

    # Check Deep Agents availability
    try:
        from deepagents import create_deep_agent
        health["components"]["deep_agents"] = "available"
    except ImportError:
        health["components"]["deep_agents"] = "not installed"

    # Overall status
    if any("unhealthy" in str(v) for v in health["components"].values()):
        health["status"] = "degraded"

    return health


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))

    uvicorn.run(api, host=host, port=port, log_level="info")
