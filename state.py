"""
State schema for GEO/SEO Knowledge Base LangGraph workflow.

This module defines the state structure using TypedDict with Annotated reducers
for proper state accumulation across nodes.
"""

from typing import TypedDict, Annotated, Literal, Optional
from operator import add
from dataclasses import dataclass
from datetime import datetime


@dataclass
class PDFDocument:
    """Represents a PDF document to be processed."""
    file_path: str
    filename: str
    upload_timestamp: datetime
    file_size_mb: float


@dataclass
class DocumentStructure:
    """Extracted structure from PDF analysis."""
    title: str
    authors: list[str]
    abstract: str
    sections: list[dict[str, str]]  # {title: content}
    themes: list[str]
    industries: list[str]
    total_pages: int
    extraction_confidence: float


@dataclass
class Guideline:
    """Represents a single extracted guideline."""
    guideline_id: str
    content: str
    source_section: str
    page_numbers: list[int]
    category: Literal[
        "universal_seo_geo",
        "industry_specific",
        "technical",
        "citation_optimization",
        "metrics"
    ]
    industries: list[str]  # Empty if universal
    extraction_confidence: float


@dataclass
class EnrichedGuideline:
    """Guideline with enriched metadata."""
    guideline: Guideline
    confidence_score: float
    priority: Literal["critical", "high", "medium", "low"]
    implementation_complexity: Literal["easy", "moderate", "complex"]
    related_guideline_ids: list[str]
    semantic_cluster_id: Optional[str]


@dataclass
class ProcessingError:
    """Represents an error that occurred during processing."""
    node_name: str
    error_type: str
    error_message: str
    timestamp: datetime
    recoverable: bool


class GraphState(TypedDict):
    """
    Main state for the GEO/SEO Knowledge Base workflow.

    Uses Annotated with add operator for list accumulation to prevent overwrites.
    """

    # Input: PDF documents to process
    documents: Annotated[list[PDFDocument], add]

    # Node 1: Document Analyzer output
    analyzed_documents: Annotated[list[dict], add]  # {doc: PDFDocument, structure: DocumentStructure}

    # Node 2: Content Extractor output
    raw_guidelines: Annotated[list[Guideline], add]

    # Node 3: Deduplication output
    deduplicated_guidelines: Annotated[list[Guideline], add]
    merge_log: Annotated[list[dict], add]  # Track which guidelines were merged

    # Node 4: Collection Router output
    routed_guidelines: dict[str, list[Guideline]]  # category -> guidelines

    # Node 5: Metadata Enricher output
    enriched_guidelines: Annotated[list[EnrichedGuideline], add]

    # Node 6: Vector Storage output
    storage_results: Annotated[list[dict], add]  # {guideline_id, collection, vector_id, status}

    # Error handling
    errors: Annotated[list[ProcessingError], add]

    # Processing metadata
    batch_id: str
    processing_start_time: datetime
    current_node: str
    retry_count: int

    # Memory management flags
    documents_processed_count: int
    total_guidelines_extracted: int
    memory_threshold_exceeded: bool


class DocumentBatchState(TypedDict):
    """
    State for parallel document processing using Send() API.

    Each document gets its own sub-state for parallel processing.
    """
    document: PDFDocument
    structure: Optional[DocumentStructure]
    guidelines: Annotated[list[Guideline], add]
    error: Optional[ProcessingError]
    status: Literal["pending", "processing", "completed", "failed"]
