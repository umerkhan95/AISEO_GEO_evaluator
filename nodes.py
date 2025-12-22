"""
LangGraph nodes for GEO/SEO Knowledge Base workflow.

Each node implements a specific processing step with proper error handling,
logging, and memory management.
"""

import hashlib
import re
from datetime import datetime
from typing import Any
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from qdrant_client.models import Distance, VectorParams, PointStruct

# Use centralized clients
from src.clients import (
    get_qdrant_client,
    get_embeddings,
    get_extractor_llm,
    ensure_collections_exist,
    calculate_priority as centralized_calculate_priority,
    calculate_complexity as centralized_calculate_complexity,
)

from state import (
    GraphState,
    PDFDocument,
    DocumentStructure,
    Guideline,
    EnrichedGuideline,
    ProcessingError,
)
from config import config, CATEGORY_DESCRIPTIONS, PRIORITY_RULES, COMPLEXITY_INDICATORS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models for Structured Output
# ============================================================================


class Section(BaseModel):
    """A section from the document."""
    title: str = Field(description="Section title")
    content: str = Field(description="Section content")


class ExtractedStructure(BaseModel):
    """Structured output from document analysis."""

    title: str = Field(description="Paper title")
    authors: list[str] = Field(default_factory=list, description="List of author names")
    abstract: str = Field(description="Paper abstract")
    sections: list[Section] = Field(
        default_factory=list,
        description="List of sections with title and content"
    )
    themes: list[str] = Field(
        default_factory=list,
        description="Main themes (e.g., 'content optimization', 'technical SEO')"
    )
    industries: list[str] = Field(
        default_factory=list,
        description="Industries mentioned (empty list if universal)"
    )
    total_pages: int = Field(description="Number of pages")
    extraction_confidence: float = Field(
        default=0.8,
        description="Confidence score 0-1", ge=0.0, le=1.0
    )


class ExtractedGuideline(BaseModel):
    """Single guideline extracted from document."""

    content: str = Field(description="The guideline text")
    source_section: str = Field(default="", description="Section it came from")
    page_numbers: list[int] = Field(default_factory=list, description="Page numbers where found")
    category: str = Field(
        default="universal_seo_geo",
        description="One of: universal_seo_geo, industry_specific, technical, citation_optimization, metrics"
    )
    industries: list[str] = Field(
        default_factory=list,
        description="Relevant industries (empty if universal)"
    )
    extraction_confidence: float = Field(default=0.8, description="Confidence 0-1", ge=0.0, le=1.0)


class GuidelinesList(BaseModel):
    """List of extracted guidelines."""

    guidelines: list[ExtractedGuideline] = Field(
        default_factory=list,
        description="All extracted guidelines"
    )


class EnrichedMetadata(BaseModel):
    """Enriched metadata for a guideline."""

    confidence_score: float = Field(default=0.8, description="Overall confidence", ge=0.0, le=1.0)
    priority: str = Field(default="medium", description="critical, high, medium, or low")
    implementation_complexity: str = Field(default="moderate", description="easy, moderate, or complex")
    related_guideline_ids: list[str] = Field(
        default_factory=list,
        description="IDs of related guidelines"
    )


# ============================================================================
# NODE 1: Document Analyzer
# ============================================================================


def document_analyzer_node(state: GraphState) -> dict[str, Any]:
    """
    Analyze PDF documents using Gemini Flash 2.0 for structure extraction.

    Uses multimodal capabilities to read PDFs directly and extract:
    - Title, authors, abstract
    - Section structure
    - Themes and industry mentions
    - Metadata
    """
    logger.info(f"=== Node 1: Document Analyzer ===")
    logger.info(f"Documents to process: {len(state.get('documents', []))}")

    analyzed_documents = []
    errors = []

    # Use centralized extractor LLM (GPT-4o for document analysis)
    llm = get_extractor_llm()

    # Use structured output
    structured_llm = llm.with_structured_output(ExtractedStructure)

    for doc in state.get("documents", []):
        try:
            logger.info(f"Analyzing document: {doc.filename}")

            # Check file size limits
            if doc.file_size_mb > config.processing.max_pdf_size_mb:
                raise ValueError(
                    f"PDF exceeds size limit: {doc.file_size_mb}MB > {config.processing.max_pdf_size_mb}MB"
                )

            # Extract text from PDF using PyMuPDF for better OpenAI compatibility
            import fitz  # PyMuPDF
            pdf_doc = fitz.open(doc.file_path)
            pdf_text = ""
            total_pages = len(pdf_doc)

            for page_num, page in enumerate(pdf_doc):
                pdf_text += f"\n--- Page {page_num + 1} ---\n"
                pdf_text += page.get_text()

            pdf_doc.close()

            # Truncate if too long (GPT-4o context limit)
            max_chars = 100000
            if len(pdf_text) > max_chars:
                pdf_text = pdf_text[:max_chars] + "\n\n[... content truncated due to length ...]"

            # Create analysis prompt
            system_prompt = """You are an expert at analyzing academic papers about SEO and GEO.
Extract the document structure, identifying:
1. Title, authors, and abstract
2. All major sections with their content
3. Main themes discussed
4. Any specific industries mentioned (healthcare, finance, etc.)
5. Total page count

Be thorough but concise. Extract actual content, not just metadata."""

            # Use OpenAI with text extraction
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=f"Analyze this PDF paper about GEO/SEO: {doc.filename}\n\nTotal pages: {total_pages}\n\nContent:\n{pdf_text}"
                ),
            ]

            # Get structured output
            extracted: ExtractedStructure = structured_llm.invoke(messages)

            # Create DocumentStructure (convert Section models to dicts)
            structure = DocumentStructure(
                title=extracted.title,
                authors=extracted.authors,
                abstract=extracted.abstract,
                sections=[{"title": s.title, "content": s.content} for s in extracted.sections],
                themes=extracted.themes,
                industries=extracted.industries,
                total_pages=extracted.total_pages,
                extraction_confidence=extracted.extraction_confidence,
            )

            analyzed_documents.append({"doc": doc, "structure": structure})

            logger.info(
                f"Successfully analyzed: {doc.filename} ({structure.total_pages} pages, {len(structure.sections)} sections)"
            )

        except Exception as e:
            logger.error(f"Error analyzing {doc.filename}: {str(e)}")
            error = ProcessingError(
                node_name="document_analyzer",
                error_type=type(e).__name__,
                error_message=str(e),
                timestamp=datetime.now(),
                recoverable=False,
            )
            errors.append(error)

    return {
        "analyzed_documents": analyzed_documents,
        "errors": errors,
        "current_node": "document_analyzer",
        "documents_processed_count": len(analyzed_documents),
    }


# ============================================================================
# NODE 2: Content Extractor & Classifier
# ============================================================================


def content_extractor_node(state: GraphState) -> dict[str, Any]:
    """
    Extract actionable guidelines from analyzed documents and classify them.

    Uses OpenAI GPT-4o to identify specific, actionable guidelines and categorize them
    into one of 5 categories.
    """
    logger.info(f"=== Node 2: Content Extractor & Classifier ===")

    raw_guidelines = []
    errors = []

    # Use centralized extractor LLM
    llm = get_extractor_llm()
    structured_llm = llm.with_structured_output(GuidelinesList)

    for analyzed in state.get("analyzed_documents", []):
        try:
            doc: PDFDocument = analyzed["doc"]
            structure: DocumentStructure = analyzed["structure"]

            logger.info(f"Extracting guidelines from: {doc.filename}")

            # Build extraction prompt with category descriptions
            categories_info = "\n".join(
                [f"- {cat}: {desc}" for cat, desc in CATEGORY_DESCRIPTIONS.items()]
            )

            system_prompt = f"""You are an expert at extracting actionable SEO/GEO guidelines from academic papers.

Extract specific, actionable guidelines and classify each into one of these categories:
{categories_info}

Guidelines should be:
- Specific and actionable (not vague observations)
- At least {config.processing.min_guideline_length} characters
- Include the source section and page numbers
- Classified accurately based on content

Extract up to {config.processing.max_guidelines_per_document} guidelines."""

            # Combine all sections for processing
            full_content = f"Title: {structure.title}\n\nAbstract: {structure.abstract}\n\n"
            for section in structure.sections:
                full_content += f"Section: {section.get('title', 'Untitled')}\n{section.get('content', '')}\n\n"

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=f"Extract actionable guidelines from this paper:\n\n{full_content[:15000]}"
                ),  # Limit to prevent token overflow
            ]

            # Get structured output
            result: GuidelinesList = structured_llm.invoke(messages)

            # Convert to Guideline objects
            for idx, extracted_guideline in enumerate(result.guidelines):
                guideline_id = hashlib.md5(
                    f"{doc.filename}_{idx}_{extracted_guideline.content}".encode()
                ).hexdigest()[:12]

                guideline = Guideline(
                    guideline_id=guideline_id,
                    content=extracted_guideline.content,
                    source_section=extracted_guideline.source_section,
                    page_numbers=extracted_guideline.page_numbers,
                    category=extracted_guideline.category,
                    industries=extracted_guideline.industries,
                    extraction_confidence=extracted_guideline.extraction_confidence,
                )

                raw_guidelines.append(guideline)

            logger.info(
                f"Extracted {len(result.guidelines)} guidelines from {doc.filename}"
            )

        except Exception as e:
            logger.error(f"Error extracting from {doc.filename}: {str(e)}")
            error = ProcessingError(
                node_name="content_extractor",
                error_type=type(e).__name__,
                error_message=str(e),
                timestamp=datetime.now(),
                recoverable=True,
            )
            errors.append(error)

    return {
        "raw_guidelines": raw_guidelines,
        "errors": errors,
        "current_node": "content_extractor",
        "total_guidelines_extracted": len(raw_guidelines),
    }


# ============================================================================
# NODE 3: Deduplication & Merging
# ============================================================================


def deduplication_node(state: GraphState) -> dict[str, Any]:
    """
    Deduplicate guidelines using semantic similarity.

    Compares guidelines using embeddings:
    - Similarity > 85%: Merge guidelines
    - Similarity < 85%: Keep separate
    """
    logger.info(f"=== Node 3: Deduplication & Merging ===")

    raw_guidelines = state.get("raw_guidelines", [])
    logger.info(f"Deduplicating {len(raw_guidelines)} guidelines")

    if len(raw_guidelines) == 0:
        return {
            "deduplicated_guidelines": [],
            "merge_log": [],
            "current_node": "deduplication",
        }

    # Use centralized embeddings client
    embeddings_model = get_embeddings()

    # Generate embeddings for all guidelines
    guideline_texts = [g.content for g in raw_guidelines]
    embeddings = embeddings_model.embed_documents(guideline_texts)

    # Simple cosine similarity deduplication
    import numpy as np

    embeddings_array = np.array(embeddings)
    deduplicated_guidelines = []
    merge_log = []
    processed_indices = set()

    for i in range(len(raw_guidelines)):
        if i in processed_indices:
            continue

        current_guideline = raw_guidelines[i]
        merged_content = [current_guideline.content]
        merged_sources = [current_guideline.source_section]
        merged_pages = current_guideline.page_numbers.copy()

        # Check similarity with remaining guidelines
        for j in range(i + 1, len(raw_guidelines)):
            if j in processed_indices:
                continue

            # Calculate cosine similarity
            similarity = np.dot(embeddings_array[i], embeddings_array[j]) / (
                np.linalg.norm(embeddings_array[i])
                * np.linalg.norm(embeddings_array[j])
            )

            if similarity > config.processing.similarity_threshold:
                # Merge guidelines
                merged_content.append(raw_guidelines[j].content)
                merged_sources.append(raw_guidelines[j].source_section)
                merged_pages.extend(raw_guidelines[j].page_numbers)
                processed_indices.add(j)

                merge_log.append(
                    {
                        "primary_id": current_guideline.guideline_id,
                        "merged_id": raw_guidelines[j].guideline_id,
                        "similarity": float(similarity),
                    }
                )

        # Create merged guideline
        if len(merged_content) > 1:
            # Combine content intelligently
            combined_content = f"{current_guideline.content} [Merged with {len(merged_content) - 1} similar guidelines]"
        else:
            combined_content = current_guideline.content

        merged_guideline = Guideline(
            guideline_id=current_guideline.guideline_id,
            content=combined_content,
            source_section="; ".join(set(merged_sources)),
            page_numbers=sorted(list(set(merged_pages))),
            category=current_guideline.category,
            industries=current_guideline.industries,
            extraction_confidence=current_guideline.extraction_confidence,
        )

        deduplicated_guidelines.append(merged_guideline)
        processed_indices.add(i)

    logger.info(
        f"Deduplication complete: {len(raw_guidelines)} -> {len(deduplicated_guidelines)} guidelines"
    )
    logger.info(f"Merged {len(merge_log)} duplicate guidelines")

    return {
        "deduplicated_guidelines": deduplicated_guidelines,
        "merge_log": merge_log,
        "current_node": "deduplication",
    }


# ============================================================================
# NODE 4: Collection Router
# ============================================================================


def collection_router_node(state: GraphState) -> dict[str, Any]:
    """
    Route guidelines to appropriate Qdrant collections based on category.
    """
    logger.info(f"=== Node 4: Collection Router ===")

    deduplicated_guidelines = state.get("deduplicated_guidelines", [])
    logger.info(f"Routing {len(deduplicated_guidelines)} guidelines to collections")

    routed_guidelines = {
        "universal_seo_geo": [],
        "industry_specific": [],
        "technical": [],
        "citation_optimization": [],
        "metrics": [],
    }

    for guideline in deduplicated_guidelines:
        category = guideline.category
        if category in routed_guidelines:
            routed_guidelines[category].append(guideline)
        else:
            logger.warning(
                f"Unknown category '{category}' for guideline {guideline.guideline_id}, routing to universal"
            )
            routed_guidelines["universal_seo_geo"].append(guideline)

    # Log routing statistics
    for category, guidelines in routed_guidelines.items():
        logger.info(f"  {category}: {len(guidelines)} guidelines")

    return {"routed_guidelines": routed_guidelines, "current_node": "collection_router"}


# ============================================================================
# NODE 5: Metadata Enricher
# ============================================================================


def metadata_enricher_node(state: GraphState) -> dict[str, Any]:
    """
    Enrich guidelines with metadata:
    - Confidence scores
    - Priority levels
    - Implementation complexity
    - Related guidelines
    """
    logger.info(f"=== Node 5: Metadata Enricher ===")

    routed_guidelines = state.get("routed_guidelines", {})
    all_guidelines = []
    for guidelines in routed_guidelines.values():
        all_guidelines.extend(guidelines)

    enriched_guidelines = []

    for guideline in all_guidelines:
        # Calculate priority based on confidence and content
        priority = _calculate_priority(guideline)

        # Determine implementation complexity
        complexity = _calculate_complexity(guideline)

        # Find related guidelines (simplified - based on category match)
        related_ids = [
            g.guideline_id
            for g in all_guidelines
            if g.guideline_id != guideline.guideline_id
            and g.category == guideline.category
        ][:5]  # Top 5 related

        enriched = EnrichedGuideline(
            guideline=guideline,
            confidence_score=guideline.extraction_confidence,
            priority=priority,
            implementation_complexity=complexity,
            related_guideline_ids=related_ids,
            semantic_cluster_id=None,  # Could add clustering later
        )

        enriched_guidelines.append(enriched)

    logger.info(f"Enriched {len(enriched_guidelines)} guidelines with metadata")

    return {"enriched_guidelines": enriched_guidelines, "current_node": "metadata_enricher"}


def _calculate_priority(guideline: Guideline) -> str:
    """Calculate priority based on confidence and impact indicators."""
    confidence = guideline.extraction_confidence

    # Simple heuristic: high confidence + action words = high priority
    action_words = ["must", "critical", "essential", "required", "always", "never"]
    impact_score = sum(1 for word in action_words if word in guideline.content.lower()) / len(
        action_words
    )

    if confidence >= 0.9 and impact_score >= 0.3:
        return "critical"
    elif confidence >= 0.75 and impact_score >= 0.2:
        return "high"
    elif confidence >= 0.6:
        return "medium"
    else:
        return "low"


def _calculate_complexity(guideline: Guideline) -> str:
    """Calculate implementation complexity based on content."""
    content_lower = guideline.content.lower()

    # Check for complexity indicators
    easy_count = sum(1 for word in COMPLEXITY_INDICATORS["easy"] if word in content_lower)
    moderate_count = sum(
        1 for word in COMPLEXITY_INDICATORS["moderate"] if word in content_lower
    )
    complex_count = sum(
        1 for word in COMPLEXITY_INDICATORS["complex"] if word in content_lower
    )

    if complex_count > moderate_count and complex_count > easy_count:
        return "complex"
    elif moderate_count > easy_count:
        return "moderate"
    else:
        return "easy"


# ============================================================================
# NODE 6: Vector Storage
# ============================================================================


def vector_storage_node(state: GraphState) -> dict[str, Any]:
    """
    Generate embeddings and store guidelines in Qdrant collections.
    """
    logger.info(f"=== Node 6: Vector Storage ===")

    enriched_guidelines = state.get("enriched_guidelines", [])
    logger.info(f"Storing {len(enriched_guidelines)} enriched guidelines in Qdrant")

    # Use centralized clients
    qdrant_client = get_qdrant_client()
    embeddings_model = get_embeddings()

    # Ensure collections exist using centralized function
    ensure_collections_exist()

    storage_results = []
    errors = []

    # Group by category for batch processing
    by_category = {}
    for enriched in enriched_guidelines:
        category = enriched.guideline.category
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(enriched)

    # Store in each collection
    for category, guidelines in by_category.items():
        try:
            collection_name = config.qdrant.collections[category]
            logger.info(f"Storing {len(guidelines)} guidelines in collection: {collection_name}")

            # Generate embeddings in batch
            texts = [g.guideline.content for g in guidelines]
            embeddings = embeddings_model.embed_documents(texts)

            # Create points
            points = []
            for idx, (enriched, embedding) in enumerate(zip(guidelines, embeddings)):
                point = PointStruct(
                    id=hash(enriched.guideline.guideline_id)
                    % (2**63 - 1),  # Ensure positive int
                    vector=embedding,
                    payload={
                        "guideline_id": enriched.guideline.guideline_id,
                        "content": enriched.guideline.content,
                        "category": enriched.guideline.category,
                        "source_section": enriched.guideline.source_section,
                        "page_numbers": enriched.guideline.page_numbers,
                        "industries": enriched.guideline.industries,
                        "confidence_score": enriched.confidence_score,
                        "priority": enriched.priority,
                        "implementation_complexity": enriched.implementation_complexity,
                        "related_guideline_ids": enriched.related_guideline_ids,
                        "extraction_confidence": enriched.guideline.extraction_confidence,
                    },
                )
                points.append(point)

            # Upsert to Qdrant
            qdrant_client.upsert(collection_name=collection_name, points=points)

            # Record success
            for guideline in guidelines:
                storage_results.append(
                    {
                        "guideline_id": guideline.guideline.guideline_id,
                        "collection": collection_name,
                        "vector_id": hash(guideline.guideline.guideline_id) % (2**63 - 1),
                        "status": "success",
                    }
                )

            logger.info(f"Successfully stored {len(points)} points in {collection_name}")

        except Exception as e:
            logger.error(f"Error storing in collection {collection_name}: {str(e)}")
            error = ProcessingError(
                node_name="vector_storage",
                error_type=type(e).__name__,
                error_message=str(e),
                timestamp=datetime.now(),
                recoverable=True,
            )
            errors.append(error)

    return {
        "storage_results": storage_results,
        "errors": errors,
        "current_node": "vector_storage",
    }


# Note: _ensure_collections_exist has been moved to src/clients/qdrant_client.py
# Use ensure_collections_exist() from src.clients instead
