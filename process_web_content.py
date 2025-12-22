"""
Process crawled web content and extract guidelines into Qdrant.
"""
import uuid
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from qdrant_client.models import PointStruct, VectorParams, Distance
from pydantic import BaseModel, Field
from typing import List

# Use centralized clients
from src.clients import (
    get_qdrant_client,
    get_embeddings,
    get_extractor_llm,
    ensure_collections_exist,
    COLLECTIONS,
)

# Models for structured output
class ExtractedGuideline(BaseModel):
    content: str = Field(description="The guideline content")
    category: str = Field(description="Category: universal_seo_geo, industry_specific, technical, citation_optimization, or metrics")
    source_section: str = Field(description="Section where guideline was found")
    industries: List[str] = Field(default_factory=list, description="Applicable industries")
    confidence_score: float = Field(default=0.85, description="Confidence score 0-1")
    priority: str = Field(default="medium", description="Priority: critical, high, medium, low")
    implementation_complexity: str = Field(default="moderate", description="Complexity: easy, moderate, complex")

class GuidelinesExtraction(BaseModel):
    guidelines: List[ExtractedGuideline] = Field(default_factory=list, description="Extracted guidelines")

# Note: COLLECTIONS mapping is now imported from src.clients

def extract_guidelines_from_content(content: str, source_name: str) -> List[dict]:
    """Extract guidelines from markdown content using OpenAI."""

    # Use centralized extractor LLM
    llm = get_extractor_llm()

    prompt = f"""Analyze this GEO/SEO content and extract specific, actionable guidelines.

SOURCE: {source_name}

CONTENT:
{content}

Extract guidelines that are:
1. Specific and actionable (not vague advice)
2. Related to GEO (Generative Engine Optimization) or SEO
3. Backed by data or expert recommendation

For each guideline, determine:
- category: universal_seo_geo (applies broadly), industry_specific (for specific industries), technical (implementation details), citation_optimization (about citations/sources), or metrics (measurement/analytics)
- priority: critical (must-do), high (important), medium (recommended), low (nice-to-have)
- implementation_complexity: easy (quick wins), moderate (requires planning), complex (major effort)
- industries: list applicable industries like B2B_SaaS, Healthcare, Ecommerce, Finance, etc. (empty if universal)

Return at least 10-20 specific guidelines."""

    structured_llm = llm.with_structured_output(GuidelinesExtraction)
    result = structured_llm.invoke(prompt)

    guidelines = []
    for g in result.guidelines:
        guidelines.append({
            "guideline_id": uuid.uuid4().hex[:12],
            "content": g.content,
            "category": g.category,
            "source_section": g.source_section,
            "source_document": source_name,
            "page_numbers": [],
            "industries": g.industries,
            "confidence_score": g.confidence_score,
            "priority": g.priority,
            "implementation_complexity": g.implementation_complexity,
            "related_guideline_ids": [],
            "created_at": datetime.now().isoformat(),
        })

    return guidelines

def store_guidelines_in_qdrant(guidelines: List[dict]):
    """Store extracted guidelines in Qdrant."""

    # Use centralized clients
    client = get_qdrant_client()
    embeddings = get_embeddings()

    # Ensure collections exist using centralized function
    ensure_collections_exist()

    # Group guidelines by category
    by_category = {}
    for g in guidelines:
        cat = g["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(g)

    total_stored = 0

    for category, items in by_category.items():
        collection_name = COLLECTIONS.get(category, COLLECTIONS["universal_seo_geo"])

        # Generate embeddings
        texts = [g["content"] for g in items]
        vectors = embeddings.embed_documents(texts)

        # Create points
        points = []
        for g, vector in zip(items, vectors):
            points.append(PointStruct(
                id=uuid.uuid4().hex,
                vector=vector,
                payload=g,
            ))

        # Upsert to collection
        client.upsert(collection_name=collection_name, points=points)
        total_stored += len(points)
        print(f"  Stored {len(points)} guidelines in {collection_name}")

    return total_stored

def process_markdown_files(directory: str):
    """Process all markdown files in directory."""

    md_files = list(Path(directory).glob("*.md"))
    print(f"Found {len(md_files)} markdown files to process")

    all_guidelines = []

    for md_file in md_files:
        print(f"\nProcessing: {md_file.name}")
        content = md_file.read_text()

        guidelines = extract_guidelines_from_content(content, md_file.stem)
        print(f"  Extracted {len(guidelines)} guidelines")
        all_guidelines.extend(guidelines)

    if all_guidelines:
        print(f"\nStoring {len(all_guidelines)} total guidelines in Qdrant...")
        stored = store_guidelines_in_qdrant(all_guidelines)
        print(f"Successfully stored {stored} guidelines")

    return all_guidelines

if __name__ == "__main__":
    content_dir = "/Users/umerkhan/CascadeProjects/windsurf-project/AISEO/crawled_content"
    guidelines = process_markdown_files(content_dir)
    print(f"\n=== Processing Complete ===")
    print(f"Total guidelines extracted and stored: {len(guidelines)}")
