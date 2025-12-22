"""
Qdrant vector database tools for the GEO/SEO Knowledge Base.

These tools enable the Deep Agent to:
1. Store guidelines in appropriate collections
2. Search for relevant guidelines
3. Get collection statistics
"""

import json
import hashlib
from typing import Optional
from datetime import datetime

from qdrant_client.models import (
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
)

# Use centralized clients
from src.clients import (
    get_qdrant_client,
    ensure_collections_exist,
    generate_embedding,
    calculate_priority,
    calculate_complexity,
    COLLECTIONS,
    COLLECTION_NAMES,
)

COLLECTION_DESCRIPTIONS = {
    "geo_seo_universal": "Universal SEO+GEO strategies applicable to all industries",
    "geo_seo_industry": "Industry-specific tactics (B2B SaaS, Healthcare, E-commerce, Finance)",
    "geo_seo_technical": "Technical implementation: schema markup, structured data, entity linking",
    "geo_seo_citation": "Citation optimization: content structuring for AI citations",
    "geo_seo_metrics": "Measurement metrics: KPIs, ROI tracking, benchmarks",
}


def store_guidelines(
    guidelines: list[dict],
    source_paper: dict
) -> str:
    """
    Store extracted guidelines in the appropriate Qdrant collections.

    This tool takes guidelines extracted from a paper and stores them
    in the correct collection based on their category.

    Args:
        guidelines: List of guideline dictionaries with:
            - content: The guideline text
            - category: One of universal_seo_geo, industry_specific, technical, citation_optimization, metrics
            - industries: List of applicable industries (empty if universal)
            - source_section: Section where found
            - confidence: Confidence score 0-1
            - has_quantitative_data: Boolean
            - has_case_study: Boolean
        source_paper: Dictionary with paper metadata:
            - title: Paper title
            - authors: List of authors
            - url: Source URL
            - filepath: Local file path

    Returns:
        JSON string with storage results
    """
    ensure_collections_exist()
    client = get_qdrant_client()

    result = {
        "total_guidelines": len(guidelines),
        "stored": 0,
        "failed": 0,
        "by_collection": {},
        "errors": [],
    }

    for guideline in guidelines:
        try:
            # Determine collection
            category = guideline.get("category", "universal_seo_geo")
            collection_name = COLLECTIONS.get(category, "geo_seo_universal")

            # Generate unique ID
            content_hash = hashlib.md5(
                f"{guideline['content']}_{source_paper.get('title', '')}".encode()
            ).hexdigest()[:12]
            guideline_id = f"guid_{content_hash}"

            # Calculate enriched confidence score
            confidence = guideline.get("confidence", 0.5)
            if guideline.get("has_quantitative_data"):
                confidence = min(confidence + 0.1, 1.0)
            if guideline.get("has_case_study"):
                confidence = min(confidence + 0.1, 1.0)

            # Use centralized priority calculation
            priority = calculate_priority(confidence, impact_score=0.5)

            # Use centralized complexity calculation
            complexity = calculate_complexity(guideline["content"])

            # Generate embedding using centralized function
            embedding = generate_embedding(guideline["content"])

            # Create payload
            payload = {
                "guideline_id": guideline_id,
                "guideline_text": guideline["content"],
                "category": category,
                "industry_tags": guideline.get("industries", []),
                "applicability": "specific" if guideline.get("industries") else "universal",
                "source_paper": {
                    "title": source_paper.get("title", "Unknown"),
                    "authors": source_paper.get("authors", []),
                    "url": source_paper.get("url", ""),
                    "excerpt": guideline.get("source_section", ""),
                },
                "confidence_score": confidence,
                "priority": priority,
                "implementation_complexity": complexity,
                "expected_impact": "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low",
                "has_quantitative_data": guideline.get("has_quantitative_data", False),
                "has_case_study": guideline.get("has_case_study", False),
                "created_at": datetime.now().isoformat(),
            }

            # Create point
            point = PointStruct(
                id=abs(hash(guideline_id)) % (2**63 - 1),
                vector=embedding,
                payload=payload,
            )

            # Upsert to Qdrant
            client.upsert(
                collection_name=collection_name,
                points=[point],
            )

            result["stored"] += 1

            # Track by collection
            if collection_name not in result["by_collection"]:
                result["by_collection"][collection_name] = 0
            result["by_collection"][collection_name] += 1

        except Exception as e:
            result["failed"] += 1
            result["errors"].append({
                "guideline": guideline.get("content", "")[:100],
                "error": str(e),
            })

    return json.dumps(result, indent=2)


def search_guidelines(
    query: str,
    collection: Optional[str] = None,
    industry: Optional[str] = None,
    priority: Optional[str] = None,
    limit: int = 10
) -> str:
    """
    Search for relevant guidelines in the knowledge base.

    This tool performs semantic search across the guidelines collections
    to find the most relevant recommendations.

    Args:
        query: Search query (e.g., "How to optimize B2B SaaS content for citations?")
        collection: Optional specific collection to search:
            - "universal": Universal SEO+GEO framework
            - "industry": Industry-specific strategies
            - "technical": Technical implementation
            - "citation": Citation optimization tactics
            - "metrics": Measurement metrics
            - None: Search all collections
        industry: Optional industry filter (e.g., "B2B_SaaS", "Healthcare", "Ecommerce")
        priority: Optional priority filter ("high", "medium", "low")
        limit: Maximum results to return (default: 10)

    Returns:
        JSON string with ranked search results including guidelines and metadata
    """
    ensure_collections_exist()
    client = get_qdrant_client()

    # Generate query embedding using centralized function
    query_embedding = generate_embedding(query)

    # Determine which collections to search
    if collection:
        collection_map = {
            "universal": "geo_seo_universal",
            "industry": "geo_seo_industry",
            "technical": "geo_seo_technical",
            "citation": "geo_seo_citation",
            "metrics": "geo_seo_metrics",
        }
        collections_to_search = [collection_map.get(collection, "geo_seo_universal")]
    else:
        collections_to_search = COLLECTION_NAMES

    # Build filter conditions
    filter_conditions = []
    if industry:
        filter_conditions.append(
            FieldCondition(key="industry_tags", match=MatchAny(any=[industry]))
        )
    if priority:
        filter_conditions.append(
            FieldCondition(key="priority", match=MatchValue(value=priority))
        )

    search_filter = Filter(must=filter_conditions) if filter_conditions else None

    # Search across collections
    all_results = []

    for collection_name in collections_to_search:
        try:
            results = client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                query_filter=search_filter,
                limit=limit,
                with_payload=True,
            )

            for result in results:
                payload = result.payload
                # Calculate combined score
                relevance_score = result.score
                confidence_score = payload.get("confidence_score", 0.5)
                priority_weight = {"high": 1.0, "medium": 0.8, "low": 0.6, "critical": 1.2}.get(
                    payload.get("priority", "medium"), 0.8
                )
                combined_score = relevance_score * confidence_score * priority_weight

                all_results.append({
                    "guideline_id": payload.get("guideline_id"),
                    "guideline_text": payload.get("guideline_text"),
                    "category": payload.get("category"),
                    "industry_tags": payload.get("industry_tags", []),
                    "confidence_score": confidence_score,
                    "relevance_score": round(relevance_score, 4),
                    "combined_score": round(combined_score, 4),
                    "priority": payload.get("priority"),
                    "implementation_complexity": payload.get("implementation_complexity"),
                    "source_paper": payload.get("source_paper", {}),
                    "collection": collection_name,
                })

        except Exception:
            # Collection might not exist or other error
            continue

    # Sort by combined score and limit
    all_results.sort(key=lambda x: x["combined_score"], reverse=True)
    all_results = all_results[:limit]

    return json.dumps({
        "query": query,
        "filters": {
            "collection": collection,
            "industry": industry,
            "priority": priority,
        },
        "total_results": len(all_results),
        "results": all_results,
    }, indent=2)


def get_collection_stats() -> str:
    """
    Get statistics about all collections in the knowledge base.

    Returns counts, distribution by category, industry, and priority.

    Returns:
        JSON string with comprehensive statistics
    """
    ensure_collections_exist()
    client = get_qdrant_client()

    stats = {
        "collections": {},
        "total_guidelines": 0,
        "by_priority": {"high": 0, "medium": 0, "low": 0, "critical": 0},
        "by_industry": {},
        "average_confidence": 0,
    }

    confidence_sum = 0
    total_count = 0

    for collection_name in COLLECTION_NAMES:
        description = COLLECTION_DESCRIPTIONS.get(collection_name, "")
        try:
            collection_info = client.get_collection(collection_name)
            count = collection_info.points_count

            stats["collections"][collection_name] = {
                "description": description,
                "count": count,
                "status": "active",
            }

            stats["total_guidelines"] += count

            # Sample some points to get distribution
            if count > 0:
                sample = client.scroll(
                    collection_name=collection_name,
                    limit=min(count, 100),
                    with_payload=True,
                )[0]

                for point in sample:
                    payload = point.payload
                    # Priority distribution
                    priority = payload.get("priority", "medium")
                    stats["by_priority"][priority] = stats["by_priority"].get(priority, 0) + 1

                    # Industry distribution
                    for industry in payload.get("industry_tags", []):
                        stats["by_industry"][industry] = stats["by_industry"].get(industry, 0) + 1

                    # Confidence average
                    confidence_sum += payload.get("confidence_score", 0.5)
                    total_count += 1

        except Exception as e:
            stats["collections"][collection_name] = {
                "description": description,
                "count": 0,
                "status": "not_found",
                "error": str(e),
            }

    # Calculate average confidence
    if total_count > 0:
        stats["average_confidence"] = round(confidence_sum / total_count, 3)

    return json.dumps(stats, indent=2)


def get_related_guidelines(
    guideline_id: str,
    limit: int = 5
) -> str:
    """
    Find guidelines related to a specific guideline.

    Uses semantic similarity to find related recommendations.

    Args:
        guideline_id: The ID of the guideline to find related ones for
        limit: Maximum related guidelines to return

    Returns:
        JSON string with related guidelines
    """
    client = get_qdrant_client()

    # First, find the original guideline
    original = None
    original_collection = None

    for collection_name in COLLECTION_NAMES:
        try:
            results = client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="guideline_id", match=MatchValue(value=guideline_id))]
                ),
                limit=1,
                with_payload=True,
                with_vectors=True,
            )[0]

            if results:
                original = results[0]
                original_collection = collection_name
                break
        except Exception:
            continue

    if not original:
        return json.dumps({
            "error": f"Guideline {guideline_id} not found",
            "related_guidelines": []
        })

    # Search for similar guidelines using the original's vector
    related = []

    for collection_name in COLLECTION_NAMES:
        try:
            results = client.search(
                collection_name=collection_name,
                query_vector=original.vector,
                limit=limit + 1,  # +1 to exclude self
                with_payload=True,
            )

            for result in results:
                if result.payload.get("guideline_id") != guideline_id:
                    related.append({
                        "guideline_id": result.payload.get("guideline_id"),
                        "guideline_text": result.payload.get("guideline_text"),
                        "category": result.payload.get("category"),
                        "similarity_score": round(result.score, 4),
                        "collection": collection_name,
                    })
        except Exception:
            continue

    # Sort by similarity and limit
    related.sort(key=lambda x: x["similarity_score"], reverse=True)
    related = related[:limit]

    return json.dumps({
        "original_guideline_id": guideline_id,
        "original_text": original.payload.get("guideline_text"),
        "related_guidelines": related,
    }, indent=2)
