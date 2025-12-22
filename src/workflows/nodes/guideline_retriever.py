"""
Guideline Retriever Node: Fetches relevant guidelines from Qdrant KB.

Queries multiple collections with industry-specific filtering to get
the most relevant optimization guidelines for each chunk.
"""

import logging
from typing import List, Dict, Optional
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

# Use centralized clients
from src.clients import (
    get_qdrant_client,
    get_embeddings,
    COLLECTIONS,
    COLLECTION_NAMES,
)

logger = logging.getLogger(__name__)


class GuidelineRetriever:
    """Retrieves guidelines from Qdrant vector database."""

    def __init__(self):
        self.client = get_qdrant_client()
        self.embeddings = get_embeddings()

    def retrieve_for_chunk(
        self,
        chunk_content: str,
        industry: Optional[str] = None,
        limit_per_collection: int = 3,
        min_score: float = 0.3
    ) -> List[Dict]:
        """
        Retrieve relevant guidelines for a content chunk.

        Args:
            chunk_content: The content to find guidelines for
            industry: Optional industry filter
            limit_per_collection: Max guidelines per collection
            min_score: Minimum similarity score

        Returns:
            List of guideline dicts with content and metadata
        """
        # Generate embedding for chunk content
        query_vector = self.embeddings.embed_query(chunk_content)

        all_guidelines = []

        # 1. Always get universal guidelines (most important)
        universal = self._query_collection(
            COLLECTIONS["universal"],
            query_vector,
            limit=limit_per_collection + 2,  # Get more universals
            min_score=min_score
        )
        all_guidelines.extend(universal)

        # 2. Get industry-specific guidelines if industry provided
        if industry:
            industry_filter = Filter(
                should=[
                    FieldCondition(key="industries", match=MatchAny(any=[industry])),
                    FieldCondition(key="industries", match=MatchValue(value=industry)),
                ]
            )
            industry_guidelines = self._query_collection(
                COLLECTIONS["industry"],
                query_vector,
                limit=limit_per_collection,
                filter_obj=industry_filter,
                min_score=min_score
            )
            all_guidelines.extend(industry_guidelines)

        # 3. Get technical guidelines
        technical = self._query_collection(
            COLLECTIONS["technical"],
            query_vector,
            limit=limit_per_collection,
            min_score=min_score
        )
        all_guidelines.extend(technical)

        # 4. Get citation guidelines
        citation = self._query_collection(
            COLLECTIONS["citation"],
            query_vector,
            limit=limit_per_collection,
            min_score=min_score
        )
        all_guidelines.extend(citation)

        # 5. Get metrics guidelines (fewer)
        metrics = self._query_collection(
            COLLECTIONS["metrics"],
            query_vector,
            limit=2,
            min_score=min_score
        )
        all_guidelines.extend(metrics)

        # Sort by score and deduplicate
        seen_ids = set()
        unique_guidelines = []
        for g in sorted(all_guidelines, key=lambda x: x["score"], reverse=True):
            if g["guideline_id"] not in seen_ids:
                seen_ids.add(g["guideline_id"])
                unique_guidelines.append(g)

        return unique_guidelines[:15]  # Return top 15

    def _query_collection(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5,
        filter_obj: Optional[Filter] = None,
        min_score: float = 0.3
    ) -> List[Dict]:
        """Query a single collection."""
        try:
            if not self.client.collection_exists(collection_name):
                return []

            results = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                query_filter=filter_obj,
                limit=limit,
            )

            guidelines = []
            for point in results.points:
                if point.score >= min_score:
                    guidelines.append({
                        "guideline_id": point.payload.get("guideline_id", ""),
                        "content": point.payload.get("content", ""),
                        "category": point.payload.get("category", ""),
                        "priority": point.payload.get("priority", "medium"),
                        "implementation_complexity": point.payload.get("implementation_complexity", "moderate"),
                        "industries": point.payload.get("industries", []),
                        "source_document": point.payload.get("source_document", ""),
                        "score": point.score,
                    })

            return guidelines

        except Exception as e:
            logger.warning(f"Error querying {collection_name}: {e}")
            return []

    def get_guidelines_summary(self, guidelines: List[Dict]) -> str:
        """Format guidelines into a summary string for LLM context."""
        if not guidelines:
            return "No specific guidelines retrieved."

        summary_parts = []
        for i, g in enumerate(guidelines, 1):
            priority_marker = "🔴" if g["priority"] == "critical" else "🟡" if g["priority"] == "high" else "⚪"
            summary_parts.append(
                f"{i}. [{priority_marker} {g['priority'].upper()}] {g['content']}\n"
                f"   Category: {g['category']} | Complexity: {g['implementation_complexity']}"
            )

        return "\n\n".join(summary_parts)


# Singleton instance
_retriever = None


def get_retriever() -> GuidelineRetriever:
    """Get or create singleton retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = GuidelineRetriever()
    return _retriever


def retrieve_guidelines(chunk_content: str, industry: str = None) -> List[Dict]:
    """Convenience function for retrieving guidelines."""
    retriever = get_retriever()
    return retriever.retrieve_for_chunk(chunk_content, industry)


if __name__ == "__main__":
    # Test retrieval
    test_content = """
    Our B2B SaaS platform helps companies automate their documentation.
    We offer API integration and enterprise features.
    """

    guidelines = retrieve_guidelines(test_content, "B2B_SaaS")
    print(f"Retrieved {len(guidelines)} guidelines:")
    for g in guidelines[:5]:
        print(f"  - [{g['category']}] {g['content'][:80]}... (score: {g['score']:.2f})")
