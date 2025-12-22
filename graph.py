"""
LangGraph workflow definition for GEO/SEO Knowledge Base system.

This is the main graph that orchestrates the 6-node processing pipeline.
Designed for deployment without checkpointer (deployment-first approach).
"""

from typing import Literal
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
import logging

from state import GraphState
from nodes import (
    document_analyzer_node,
    content_extractor_node,
    deduplication_node,
    collection_router_node,
    metadata_enricher_node,
    vector_storage_node,
)

logger = logging.getLogger(__name__)


def should_continue_to_extraction(state: GraphState) -> Literal["content_extractor", "end"]:
    """
    Conditional edge: Check if document analysis was successful.

    If no documents were analyzed successfully, end the workflow.
    """
    analyzed_count = state.get("documents_processed_count", 0)

    if analyzed_count == 0:
        logger.warning("No documents were successfully analyzed. Ending workflow.")
        return "end"

    return "content_extractor"


def should_continue_to_deduplication(state: GraphState) -> Literal["deduplication", "end"]:
    """
    Conditional edge: Check if any guidelines were extracted.
    """
    guidelines_count = state.get("total_guidelines_extracted", 0)

    if guidelines_count == 0:
        logger.warning("No guidelines extracted. Ending workflow.")
        return "end"

    return "deduplication"


def create_workflow(enable_checkpointing: bool = False) -> StateGraph:
    """
    Create the GEO/SEO Knowledge Base LangGraph workflow.

    Args:
        enable_checkpointing: If True, enables checkpointing for persistence.
                            Default False for deployment-first approach.

    Returns:
        Compiled StateGraph ready for execution
    """

    # Create the graph
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("document_analyzer", document_analyzer_node)
    workflow.add_node("content_extractor", content_extractor_node)
    workflow.add_node("deduplication", deduplication_node)
    workflow.add_node("collection_router", collection_router_node)
    workflow.add_node("metadata_enricher", metadata_enricher_node)
    workflow.add_node("vector_storage", vector_storage_node)

    # Add edges
    # START -> Document Analyzer (always starts here)
    workflow.add_edge(START, "document_analyzer")

    # Document Analyzer -> Content Extractor (conditional)
    workflow.add_conditional_edges(
        "document_analyzer",
        should_continue_to_extraction,
        {
            "content_extractor": "content_extractor",
            "end": END,
        },
    )

    # Content Extractor -> Deduplication (conditional)
    workflow.add_conditional_edges(
        "content_extractor",
        should_continue_to_deduplication,
        {
            "deduplication": "deduplication",
            "end": END,
        },
    )

    # Deduplication -> Collection Router (always continues)
    workflow.add_edge("deduplication", "collection_router")

    # Collection Router -> Metadata Enricher (always continues)
    workflow.add_edge("collection_router", "metadata_enricher")

    # Metadata Enricher -> Vector Storage (always continues)
    workflow.add_edge("metadata_enricher", "vector_storage")

    # Vector Storage -> END (workflow complete)
    workflow.add_edge("vector_storage", END)

    # Compile the graph
    if enable_checkpointing:
        # Use MemorySaver for checkpointing (can be replaced with PostgresSaver, etc.)
        checkpointer = MemorySaver()
        logger.info("Compiling graph WITH checkpointing enabled")
        compiled_graph = workflow.compile(checkpointer=checkpointer)
    else:
        logger.info("Compiling graph WITHOUT checkpointing (deployment-first)")
        compiled_graph = workflow.compile()

    return compiled_graph


# ============================================================================
# Export the compiled graph as 'app' for LangGraph deployment
# ============================================================================

# DEPLOYMENT EXPORT: No checkpointer for production deployment
app = create_workflow(enable_checkpointing=False)

# For local testing with persistence, use this instead:
# app = create_workflow(enable_checkpointing=True)


# ============================================================================
# Visualization Helper
# ============================================================================


def visualize_graph(output_path: str = "graph_visualization.png"):
    """
    Generate a visual representation of the graph.

    Args:
        output_path: Path to save the visualization image
    """
    try:
        graph = create_workflow(enable_checkpointing=False)
        graph_image = graph.get_graph().draw_mermaid_png()

        with open(output_path, "wb") as f:
            f.write(graph_image)

        logger.info(f"Graph visualization saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to generate visualization: {str(e)}")


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    # Example: Run the workflow with sample data
    from datetime import datetime
    from state import PDFDocument

    # Create sample input
    sample_documents = [
        PDFDocument(
            file_path="/path/to/sample.pdf",
            filename="sample_geo_paper.pdf",
            upload_timestamp=datetime.now(),
            file_size_mb=2.5,
        )
    ]

    initial_state = {
        "documents": sample_documents,
        "analyzed_documents": [],
        "raw_guidelines": [],
        "deduplicated_guidelines": [],
        "routed_guidelines": {},
        "enriched_guidelines": [],
        "storage_results": [],
        "errors": [],
        "batch_id": "test_batch_001",
        "processing_start_time": datetime.now(),
        "current_node": "start",
        "retry_count": 0,
        "documents_processed_count": 0,
        "total_guidelines_extracted": 0,
        "memory_threshold_exceeded": False,
    }

    # Run the workflow
    graph = create_workflow(enable_checkpointing=True)

    config = {"configurable": {"thread_id": "test_run_001"}}

    print("Starting workflow execution...")
    final_state = graph.invoke(initial_state, config=config)

    print("\n=== Workflow Complete ===")
    print(f"Documents processed: {final_state.get('documents_processed_count', 0)}")
    print(f"Guidelines extracted: {final_state.get('total_guidelines_extracted', 0)}")
    print(f"Guidelines stored: {len(final_state.get('storage_results', []))}")
    print(f"Errors encountered: {len(final_state.get('errors', []))}")

    # Visualize the graph
    visualize_graph()
