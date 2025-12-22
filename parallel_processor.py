"""
Parallel document processing using LangGraph's Send() API.

This module demonstrates how to process multiple PDFs in parallel using
the Send() API for improved performance with large document batches.
"""

from typing import Literal, Any
from langgraph.graph import StateGraph, END, START
from langgraph.types import Send
import logging

from state import DocumentBatchState, PDFDocument, ProcessingError
from nodes import document_analyzer_node, content_extractor_node
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# Parallel Processing Nodes
# ============================================================================


def parallel_document_processor(batch_state: DocumentBatchState) -> dict[str, Any]:
    """
    Process a single document in parallel.

    This node is executed in parallel for each document using Send() API.
    """
    doc = batch_state["document"]
    logger.info(f"[Parallel] Processing document: {doc.filename}")

    try:
        # Simulate document analysis for a single doc
        from nodes import document_analyzer_node

        # Create minimal state for single document
        single_doc_state = {
            "documents": [doc],
            "analyzed_documents": [],
            "raw_guidelines": [],
            "errors": [],
            "current_node": "parallel_analyzer",
            "documents_processed_count": 0,
            "total_guidelines_extracted": 0,
        }

        # Run analyzer
        result = document_analyzer_node(single_doc_state)

        if len(result.get("analyzed_documents", [])) > 0:
            analyzed = result["analyzed_documents"][0]

            # Run extractor on this single document
            extraction_state = {
                "analyzed_documents": [analyzed],
                "raw_guidelines": [],
                "errors": [],
                "current_node": "parallel_extractor",
                "total_guidelines_extracted": 0,
            }

            extraction_result = content_extractor_node(extraction_state)

            return {
                "document": doc,
                "structure": analyzed["structure"],
                "guidelines": extraction_result.get("raw_guidelines", []),
                "error": None,
                "status": "completed",
            }
        else:
            # Analysis failed
            return {
                "document": doc,
                "structure": None,
                "guidelines": [],
                "error": ProcessingError(
                    node_name="parallel_processor",
                    error_type="AnalysisError",
                    error_message="Document analysis failed",
                    timestamp=datetime.now(),
                    recoverable=False,
                ),
                "status": "failed",
            }

    except Exception as e:
        logger.error(f"[Parallel] Error processing {doc.filename}: {str(e)}")
        return {
            "document": doc,
            "structure": None,
            "guidelines": [],
            "error": ProcessingError(
                node_name="parallel_processor",
                error_type=type(e).__name__,
                error_message=str(e),
                timestamp=datetime.now(),
                recoverable=False,
            ),
            "status": "failed",
        }


# ============================================================================
# Parallel Workflow with Send() API
# ============================================================================


def create_parallel_workflow() -> StateGraph:
    """
    Create a parallel document processing workflow using Send() API.

    This workflow processes multiple documents simultaneously for better performance.
    """

    # Define a custom state for parallel processing
    from typing import TypedDict, Annotated
    from operator import add

    class ParallelGraphState(TypedDict):
        """State for parallel document processing."""

        documents: list[PDFDocument]
        batch_results: Annotated[list[DocumentBatchState], add]
        all_guidelines: Annotated[list, add]
        errors: Annotated[list[ProcessingError], add]

    def dispatch_documents(state: ParallelGraphState) -> list[Send]:
        """
        Dispatcher node that sends each document to parallel processing.

        Uses Send() API to create parallel execution paths.
        """
        documents = state.get("documents", [])
        logger.info(f"[Dispatcher] Sending {len(documents)} documents for parallel processing")

        # Create a Send for each document
        return [
            Send(
                "process_document",
                {
                    "document": doc,
                    "structure": None,
                    "guidelines": [],
                    "error": None,
                    "status": "pending",
                },
            )
            for doc in documents
        ]

    def aggregate_results(state: ParallelGraphState) -> dict[str, Any]:
        """
        Aggregate results from all parallel document processing.
        """
        batch_results = state.get("batch_results", [])
        logger.info(f"[Aggregator] Aggregating {len(batch_results)} parallel results")

        all_guidelines = []
        errors = []

        for result in batch_results:
            if result["status"] == "completed":
                all_guidelines.extend(result.get("guidelines", []))
            elif result["error"] is not None:
                errors.append(result["error"])

        logger.info(
            f"[Aggregator] Collected {len(all_guidelines)} guidelines, {len(errors)} errors"
        )

        return {"all_guidelines": all_guidelines, "errors": errors}

    # Build the graph
    workflow = StateGraph(ParallelGraphState)

    # Add nodes
    workflow.add_node("dispatch", dispatch_documents)
    workflow.add_node("process_document", parallel_document_processor)
    workflow.add_node("aggregate", aggregate_results)

    # Add edges
    workflow.add_edge(START, "dispatch")
    workflow.add_conditional_edges(
        "dispatch",
        lambda state: [
            Send("process_document", batch)
            for batch in dispatch_documents(state)
        ],
    )
    workflow.add_edge("process_document", "aggregate")
    workflow.add_edge("aggregate", END)

    return workflow.compile()


# ============================================================================
# Memory-Aware Batch Processor
# ============================================================================


class MemoryAwareBatchProcessor:
    """
    Processes documents in memory-safe batches.

    Monitors memory usage and processes documents in chunks to prevent OOM errors.
    """

    def __init__(self, max_concurrent_docs: int = 3, max_memory_mb: int = 2048):
        self.max_concurrent_docs = max_concurrent_docs
        self.max_memory_mb = max_memory_mb

    def process_in_batches(self, documents: list[PDFDocument], graph: StateGraph) -> dict:
        """
        Process documents in memory-safe batches.

        Args:
            documents: List of PDFs to process
            graph: Compiled LangGraph workflow

        Returns:
            Aggregated results from all batches
        """
        import psutil
        import os

        process = psutil.Process(os.getpid())

        all_guidelines = []
        all_errors = []

        # Split into batches
        for i in range(0, len(documents), self.max_concurrent_docs):
            batch = documents[i : i + self.max_concurrent_docs]

            # Check memory before processing
            memory_mb = process.memory_info().rss / 1024 / 1024
            logger.info(
                f"[Batch {i // self.max_concurrent_docs + 1}] Current memory: {memory_mb:.2f}MB"
            )

            if memory_mb > self.max_memory_mb * 0.8:  # 80% threshold
                logger.warning(
                    f"Memory usage high ({memory_mb:.2f}MB), waiting for garbage collection..."
                )
                import gc

                gc.collect()

            # Process batch
            logger.info(f"Processing batch of {len(batch)} documents...")

            batch_state = {
                "documents": batch,
                "batch_results": [],
                "all_guidelines": [],
                "errors": [],
            }

            result = graph.invoke(batch_state)

            # Aggregate
            all_guidelines.extend(result.get("all_guidelines", []))
            all_errors.extend(result.get("errors", []))

            logger.info(
                f"Batch complete. Total guidelines so far: {len(all_guidelines)}"
            )

        return {
            "all_guidelines": all_guidelines,
            "errors": all_errors,
            "total_batches": (len(documents) + self.max_concurrent_docs - 1)
            // self.max_concurrent_docs,
        }


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    from datetime import datetime

    # Create sample documents
    sample_docs = [
        PDFDocument(
            file_path=f"/path/to/paper_{i}.pdf",
            filename=f"paper_{i}.pdf",
            upload_timestamp=datetime.now(),
            file_size_mb=2.0 + i * 0.5,
        )
        for i in range(10)
    ]

    # Option 1: Use parallel workflow directly
    parallel_graph = create_parallel_workflow()

    initial_state = {
        "documents": sample_docs,
        "batch_results": [],
        "all_guidelines": [],
        "errors": [],
    }

    print("Running parallel workflow...")
    result = parallel_graph.invoke(initial_state)

    print(f"\nParallel processing complete!")
    print(f"Guidelines extracted: {len(result.get('all_guidelines', []))}")
    print(f"Errors: {len(result.get('errors', []))}")

    # Option 2: Use memory-aware batch processor
    print("\n\nRunning memory-aware batch processor...")
    processor = MemoryAwareBatchProcessor(max_concurrent_docs=3, max_memory_mb=2048)

    batch_result = processor.process_in_batches(sample_docs, parallel_graph)

    print(f"\nBatch processing complete!")
    print(f"Total batches: {batch_result['total_batches']}")
    print(f"Guidelines extracted: {len(batch_result['all_guidelines'])}")
    print(f"Errors: {len(batch_result['errors'])}")
