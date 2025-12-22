"""
Example usage of the GEO/SEO Knowledge Base system.

Demonstrates:
1. Basic workflow execution
2. Parallel processing
3. Memory-aware batch processing
4. API integration
5. Error handling
"""

import os
from datetime import datetime
from pathlib import Path

# Ensure environment variables are set
os.environ["GOOGLE_API_KEY"] = "your_gemini_api_key"
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"


# ============================================================================
# Example 1: Basic Single PDF Processing
# ============================================================================


def example_single_pdf():
    """Process a single PDF through the workflow."""
    from graph import create_workflow
    from state import PDFDocument

    print("=== Example 1: Single PDF Processing ===\n")

    # Create workflow with checkpointing for local testing
    graph = create_workflow(enable_checkpointing=True)

    # Create sample PDF document
    pdf_doc = PDFDocument(
        file_path="/path/to/geo_paper.pdf",  # Update with actual path
        filename="geo_optimization_paper.pdf",
        upload_timestamp=datetime.now(),
        file_size_mb=3.2,
    )

    # Prepare initial state
    initial_state = {
        "documents": [pdf_doc],
        "analyzed_documents": [],
        "raw_guidelines": [],
        "deduplicated_guidelines": [],
        "routed_guidelines": {},
        "enriched_guidelines": [],
        "storage_results": [],
        "errors": [],
        "batch_id": "example_001",
        "processing_start_time": datetime.now(),
        "current_node": "start",
        "retry_count": 0,
        "documents_processed_count": 0,
        "total_guidelines_extracted": 0,
        "memory_threshold_exceeded": False,
    }

    # Execute workflow
    config = {"configurable": {"thread_id": "example_001"}}
    print("Starting workflow execution...")

    final_state = graph.invoke(initial_state, config=config)

    # Display results
    print("\n=== Results ===")
    print(f"Documents processed: {final_state.get('documents_processed_count', 0)}")
    print(f"Guidelines extracted: {final_state.get('total_guidelines_extracted', 0)}")
    print(
        f"Guidelines after deduplication: {len(final_state.get('deduplicated_guidelines', []))}"
    )
    print(f"Guidelines stored: {len(final_state.get('storage_results', []))}")
    print(f"Errors: {len(final_state.get('errors', []))}")

    # Show category distribution
    routed = final_state.get("routed_guidelines", {})
    print("\nCategory Distribution:")
    for category, guidelines in routed.items():
        print(f"  {category}: {len(guidelines)} guidelines")

    return final_state


# ============================================================================
# Example 2: Parallel Processing with Multiple PDFs
# ============================================================================


def example_parallel_processing():
    """Process multiple PDFs in parallel using Send() API."""
    from parallel_processor import create_parallel_workflow
    from state import PDFDocument

    print("\n=== Example 2: Parallel Processing ===\n")

    # Create multiple PDF documents
    pdf_files = [
        "paper1_geo_fundamentals.pdf",
        "paper2_seo_technical.pdf",
        "paper3_citation_optimization.pdf",
    ]

    documents = [
        PDFDocument(
            file_path=f"/path/to/{filename}",  # Update with actual paths
            filename=filename,
            upload_timestamp=datetime.now(),
            file_size_mb=2.0 + i * 0.5,
        )
        for i, filename in enumerate(pdf_files)
    ]

    # Create parallel workflow
    graph = create_parallel_workflow()

    # Prepare state
    initial_state = {
        "documents": documents,
        "batch_results": [],
        "all_guidelines": [],
        "errors": [],
    }

    print(f"Processing {len(documents)} PDFs in parallel...")
    result = graph.invoke(initial_state)

    print("\n=== Parallel Processing Results ===")
    print(f"Total guidelines extracted: {len(result.get('all_guidelines', []))}")
    print(f"Errors encountered: {len(result.get('errors', []))}")

    return result


# ============================================================================
# Example 3: Memory-Aware Batch Processing
# ============================================================================


def example_memory_aware_batch():
    """Process large batch of PDFs with memory management."""
    from parallel_processor import MemoryAwareBatchProcessor, create_parallel_workflow
    from state import PDFDocument

    print("\n=== Example 3: Memory-Aware Batch Processing ===\n")

    # Simulate large batch of PDFs
    documents = [
        PDFDocument(
            file_path=f"/path/to/paper_{i}.pdf",
            filename=f"paper_{i}.pdf",
            upload_timestamp=datetime.now(),
            file_size_mb=3.0,
        )
        for i in range(15)  # 15 PDFs
    ]

    # Create processor with memory limits
    processor = MemoryAwareBatchProcessor(
        max_concurrent_docs=3,  # Process 3 at a time
        max_memory_mb=2048,  # 2GB memory limit
    )

    # Create workflow
    graph = create_parallel_workflow()

    print(f"Processing {len(documents)} PDFs in memory-safe batches...")
    result = processor.process_in_batches(documents, graph)

    print("\n=== Batch Processing Results ===")
    print(f"Total batches: {result['total_batches']}")
    print(f"Guidelines extracted: {len(result['all_guidelines'])}")
    print(f"Errors: {len(result['errors'])}")

    return result


# ============================================================================
# Example 4: Using the REST API
# ============================================================================


def example_api_usage():
    """Demonstrate REST API usage."""
    import requests
    import time

    print("\n=== Example 4: REST API Usage ===\n")

    # Note: Start the API server first with: python api.py
    BASE_URL = "http://localhost:8000"

    # 1. Upload PDF
    print("1. Uploading PDF...")
    with open("/path/to/sample.pdf", "rb") as f:
        response = requests.post(f"{BASE_URL}/upload", files={"file": f})

    if response.status_code == 200:
        upload_result = response.json()
        batch_id = upload_result["batch_id"]
        print(f"   Batch ID: {batch_id}")
        print(f"   Status: {upload_result['status']}")
    else:
        print(f"   Upload failed: {response.text}")
        return

    # 2. Poll for completion
    print("\n2. Monitoring processing status...")
    max_attempts = 30
    for attempt in range(max_attempts):
        response = requests.get(f"{BASE_URL}/batch/{batch_id}")
        if response.status_code == 200:
            status = response.json()
            print(f"   Attempt {attempt + 1}: {status['status']}")

            if status["status"] == "completed":
                print(f"   Documents processed: {status['documents_processed']}")
                print(f"   Guidelines extracted: {status['guidelines_extracted']}")
                print(f"   Guidelines stored: {status['guidelines_stored']}")
                break
            elif status["status"] == "failed":
                print(f"   Processing failed!")
                break

        time.sleep(2)  # Wait 2 seconds between polls

    # 3. Search guidelines
    print("\n3. Searching for guidelines...")
    search_request = {
        "query": "How to optimize content for generative AI search engines?",
        "category": "universal_seo_geo",
        "priority": "high",
        "limit": 5,
    }

    response = requests.post(f"{BASE_URL}/search", json=search_request)

    if response.status_code == 200:
        results = response.json()
        print(f"   Found {results['total_found']} guidelines")

        for i, guideline in enumerate(results["results"][:3], 1):
            print(f"\n   Result {i}:")
            print(f"   Content: {guideline['content'][:100]}...")
            print(f"   Priority: {guideline['priority']}")
            print(f"   Complexity: {guideline['implementation_complexity']}")
            print(f"   Similarity: {guideline['similarity_score']:.3f}")

    # 4. Get specific guideline
    if results["results"]:
        guideline_id = results["results"][0]["guideline_id"]
        print(f"\n4. Fetching specific guideline: {guideline_id}")

        response = requests.get(f"{BASE_URL}/guideline/{guideline_id}")
        if response.status_code == 200:
            guideline = response.json()
            print(f"   Category: {guideline['category']}")
            print(f"   Source: {guideline['source_section']}")
            print(f"   Pages: {guideline['page_numbers']}")

    # 5. List collections
    print("\n5. Listing collections...")
    response = requests.get(f"{BASE_URL}/collections")

    if response.status_code == 200:
        collections = response.json()
        for coll in collections["collections"]:
            if "status" in coll:
                print(f"   {coll['category']}: {coll['status']}")
            else:
                print(
                    f"   {coll['category']}: {coll['points_count']} guidelines in {coll['collection_name']}"
                )


# ============================================================================
# Example 5: Error Handling and Retry
# ============================================================================


def example_error_handling():
    """Demonstrate error handling capabilities."""
    from error_handling import (
        retry_with_backoff,
        CircuitBreaker,
        MemoryMonitor,
        RetryStrategy,
    )

    print("\n=== Example 5: Error Handling ===\n")

    # 1. Retry with exponential backoff
    print("1. Testing retry with exponential backoff...")

    @retry_with_backoff(max_retries=3, strategy=RetryStrategy.EXPONENTIAL)
    def flaky_function():
        import random

        if random.random() < 0.5:
            raise ConnectionError("Temporary network error")
        return "Success!"

    try:
        result = flaky_function()
        print(f"   Result: {result}")
    except Exception as e:
        print(f"   Failed after retries: {e}")

    # 2. Circuit breaker
    print("\n2. Testing circuit breaker...")
    circuit_breaker = CircuitBreaker(failure_threshold=3, timeout_seconds=5.0)

    def unreliable_service():
        import random

        if random.random() < 0.9:  # 90% failure rate
            raise Exception("Service unavailable")
        return "Service response"

    for i in range(5):
        try:
            result = circuit_breaker.call(unreliable_service)
            print(f"   Attempt {i + 1}: {result}")
        except Exception as e:
            print(f"   Attempt {i + 1}: {str(e)}")

    # 3. Memory monitoring
    print("\n3. Testing memory monitoring...")
    monitor = MemoryMonitor(threshold_mb=1024)

    memory_stats = monitor.check_memory()
    print(f"   Current memory: {memory_stats['current_mb']:.2f}MB")
    print(f"   Threshold: {memory_stats['threshold_mb']}MB")
    print(f"   Usage: {memory_stats['percentage']:.1f}%")
    print(f"   Threshold exceeded: {memory_stats['threshold_exceeded']}")


# ============================================================================
# Example 6: Custom Workflow Configuration
# ============================================================================


def example_custom_configuration():
    """Demonstrate custom configuration options."""
    from config import config, ProcessingConfig

    print("\n=== Example 6: Custom Configuration ===\n")

    # Display current configuration
    print("Current Configuration:")
    print(f"  Gemini Model: {config.models.gemini_model}")
    print(f"  Embedding Model: {config.models.embedding_model}")
    print(f"  Similarity Threshold: {config.processing.similarity_threshold}")
    print(f"  Max Concurrent Docs: {config.processing.max_concurrent_documents}")
    print(f"  Max Memory: {config.processing.max_memory_mb}MB")
    print(f"  Max PDF Size: {config.processing.max_pdf_size_mb}MB")

    # Show Qdrant collections
    print("\nQdrant Collections:")
    for category, collection_name in config.qdrant.collections.items():
        print(f"  {category} → {collection_name}")

    # Custom configuration example
    print("\nCreating custom processing config...")
    custom_config = ProcessingConfig(
        similarity_threshold=0.80,  # Lower threshold for more merging
        max_concurrent_documents=5,  # More parallel processing
        max_memory_mb=4096,  # 4GB limit
        max_pdf_size_mb=100,  # Larger PDFs allowed
    )

    print(f"  Custom similarity threshold: {custom_config.similarity_threshold}")
    print(f"  Custom concurrent docs: {custom_config.max_concurrent_documents}")


# ============================================================================
# Main Execution
# ============================================================================


def main():
    """Run all examples."""
    print("=" * 70)
    print("GEO/SEO Knowledge Base - Example Usage")
    print("=" * 70)

    # Choose which examples to run
    examples = {
        "1": ("Single PDF Processing", example_single_pdf),
        "2": ("Parallel Processing", example_parallel_processing),
        "3": ("Memory-Aware Batch", example_memory_aware_batch),
        "4": ("REST API Usage", example_api_usage),
        "5": ("Error Handling", example_error_handling),
        "6": ("Custom Configuration", example_custom_configuration),
    }

    print("\nAvailable Examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")

    print("\n  0. Run all examples")
    print("  q. Quit")

    choice = input("\nSelect example to run: ").strip()

    if choice == "0":
        # Run all examples
        for name, func in examples.values():
            try:
                func()
            except Exception as e:
                print(f"\nError in {name}: {str(e)}")
    elif choice in examples:
        name, func = examples[choice]
        try:
            func()
        except Exception as e:
            print(f"\nError: {str(e)}")
    elif choice.lower() == "q":
        print("Goodbye!")
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
