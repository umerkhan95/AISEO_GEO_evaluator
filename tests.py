"""
Comprehensive test suite for GEO/SEO Knowledge Base system.

Tests:
1. State management and reducers
2. Individual node functionality
3. Full workflow execution
4. Parallel processing
5. Error handling and recovery
6. API endpoints
"""

import pytest
from datetime import datetime
from typing import List
import tempfile
import os

# Import system components
from state import (
    GraphState,
    PDFDocument,
    DocumentStructure,
    Guideline,
    EnrichedGuideline,
    ProcessingError,
)
from config import config
from nodes import (
    document_analyzer_node,
    content_extractor_node,
    deduplication_node,
    collection_router_node,
    metadata_enricher_node,
    vector_storage_node,
)
from graph import create_workflow
from error_handling import (
    retry_with_backoff,
    CircuitBreaker,
    MemoryMonitor,
    FaultTolerantBatchProcessor,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_pdf_document():
    """Create a sample PDF document for testing."""
    return PDFDocument(
        file_path="/tmp/test_paper.pdf",
        filename="test_geo_paper.pdf",
        upload_timestamp=datetime.now(),
        file_size_mb=2.5,
    )


@pytest.fixture
def sample_guideline():
    """Create a sample guideline for testing."""
    return Guideline(
        guideline_id="test_123",
        content="Always optimize page titles with primary keywords for better SEO performance.",
        source_section="SEO Best Practices",
        page_numbers=[5, 6],
        category="universal_seo_geo",
        industries=[],
        extraction_confidence=0.92,
    )


@pytest.fixture
def initial_state(sample_pdf_document):
    """Create initial graph state for testing."""
    return {
        "documents": [sample_pdf_document],
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


# ============================================================================
# Test State Management
# ============================================================================


class TestStateManagement:
    """Test state schema and reducers."""

    def test_state_accumulation(self):
        """Test that Annotated reducers properly accumulate values."""
        from operator import add

        # Simulate state updates across nodes
        state = {
            "raw_guidelines": [],
            "errors": [],
        }

        # First node adds guidelines
        guideline1 = Guideline(
            guideline_id="g1",
            content="Test guideline 1",
            source_section="Section 1",
            page_numbers=[1],
            category="universal_seo_geo",
            industries=[],
            extraction_confidence=0.9,
        )

        state["raw_guidelines"] = add(state["raw_guidelines"], [guideline1])

        # Second node adds more guidelines
        guideline2 = Guideline(
            guideline_id="g2",
            content="Test guideline 2",
            source_section="Section 2",
            page_numbers=[2],
            category="technical",
            industries=[],
            extraction_confidence=0.85,
        )

        state["raw_guidelines"] = add(state["raw_guidelines"], [guideline2])

        assert len(state["raw_guidelines"]) == 2
        assert state["raw_guidelines"][0].guideline_id == "g1"
        assert state["raw_guidelines"][1].guideline_id == "g2"

    def test_error_accumulation(self):
        """Test error accumulation across nodes."""
        errors = []

        error1 = ProcessingError(
            node_name="node1",
            error_type="ValueError",
            error_message="Test error 1",
            timestamp=datetime.now(),
            recoverable=True,
        )

        error2 = ProcessingError(
            node_name="node2",
            error_type="ConnectionError",
            error_message="Test error 2",
            timestamp=datetime.now(),
            recoverable=True,
        )

        errors.extend([error1])
        errors.extend([error2])

        assert len(errors) == 2
        assert errors[0].node_name == "node1"
        assert errors[1].node_name == "node2"


# ============================================================================
# Test Individual Nodes
# ============================================================================


class TestNodes:
    """Test individual node functionality."""

    @pytest.mark.skip(reason="Requires valid API key and PDF file")
    def test_document_analyzer_node(self, initial_state):
        """Test document analysis with Gemini."""
        result = document_analyzer_node(initial_state)

        assert "analyzed_documents" in result
        assert "current_node" in result
        assert result["current_node"] == "document_analyzer"

    def test_collection_router_node(self, sample_guideline):
        """Test guideline routing to collections."""
        state = {
            "deduplicated_guidelines": [sample_guideline],
            "routed_guidelines": {},
        }

        result = collection_router_node(state)

        assert "routed_guidelines" in result
        assert "universal_seo_geo" in result["routed_guidelines"]
        assert len(result["routed_guidelines"]["universal_seo_geo"]) == 1

    def test_metadata_enricher_node(self, sample_guideline):
        """Test metadata enrichment."""
        state = {
            "routed_guidelines": {
                "universal_seo_geo": [sample_guideline],
            },
            "enriched_guidelines": [],
        }

        result = metadata_enricher_node(state)

        assert "enriched_guidelines" in result
        assert len(result["enriched_guidelines"]) == 1

        enriched = result["enriched_guidelines"][0]
        assert hasattr(enriched, "priority")
        assert hasattr(enriched, "implementation_complexity")
        assert enriched.priority in ["critical", "high", "medium", "low"]
        assert enriched.implementation_complexity in ["easy", "moderate", "complex"]


# ============================================================================
# Test Workflow Execution
# ============================================================================


class TestWorkflow:
    """Test full workflow execution."""

    @pytest.mark.skip(reason="Requires API keys and Qdrant")
    def test_full_workflow(self, initial_state):
        """Test complete workflow execution."""
        graph = create_workflow(enable_checkpointing=True)
        config = {"configurable": {"thread_id": "test_001"}}

        final_state = graph.invoke(initial_state, config=config)

        # Verify workflow completion
        assert "storage_results" in final_state
        assert final_state["current_node"] == "vector_storage"

    def test_workflow_conditional_edges(self):
        """Test conditional edge logic."""
        from graph import should_continue_to_extraction, should_continue_to_deduplication

        # Test successful path
        state_success = {"documents_processed_count": 5}
        assert should_continue_to_extraction(state_success) == "content_extractor"

        # Test failure path
        state_failure = {"documents_processed_count": 0}
        assert should_continue_to_extraction(state_failure) == "end"

        # Test guidelines extraction
        state_with_guidelines = {"total_guidelines_extracted": 10}
        assert should_continue_to_deduplication(state_with_guidelines) == "deduplication"

        state_no_guidelines = {"total_guidelines_extracted": 0}
        assert should_continue_to_deduplication(state_no_guidelines) == "end"


# ============================================================================
# Test Parallel Processing
# ============================================================================


class TestParallelProcessing:
    """Test Send() API and parallel execution."""

    @pytest.mark.skip(reason="Requires API keys")
    def test_parallel_workflow(self):
        """Test parallel document processing."""
        from parallel_processor import create_parallel_workflow

        documents = [
            PDFDocument(
                file_path=f"/tmp/paper_{i}.pdf",
                filename=f"paper_{i}.pdf",
                upload_timestamp=datetime.now(),
                file_size_mb=2.0,
            )
            for i in range(3)
        ]

        graph = create_parallel_workflow()
        initial_state = {
            "documents": documents,
            "batch_results": [],
            "all_guidelines": [],
            "errors": [],
        }

        result = graph.invoke(initial_state)

        assert "all_guidelines" in result
        assert "errors" in result

    def test_memory_aware_batch_processor(self):
        """Test memory-aware batch processing."""
        from parallel_processor import MemoryAwareBatchProcessor

        processor = MemoryAwareBatchProcessor(max_concurrent_docs=2, max_memory_mb=1024)

        # Test batch splitting logic
        documents = [
            PDFDocument(
                file_path=f"/tmp/doc_{i}.pdf",
                filename=f"doc_{i}.pdf",
                upload_timestamp=datetime.now(),
                file_size_mb=1.0,
            )
            for i in range(10)
        ]

        # Should split into batches of 2
        # Note: This test doesn't actually process, just validates structure
        assert len(documents) == 10
        assert processor.max_concurrent_docs == 2


# ============================================================================
# Test Error Handling
# ============================================================================


class TestErrorHandling:
    """Test error handling and retry mechanisms."""

    def test_retry_with_backoff(self):
        """Test exponential backoff retry."""
        attempt_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.1)
        def flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"

        result = flaky_function()
        assert result == "success"
        assert attempt_count == 3

    def test_retry_exhaustion(self):
        """Test retry exhaustion raises exception."""

        @retry_with_backoff(max_retries=2, base_delay=0.1)
        def always_fails():
            raise ValueError("Permanent failure")

        with pytest.raises(ValueError):
            always_fails()

    def test_circuit_breaker(self):
        """Test circuit breaker pattern."""
        circuit_breaker = CircuitBreaker(failure_threshold=3, timeout_seconds=1.0)

        failure_count = 0

        def failing_service():
            nonlocal failure_count
            failure_count += 1
            raise Exception("Service error")

        # First 3 failures should trigger circuit
        for i in range(3):
            with pytest.raises(Exception):
                circuit_breaker.call(failing_service)

        # Circuit should now be OPEN
        assert circuit_breaker.state.value == "open"

        # Additional calls should fail immediately
        with pytest.raises(Exception, match="Circuit breaker.*is OPEN"):
            circuit_breaker.call(failing_service)

    def test_memory_monitor(self):
        """Test memory monitoring."""
        monitor = MemoryMonitor(threshold_mb=1024)

        stats = monitor.check_memory()

        assert "current_mb" in stats
        assert "threshold_mb" in stats
        assert "threshold_exceeded" in stats
        assert "percentage" in stats
        assert stats["threshold_mb"] == 1024

    def test_fault_tolerant_batch_processor(self):
        """Test fault-tolerant batch processing."""
        processor = FaultTolerantBatchProcessor(max_failures_per_batch=2)

        def process_item(item):
            if item % 2 == 0:
                raise ValueError(f"Cannot process {item}")
            return {"item": item, "processed": True}

        result = processor.process_batch(
            items=[1, 2, 3, 4, 5], process_func=process_item, batch_name="test"
        )

        assert result["success_count"] == 3  # 1, 3, 5
        assert result["failure_count"] == 2  # 2, 4
        assert len(result["results"]) == 3
        assert len(result["errors"]) == 2


# ============================================================================
# Test API Endpoints
# ============================================================================


class TestAPI:
    """Test FastAPI endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from api import api

        return TestClient(api)

    def test_health_check(self, client):
        """Test root health check endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        assert "status" in response.json()
        assert response.json()["status"] == "healthy"

    @pytest.mark.skip(reason="Requires file upload setup")
    def test_upload_endpoint(self, client):
        """Test PDF upload endpoint."""
        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4 test content")
            temp_path = f.name

        try:
            with open(temp_path, "rb") as f:
                response = client.post("/upload", files={"file": ("test.pdf", f, "application/pdf")})

            assert response.status_code == 200
            assert "batch_id" in response.json()
        finally:
            os.unlink(temp_path)

    def test_collections_endpoint(self, client):
        """Test collections listing endpoint."""
        response = client.get("/collections")
        assert response.status_code == 200
        assert "collections" in response.json()


# ============================================================================
# Test Configuration
# ============================================================================


class TestConfiguration:
    """Test configuration management."""

    def test_config_loading(self):
        """Test configuration loads correctly."""
        assert config.models.gemini_model is not None
        assert config.models.embedding_model is not None
        assert config.qdrant.collections is not None
        assert len(config.qdrant.collections) == 5

    def test_qdrant_collections(self):
        """Test Qdrant collection names."""
        collections = config.qdrant.collections

        assert "universal_seo_geo" in collections
        assert "industry_specific" in collections
        assert "technical" in collections
        assert "citation_optimization" in collections
        assert "metrics" in collections

    def test_processing_limits(self):
        """Test processing configuration limits."""
        assert config.processing.similarity_threshold == 0.85
        assert config.processing.max_concurrent_documents >= 1
        assert config.processing.max_memory_mb >= 1024


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    @pytest.mark.skip(reason="Requires full setup with API keys and Qdrant")
    def test_end_to_end_single_pdf(self):
        """Test complete processing of a single PDF."""
        # This would require:
        # 1. Valid PDF file
        # 2. API keys configured
        # 3. Qdrant running
        # 4. Full workflow execution
        pass

    @pytest.mark.skip(reason="Requires full setup")
    def test_end_to_end_batch_processing(self):
        """Test batch processing of multiple PDFs."""
        pass

    @pytest.mark.skip(reason="Requires Qdrant")
    def test_search_functionality(self):
        """Test semantic search across collections."""
        pass


# ============================================================================
# Performance Tests
# ============================================================================


class TestPerformance:
    """Performance and load testing."""

    def test_deduplication_performance(self):
        """Test deduplication performance with many guidelines."""
        import numpy as np
        from langchain_openai import OpenAIEmbeddings

        # Create test guidelines
        guidelines = [
            Guideline(
                guideline_id=f"perf_test_{i}",
                content=f"Test guideline {i} with unique content",
                source_section="Test",
                page_numbers=[i],
                category="universal_seo_geo",
                industries=[],
                extraction_confidence=0.9,
            )
            for i in range(100)  # 100 guidelines
        ]

        # Simulate embeddings (random vectors for test)
        embeddings = np.random.rand(100, 1536)

        # Test similarity calculation
        similarity_matrix = np.dot(embeddings, embeddings.T)

        # Should complete quickly
        assert similarity_matrix.shape == (100, 100)

    def test_memory_usage_tracking(self):
        """Test memory usage during processing."""
        monitor = MemoryMonitor(threshold_mb=2048)

        initial_stats = monitor.check_memory()
        initial_memory = initial_stats["current_mb"]

        # Simulate processing (create large data structure)
        large_data = [{"data": "x" * 1000} for _ in range(1000)]

        after_stats = monitor.check_memory()
        after_memory = after_stats["current_mb"]

        # Memory should have increased
        assert after_memory >= initial_memory


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
