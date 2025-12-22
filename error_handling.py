"""
Error handling, retry patterns, and resilience mechanisms for the workflow.

Implements:
- Exponential backoff retry
- Circuit breaker pattern
- Error recovery strategies
- Graceful degradation
"""

import time
import logging
from typing import Callable, Any, Optional
from functools import wraps
from datetime import datetime, timedelta
from enum import Enum

from state import ProcessingError

logger = logging.getLogger(__name__)


# ============================================================================
# Retry Decorator with Exponential Backoff
# ============================================================================


class RetryStrategy(Enum):
    """Retry strategy types."""

    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    CONSTANT = "constant"


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    exceptions: tuple = (Exception,),
):
    """
    Retry decorator with configurable backoff strategy.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        strategy: Retry strategy (exponential, linear, constant)
        exceptions: Tuple of exceptions to catch and retry
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries} retries: {str(e)}"
                        )
                        raise

                    # Calculate delay based on strategy
                    if strategy == RetryStrategy.EXPONENTIAL:
                        delay = min(base_delay * (2**attempt), max_delay)
                    elif strategy == RetryStrategy.LINEAR:
                        delay = min(base_delay * (attempt + 1), max_delay)
                    else:  # CONSTANT
                        delay = base_delay

                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries}). "
                        f"Retrying in {delay:.2f}s... Error: {str(e)}"
                    )

                    time.sleep(delay)

            raise last_exception

        return wrapper

    return decorator


# ============================================================================
# Circuit Breaker Pattern
# ============================================================================


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures.

    Opens after failure_threshold consecutive failures,
    then allows occasional test requests after timeout.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: float = 60.0,
        name: str = "default",
    ):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.name = name

        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitBreakerState.CLOSED

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        """
        if self.state == CircuitBreakerState.OPEN:
            # Check if timeout has passed
            if self.last_failure_time and (
                datetime.now() - self.last_failure_time
            ).total_seconds() > self.timeout_seconds:
                logger.info(f"Circuit breaker '{self.name}': Transitioning to HALF_OPEN")
                self.state = CircuitBreakerState.HALF_OPEN
            else:
                raise Exception(
                    f"Circuit breaker '{self.name}' is OPEN. Service unavailable."
                )

        try:
            result = func(*args, **kwargs)

            # Success - reset failure count
            if self.state == CircuitBreakerState.HALF_OPEN:
                logger.info(f"Circuit breaker '{self.name}': Transitioning to CLOSED")
                self.state = CircuitBreakerState.CLOSED

            self.failure_count = 0
            return result

        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.now()

            logger.warning(
                f"Circuit breaker '{self.name}': Failure #{self.failure_count} - {str(e)}"
            )

            if self.failure_count >= self.failure_threshold:
                logger.error(
                    f"Circuit breaker '{self.name}': Threshold reached, transitioning to OPEN"
                )
                self.state = CircuitBreakerState.OPEN

            raise


# ============================================================================
# Error Recovery Strategies
# ============================================================================


class ErrorRecoveryStrategy:
    """
    Defines recovery strategies for different error types.
    """

    @staticmethod
    def should_retry(error: ProcessingError) -> bool:
        """Determine if error is recoverable."""
        return error.recoverable

    @staticmethod
    def get_fallback_value(error: ProcessingError, node_name: str) -> Any:
        """
        Get fallback value for failed node processing.
        """
        fallback_values = {
            "document_analyzer": {
                "analyzed_documents": [],
                "documents_processed_count": 0,
            },
            "content_extractor": {
                "raw_guidelines": [],
                "total_guidelines_extracted": 0,
            },
            "deduplication": {
                "deduplicated_guidelines": [],
                "merge_log": [],
            },
            "collection_router": {"routed_guidelines": {}},
            "metadata_enricher": {"enriched_guidelines": []},
            "vector_storage": {"storage_results": []},
        }

        return fallback_values.get(node_name, {})


# ============================================================================
# Safe Node Execution Wrapper
# ============================================================================


def safe_node_execution(node_name: str):
    """
    Decorator to wrap node execution with error handling and recovery.
    """

    def decorator(node_func: Callable) -> Callable:
        @wraps(node_func)
        def wrapper(state, *args, **kwargs):
            try:
                # Execute node
                result = node_func(state, *args, **kwargs)
                return result

            except Exception as e:
                logger.error(f"Error in node '{node_name}': {str(e)}", exc_info=True)

                # Create error record
                error = ProcessingError(
                    node_name=node_name,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    timestamp=datetime.now(),
                    recoverable=_is_recoverable_error(e),
                )

                # Get fallback values
                fallback = ErrorRecoveryStrategy.get_fallback_value(error, node_name)

                # Return error state with fallback values
                return {
                    **fallback,
                    "errors": [error],
                    "current_node": node_name,
                }

        return wrapper

    return decorator


def _is_recoverable_error(exception: Exception) -> bool:
    """
    Determine if an error is recoverable.

    Transient errors (network, rate limits) are recoverable.
    Permanent errors (invalid data, missing files) are not.
    """
    recoverable_types = [
        "TimeoutError",
        "ConnectionError",
        "RateLimitError",
        "ServiceUnavailableError",
    ]

    non_recoverable_types = [
        "FileNotFoundError",
        "PermissionError",
        "ValidationError",
        "ValueError",
    ]

    error_type = type(exception).__name__

    if error_type in non_recoverable_types:
        return False
    elif error_type in recoverable_types:
        return True
    else:
        # Default: assume recoverable for unknown errors
        return True


# ============================================================================
# Memory Management
# ============================================================================


class MemoryMonitor:
    """
    Monitor and manage memory usage during processing.
    """

    def __init__(self, threshold_mb: int = 2048):
        self.threshold_mb = threshold_mb

    def check_memory(self) -> dict:
        """
        Check current memory usage.

        Returns:
            Dict with memory stats and threshold status
        """
        import psutil
        import os

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        current_mb = memory_info.rss / 1024 / 1024
        threshold_exceeded = current_mb > self.threshold_mb

        return {
            "current_mb": current_mb,
            "threshold_mb": self.threshold_mb,
            "threshold_exceeded": threshold_exceeded,
            "percentage": (current_mb / self.threshold_mb) * 100,
        }

    def force_garbage_collection(self):
        """Force garbage collection to free memory."""
        import gc

        logger.info("Forcing garbage collection...")
        gc.collect()

        memory_stats = self.check_memory()
        logger.info(f"Memory after GC: {memory_stats['current_mb']:.2f}MB")

    def should_pause_processing(self) -> bool:
        """
        Check if processing should be paused due to memory.
        """
        stats = self.check_memory()

        if stats["percentage"] > 90:
            logger.warning(
                f"Memory usage critical: {stats['current_mb']:.2f}MB ({stats['percentage']:.1f}%)"
            )
            self.force_garbage_collection()
            return True

        return False


# ============================================================================
# Batch Processing with Error Tolerance
# ============================================================================


class FaultTolerantBatchProcessor:
    """
    Process items in batches with error tolerance.

    Continues processing even if individual items fail.
    """

    def __init__(
        self, max_failures_per_batch: int = 3, stop_on_critical_error: bool = True
    ):
        self.max_failures_per_batch = max_failures_per_batch
        self.stop_on_critical_error = stop_on_critical_error

    def process_batch(
        self, items: list, process_func: Callable, batch_name: str = "batch"
    ) -> dict:
        """
        Process a batch of items with error tolerance.

        Args:
            items: List of items to process
            process_func: Function to process each item
            batch_name: Name for logging

        Returns:
            Dict with successful results and errors
        """
        results = []
        errors = []
        failure_count = 0

        for idx, item in enumerate(items):
            try:
                result = process_func(item)
                results.append(result)

            except Exception as e:
                logger.error(
                    f"Error processing item {idx + 1}/{len(items)} in {batch_name}: {str(e)}"
                )

                error = ProcessingError(
                    node_name=batch_name,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    timestamp=datetime.now(),
                    recoverable=_is_recoverable_error(e),
                )

                errors.append(error)
                failure_count += 1

                # Check if we should stop
                if (
                    not error.recoverable and self.stop_on_critical_error
                ) or failure_count >= self.max_failures_per_batch:
                    logger.error(
                        f"Stopping batch processing: {failure_count} failures (max: {self.max_failures_per_batch})"
                    )
                    break

        return {
            "results": results,
            "errors": errors,
            "success_count": len(results),
            "failure_count": len(errors),
            "total_items": len(items),
        }


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    # Example 1: Retry with backoff
    @retry_with_backoff(
        max_retries=3, base_delay=1.0, strategy=RetryStrategy.EXPONENTIAL
    )
    def flaky_api_call():
        """Simulated flaky API call."""
        import random

        if random.random() < 0.7:  # 70% failure rate
            raise ConnectionError("API temporarily unavailable")
        return {"status": "success"}

    # Example 2: Circuit breaker
    circuit_breaker = CircuitBreaker(failure_threshold=3, timeout_seconds=10.0)

    def call_external_service():
        """Simulated external service."""
        import random

        if random.random() < 0.8:  # 80% failure
            raise Exception("Service error")
        return "success"

    # Example 3: Memory monitoring
    monitor = MemoryMonitor(threshold_mb=1024)

    memory_stats = monitor.check_memory()
    print(f"Current memory: {memory_stats['current_mb']:.2f}MB")

    if monitor.should_pause_processing():
        print("Memory threshold exceeded, pausing...")

    # Example 4: Fault-tolerant batch processing
    processor = FaultTolerantBatchProcessor(max_failures_per_batch=2)

    def process_item(item):
        """Simulated item processing."""
        if item % 3 == 0:  # Fail every 3rd item
            raise ValueError(f"Cannot process item {item}")
        return {"item": item, "processed": True}

    batch_result = processor.process_batch(
        items=list(range(10)), process_func=process_item, batch_name="test_batch"
    )

    print(f"\nBatch results:")
    print(f"Success: {batch_result['success_count']}")
    print(f"Failures: {batch_result['failure_count']}")
