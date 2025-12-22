# GEO/SEO Knowledge Base - Architecture Deep Dive

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GEO/SEO Knowledge Base                       │
│                         LangGraph Workflow System                    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                            INPUT LAYER                               │
├─────────────────────────────────────────────────────────────────────┤
│  FastAPI REST API         │  Direct Invocation  │  LangGraph Cloud  │
│  - Upload PDFs            │  - Python SDK       │  - HTTP Streaming │
│  - Search guidelines      │  - Batch processing │  - WebSocket      │
│  - Monitor status         │  - Custom configs   │  - API Gateway    │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      LANGGRAPH WORKFLOW (6 NODES)                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  START                                                                │
│    ↓                                                                  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ NODE 1: Document Analyzer                                     │  │
│  │ ────────────────────────────                                  │  │
│  │ • Model: Gemini Flash 2.0 (Multimodal)                        │  │
│  │ • Input: PDF files (binary)                                   │  │
│  │ • Processing:                                                  │  │
│  │   - Read PDF with vision capabilities                         │  │
│  │   - Extract structure (title, authors, abstract)              │  │
│  │   - Identify sections and content                             │  │
│  │   - Detect themes (SEO, GEO, technical, etc.)                 │  │
│  │   - Find industry mentions                                    │  │
│  │ • Output: DocumentStructure (Pydantic)                        │  │
│  │ • Error Handling: Retry with backoff, file size validation    │  │
│  └───────────────────────────────────────────────────────────────┘  │
│    ↓                                                                  │
│  [Conditional Edge: documents_processed_count > 0?]                  │
│    ↓ (yes)                                                            │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ NODE 2: Content Extractor & Classifier                        │  │
│  │ ─────────────────────────────────────                         │  │
│  │ • Model: Gemini Flash 2.0                                     │  │
│  │ • Input: Analyzed documents with structure                    │  │
│  │ • Processing:                                                  │  │
│  │   - Extract actionable guidelines                             │  │
│  │   - Filter by min length (20 chars)                           │  │
│  │   - Classify into 5 categories:                               │  │
│  │     1. universal_seo_geo (all industries)                     │  │
│  │     2. industry_specific (healthcare, finance, etc.)          │  │
│  │     3. technical (implementation details)                     │  │
│  │     4. citation_optimization (references)                     │  │
│  │     5. metrics (analytics, measurement)                       │  │
│  │   - Assign confidence scores                                  │  │
│  │ • Output: List[Guideline]                                     │  │
│  │ • Error Handling: Graceful degradation, partial results       │  │
│  └───────────────────────────────────────────────────────────────┘  │
│    ↓                                                                  │
│  [Conditional Edge: total_guidelines_extracted > 0?]                 │
│    ↓ (yes)                                                            │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ NODE 3: Deduplication & Merging                               │  │
│  │ ─────────────────────────────────                             │  │
│  │ • Model: OpenAI text-embedding-3-small                        │  │
│  │ • Input: Raw guidelines                                       │  │
│  │ • Processing:                                                  │  │
│  │   - Generate embeddings for all guidelines                    │  │
│  │   - Calculate cosine similarity (numpy)                       │  │
│  │   - Similarity > 85%: Merge guidelines                        │  │
│  │   - Similarity < 85%: Keep separate                           │  │
│  │   - Maintain merge log for traceability                       │  │
│  │ • Algorithm: O(n²) pairwise comparison                        │  │
│  │ • Output: Deduplicated guidelines + merge log                 │  │
│  │ • Memory: Uses numpy arrays for efficiency                    │  │
│  └───────────────────────────────────────────────────────────────┘  │
│    ↓                                                                  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ NODE 4: Collection Router                                     │  │
│  │ ────────────────────────                                      │  │
│  │ • Input: Deduplicated guidelines                              │  │
│  │ • Processing:                                                  │  │
│  │   - Route by category to appropriate collection               │  │
│  │   - Validate category exists                                  │  │
│  │   - Fallback to universal if unknown                          │  │
│  │ • Output: Dict[category → List[Guideline]]                    │  │
│  │ • Collections:                                                 │  │
│  │   - geo_seo_universal                                         │  │
│  │   - geo_seo_industry                                          │  │
│  │   - geo_seo_technical                                         │  │
│  │   - geo_seo_citation                                          │  │
│  │   - geo_seo_metrics                                           │  │
│  └───────────────────────────────────────────────────────────────┘  │
│    ↓                                                                  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ NODE 5: Metadata Enricher                                     │  │
│  │ ────────────────────────                                      │  │
│  │ • Input: Routed guidelines                                    │  │
│  │ • Processing:                                                  │  │
│  │   - Calculate priority (critical/high/medium/low)             │  │
│  │     Based on: confidence + action word frequency              │  │
│  │   - Determine complexity (easy/moderate/complex)              │  │
│  │     Based on: keyword indicators in content                   │  │
│  │   - Find related guidelines (same category, top 5)            │  │
│  │   - Optional: Semantic clustering (cluster_id)                │  │
│  │ • Output: List[EnrichedGuideline]                             │  │
│  │ • Enrichment Rules: Configurable in config.py                 │  │
│  └───────────────────────────────────────────────────────────────┘  │
│    ↓                                                                  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ NODE 6: Vector Storage                                        │  │
│  │ ───────────────────────                                       │  │
│  │ • Database: Qdrant                                            │  │
│  │ • Embedding: OpenAI text-embedding-3-small (1536 dim)         │  │
│  │ • Input: Enriched guidelines                                  │  │
│  │ • Processing:                                                  │  │
│  │   - Generate embeddings (batch mode)                          │  │
│  │   - Create PointStruct with metadata                          │  │
│  │   - Ensure collections exist (auto-create)                    │  │
│  │   - Upsert to Qdrant (by category)                            │  │
│  │ • Vector Config:                                               │  │
│  │   - Distance: COSINE                                          │  │
│  │   - Dimensions: 1536                                          │  │
│  │ • Payload: Full guideline + enriched metadata                 │  │
│  │ • Output: Storage results with vector IDs                     │  │
│  └───────────────────────────────────────────────────────────────┘  │
│    ↓                                                                  │
│  END                                                                  │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                          STORAGE LAYER                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Qdrant Vector Database (5 Collections)                             │
│  ───────────────────────────────────────                             │
│                                                                       │
│  ┌────────────────────┐  ┌────────────────────┐                     │
│  │ geo_seo_universal  │  │ geo_seo_industry   │                     │
│  │ - Universal rules  │  │ - Industry-specific│                     │
│  │ - All contexts     │  │ - Healthcare, etc. │                     │
│  └────────────────────┘  └────────────────────┘                     │
│                                                                       │
│  ┌────────────────────┐  ┌────────────────────┐                     │
│  │ geo_seo_technical  │  │ geo_seo_citation   │                     │
│  │ - Implementation   │  │ - References       │                     │
│  │ - Code/config      │  │ - Source quality   │                     │
│  └────────────────────┘  └────────────────────┘                     │
│                                                                       │
│  ┌────────────────────┐                                              │
│  │ geo_seo_metrics    │                                              │
│  │ - Analytics        │                                              │
│  │ - Measurement      │                                              │
│  └────────────────────┘                                              │
│                                                                       │
│  Each Point Contains:                                                │
│  • Vector: 1536-dim embedding                                        │
│  • Payload:                                                           │
│    - guideline_id, content, category                                 │
│    - source_section, page_numbers                                    │
│    - industries, confidence_score                                    │
│    - priority, implementation_complexity                             │
│    - related_guideline_ids                                           │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

## State Flow Diagram

```
GraphState (TypedDict with Annotated reducers)
──────────────────────────────────────────────

documents: Annotated[list[PDFDocument], add]
    ↓
analyzed_documents: Annotated[list[dict], add]
    ↓
raw_guidelines: Annotated[list[Guideline], add]
    ↓
deduplicated_guidelines: Annotated[list[Guideline], add]
merge_log: Annotated[list[dict], add]
    ↓
routed_guidelines: dict[str, list[Guideline]]
    ↓
enriched_guidelines: Annotated[list[EnrichedGuideline], add]
    ↓
storage_results: Annotated[list[dict], add]

errors: Annotated[list[ProcessingError], add] (accumulated across all nodes)
```

## Parallel Processing Architecture (Send() API)

```
┌──────────────────────────────────────────────────────────────┐
│                  Parallel Processing Mode                     │
└──────────────────────────────────────────────────────────────┘

Input: List[PDFDocument]
   ↓
┌──────────────────────────┐
│   Dispatcher Node        │
│   Creates Send() calls   │
└──────────────────────────┘
   ↓
   ├─→ Send(process_doc, doc1) ─→ [Parallel Execution]
   ├─→ Send(process_doc, doc2) ─→ [Parallel Execution]
   ├─→ Send(process_doc, doc3) ─→ [Parallel Execution]
   └─→ Send(process_doc, docN) ─→ [Parallel Execution]
   ↓
   All results converge
   ↓
┌──────────────────────────┐
│   Aggregator Node        │
│   Combines results       │
└──────────────────────────┘
   ↓
Output: Aggregated guidelines + errors

Benefits:
- 3x faster for 3+ PDFs
- Independent failure handling
- Memory-aware batching
```

## Memory Management Strategy

```
┌────────────────────────────────────────────────────────────┐
│              Memory-Aware Batch Processing                  │
└────────────────────────────────────────────────────────────┘

Configuration:
- max_concurrent_docs: 3
- max_memory_mb: 2048
- chunk_size: 10

Flow:
1. Split documents into chunks of size 10
2. For each chunk:
   a. Check current memory usage
   b. If > 80% threshold: Force garbage collection
   c. Process chunk with max 3 concurrent docs
   d. Wait for chunk completion
   e. Clear intermediate results
3. Aggregate all chunk results

Memory Profile:
- Base system: ~500MB
- Per PDF (concurrent): ~200MB
- Peak for 3 PDFs: ~1.1GB
- Safe under 2GB limit
```

## Error Handling Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                  Error Handling Layers                        │
└──────────────────────────────────────────────────────────────┘

Layer 1: Retry with Exponential Backoff
────────────────────────────────────────
- Transient errors (network, rate limits)
- Max retries: 3
- Base delay: 1s
- Max delay: 60s
- Strategy: Exponential (2^attempt)

Layer 2: Circuit Breaker
─────────────────────────
- Prevents cascading failures
- Failure threshold: 5 consecutive failures
- Timeout: 60 seconds
- States: CLOSED → OPEN → HALF_OPEN → CLOSED

Layer 3: Safe Node Execution
─────────────────────────────
- Wraps each node with try-catch
- Returns fallback values on error
- Accumulates errors in state
- Continues workflow with partial results

Layer 4: Fault-Tolerant Batch Processing
─────────────────────────────────────────
- Continues on individual item failures
- Max failures per batch: 3
- Stops on critical errors
- Returns success + failure counts

Error Classification:
- Recoverable: Network, timeouts, rate limits
- Non-recoverable: Invalid files, permissions, validation errors
```

## Data Models

```
┌────────────────────────────────────────────────────────────┐
│                    Core Data Models                         │
└────────────────────────────────────────────────────────────┘

PDFDocument
───────────
- file_path: str
- filename: str
- upload_timestamp: datetime
- file_size_mb: float

DocumentStructure
─────────────────
- title: str
- authors: list[str]
- abstract: str
- sections: list[dict]
- themes: list[str]
- industries: list[str]
- total_pages: int
- extraction_confidence: float

Guideline
─────────
- guideline_id: str (hash-based)
- content: str
- source_section: str
- page_numbers: list[int]
- category: Literal[5 categories]
- industries: list[str]
- extraction_confidence: float

EnrichedGuideline
─────────────────
- guideline: Guideline
- confidence_score: float
- priority: Literal["critical", "high", "medium", "low"]
- implementation_complexity: Literal["easy", "moderate", "complex"]
- related_guideline_ids: list[str]
- semantic_cluster_id: Optional[str]

ProcessingError
───────────────
- node_name: str
- error_type: str
- error_message: str
- timestamp: datetime
- recoverable: bool
```

## Deployment Patterns

### Pattern 1: LangGraph Cloud (Recommended)

```
Local Development
    ↓
  git push
    ↓
LangGraph Cloud Auto-Deploy
    ↓
Available Endpoints:
- POST /runs (start workflow)
- GET /runs/{run_id} (check status)
- POST /runs/{run_id}/stream (streaming)
- POST /threads/{thread_id}/state (get state)

Benefits:
✓ Auto-scaling
✓ Built-in monitoring
✓ Managed checkpointing
✓ Zero infrastructure
```

### Pattern 2: Docker Deployment

```
Dockerfile
    ↓
docker build -t geo-seo-kb .
    ↓
docker run -p 8000:8000 geo-seo-kb
    ↓
Load Balancer → Multiple Containers

Components:
- App Container: FastAPI + LangGraph
- Qdrant Container: Vector DB
- Redis Container: Batch tracking (optional)
```

### Pattern 3: Kubernetes Deployment

```
Kubernetes Cluster
    ↓
Deployments:
- geo-seo-api (FastAPI, replicas=3)
- qdrant (StatefulSet, persistent volume)

Services:
- api-service (LoadBalancer)
- qdrant-service (ClusterIP)

ConfigMaps: API keys, configs
Secrets: Sensitive credentials
```

## Performance Optimization

### Optimization 1: Embedding Batching

```python
# Instead of:
for guideline in guidelines:
    embedding = embeddings_model.embed_query(guideline.content)

# Use batching:
texts = [g.content for g in guidelines]
embeddings = embeddings_model.embed_documents(texts)  # Single API call
```

### Optimization 2: Parallel Document Processing

```python
# Sequential: 30s × 10 PDFs = 300s
for pdf in pdfs:
    process(pdf)

# Parallel (3 concurrent): 30s × (10/3) = ~100s
with Send() API:
    process_all_parallel(pdfs, max_concurrent=3)
```

### Optimization 3: Smart Deduplication

```python
# Use numpy for vectorized similarity
import numpy as np

embeddings_array = np.array(embeddings)
# Cosine similarity matrix
similarity_matrix = np.dot(embeddings_array, embeddings_array.T)

# 10x faster than nested loops for 1000+ guidelines
```

## Security Considerations

1. **API Key Management**
   - Never commit .env files
   - Use environment variables
   - Rotate keys regularly

2. **PDF Validation**
   - File size limits (50MB default)
   - MIME type checking
   - Virus scanning (recommended)

3. **Rate Limiting**
   - Implement on API endpoints
   - Prevent abuse
   - Queue management

4. **Data Privacy**
   - Sanitize uploaded PDFs
   - Anonymize sensitive content
   - GDPR compliance

## Monitoring & Observability

### LangSmith Integration

```python
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your_key
export LANGCHAIN_PROJECT=geo-seo-kb

Provides:
- Trace visualization
- Latency metrics
- Error tracking
- Token usage monitoring
```

### Custom Metrics

```python
Key Metrics to Track:
- PDFs processed per hour
- Average processing time per PDF
- Guidelines extracted per document
- Deduplication merge rate
- Vector storage success rate
- Error rate by node
- Memory usage peaks
```

## Scaling Recommendations

### Small Scale (< 100 PDFs/day)
- Single instance deployment
- Local Qdrant
- In-memory batch tracking

### Medium Scale (100-1000 PDFs/day)
- 3-5 API replicas
- Managed Qdrant (cloud)
- Redis for batch tracking
- Load balancer

### Large Scale (> 1000 PDFs/day)
- Auto-scaling (5-20 replicas)
- Qdrant cluster
- PostgreSQL checkpointer
- CDN for static assets
- Message queue (RabbitMQ/Kafka)
- Distributed caching (Redis cluster)

## Future Architecture Enhancements

1. **Semantic Clustering**
   - K-means clustering of guidelines
   - Topic modeling with LDA
   - Hierarchical organization

2. **Multi-Language Support**
   - Language detection
   - Translation pipeline
   - Language-specific embeddings

3. **Real-Time Updates**
   - WebSocket streaming
   - Server-sent events
   - Live progress updates

4. **Advanced Analytics**
   - Citation graph analysis
   - Guideline effectiveness scoring
   - Industry trend detection

5. **Automated Quality Control**
   - Confidence thresholding
   - Human-in-the-loop validation
   - Active learning for classification
