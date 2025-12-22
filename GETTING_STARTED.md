# Getting Started with GEO/SEO Knowledge Base

A step-by-step guide to get your LangGraph-powered knowledge base up and running in under 15 minutes.

## What You'll Build

A production-ready system that:
1. Ingests PDF papers about GEO/SEO
2. Extracts actionable guidelines
3. Deduplicates using semantic similarity
4. Stores in queryable vector database
5. Provides REST API for search

## Prerequisites

Before you begin, ensure you have:

- [x] Python 3.11+ installed
- [x] pip or conda package manager
- [x] Docker (for Qdrant) or access to Qdrant Cloud
- [x] Google API key (for Gemini)
- [x] OpenAI API key (for embeddings)

### Get API Keys

1. **Google (Gemini)**
   - Visit: https://makersuite.google.com/app/apikey
   - Create new API key
   - Copy the key

2. **OpenAI**
   - Visit: https://platform.openai.com/api-keys
   - Create new secret key
   - Copy the key

## Step 1: Install Dependencies (2 minutes)

```bash
# Navigate to project directory
cd AISEO

# Install Python packages
pip install -r requirements.txt

# Verify installation
python -c "import langgraph; print('LangGraph version:', langgraph.__version__)"
```

Expected output:
```
LangGraph version: 0.2.x
```

## Step 2: Configure Environment (3 minutes)

```bash
# Copy environment template
cp .env.example .env

# Edit with your API keys
nano .env
# or
code .env
# or
vim .env
```

Add your keys:
```bash
# Required
GOOGLE_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Optional (defaults work for local development)
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

Save and close the file.

## Step 3: Start Qdrant Vector Database (2 minutes)

### Option A: Docker (Recommended)

```bash
# Start Qdrant
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage:z \
  qdrant/qdrant

# Verify it's running
curl http://localhost:6333/health
```

Expected output:
```json
{"title":"qdrant - vector search engine","version":"1.x.x"}
```

### Option B: Qdrant Cloud

1. Sign up at https://cloud.qdrant.io
2. Create a cluster
3. Get connection details
4. Update .env:
```bash
QDRANT_HOST=your-cluster.qdrant.io
QDRANT_PORT=6333
QDRANT_API_KEY=your_api_key
```

## Step 4: Test the System (5 minutes)

### Quick Test: Run Examples

```bash
# Run interactive examples
python example_usage.py
```

You'll see a menu:
```
Available Examples:
  1. Single PDF Processing
  2. Parallel Processing
  3. Memory-Aware Batch
  4. REST API Usage
  5. Error Handling
  6. Custom Configuration

Select example to run:
```

Choose `6` to test configuration (no API calls required).

### Full Test: Start API Server

```bash
# Start the FastAPI server
python api.py
```

Expected output:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## Step 5: Upload Your First PDF (3 minutes)

### Via Web Interface

1. Open browser: http://localhost:8000/docs
2. Click on `POST /upload`
3. Click "Try it out"
4. Upload a PDF file
5. Click "Execute"

You'll get a response:
```json
{
  "batch_id": "abc-123-def",
  "filename": "your_paper.pdf",
  "file_size_mb": 2.5,
  "status": "queued",
  "message": "PDF uploaded successfully. Processing started in background."
}
```

### Via Command Line

```bash
# Upload PDF
curl -X POST http://localhost:8000/upload \
  -F "file=@/path/to/your/paper.pdf"

# Save the batch_id from response
```

## Step 6: Monitor Processing

### Check Status

```bash
# Replace {batch_id} with your actual batch ID
curl http://localhost:8000/batch/{batch_id}
```

Response:
```json
{
  "batch_id": "abc-123-def",
  "status": "processing",
  "documents_processed": 1,
  "guidelines_extracted": 45,
  "guidelines_stored": 42,
  "errors": 0
}
```

Status values:
- `queued`: Waiting to start
- `processing`: Currently running
- `completed`: Successfully finished
- `failed`: Error occurred

### Watch Logs

In the terminal running `python api.py`, you'll see:
```
INFO:     === Node 1: Document Analyzer ===
INFO:     Analyzing document: your_paper.pdf
INFO:     === Node 2: Content Extractor ===
INFO:     Extracted 45 guidelines
INFO:     === Node 3: Deduplication ===
INFO:     45 -> 42 guidelines after deduplication
...
```

## Step 7: Search Guidelines

### Via Web Interface

1. Go to: http://localhost:8000/docs
2. Click `POST /search`
3. Try it out with:
```json
{
  "query": "How to optimize content for AI search engines?",
  "limit": 5
}
```

### Via Command Line

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "optimize content for generative AI",
    "category": "universal_seo_geo",
    "priority": "high",
    "limit": 10
  }'
```

Response:
```json
{
  "query": "optimize content for generative AI",
  "results": [
    {
      "guideline_id": "abc123",
      "content": "Use structured data and schema markup...",
      "category": "universal_seo_geo",
      "priority": "high",
      "implementation_complexity": "moderate",
      "similarity_score": 0.89
    }
  ],
  "total_found": 10
}
```

## Congratulations!

You now have a working GEO/SEO Knowledge Base system. Here's what you can do next:

## Next Steps

### 1. Process More PDFs

```bash
# Batch upload (up to 10 PDFs)
curl -X POST http://localhost:8000/upload-batch \
  -F "files=@paper1.pdf" \
  -F "files=@paper2.pdf" \
  -F "files=@paper3.pdf"
```

### 2. Explore Collections

```bash
# List all collections
curl http://localhost:8000/collections
```

### 3. Get Specific Guidelines

```bash
# Get guideline by ID
curl http://localhost:8000/guideline/{guideline_id}
```

### 4. Filter Searches

```bash
# Search with filters
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "citation optimization",
    "category": "citation_optimization",
    "priority": "critical",
    "complexity": "easy",
    "industries": ["healthcare"],
    "limit": 20
  }'
```

## Common Issues & Solutions

### Issue 1: "Module not found" Error

**Solution:**
```bash
# Ensure you're in the AISEO directory
cd AISEO

# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

### Issue 2: Qdrant Connection Failed

**Solution:**
```bash
# Check if Qdrant is running
docker ps | grep qdrant

# If not running, start it
docker start qdrant

# Verify health
curl http://localhost:6333/health
```

### Issue 3: API Keys Not Working

**Solution:**
```bash
# Verify environment variables are loaded
python -c "from config import config; print('Gemini key:', config.models.gemini_api_key[:10])"

# Should show first 10 chars of your key
# If empty, check .env file exists and has correct keys
```

### Issue 4: Out of Memory

**Solution:**
```bash
# Reduce concurrent documents in .env
MAX_CONCURRENT_DOCUMENTS=2
MAX_MEMORY_MB=1024

# Restart the API
```

### Issue 5: PDF Processing Fails

**Solution:**
```bash
# Check PDF is valid
file your_paper.pdf

# Should output: "PDF document..."

# Check file size
ls -lh your_paper.pdf

# Must be < 50MB (default limit)
```

## Understanding the Workflow

When you upload a PDF, here's what happens:

1. **Document Analyzer** (30-60s)
   - Gemini reads the PDF
   - Extracts structure, themes, industries

2. **Content Extractor** (20-40s)
   - Identifies actionable guidelines
   - Classifies into 5 categories

3. **Deduplication** (5-10s)
   - Compares guidelines semantically
   - Merges duplicates (>85% similarity)

4. **Collection Router** (<1s)
   - Routes to appropriate Qdrant collection

5. **Metadata Enricher** (2-5s)
   - Adds priority, complexity scores
   - Links related guidelines

6. **Vector Storage** (5-10s)
   - Generates embeddings
   - Stores in Qdrant

**Total Time:** ~60-120 seconds per PDF

## Performance Tips

### Faster Processing

```bash
# Increase concurrent documents (if you have RAM)
MAX_CONCURRENT_DOCUMENTS=5
MAX_MEMORY_MB=4096
```

### More Accurate Deduplication

```bash
# Increase similarity threshold (less merging)
SIMILARITY_THRESHOLD=0.90
```

### Extract More Guidelines

```bash
# Increase per-document limit
MAX_GUIDELINES_PER_DOCUMENT=500
```

## Development Workflow

### 1. Local Development

```bash
# Enable debug mode
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your_langsmith_key

# Run with auto-reload
uvicorn api:api --reload --host 0.0.0.0 --port 8000
```

### 2. Run Tests

```bash
# Run all tests
python tests.py

# Run specific test
pytest tests.py::TestNodes::test_collection_router_node -v

# Run with coverage
pytest tests.py --cov=. --cov-report=html
```

### 3. Check Graph Visualization

```python
from graph import visualize_graph

visualize_graph("my_graph.png")
```

### 4. Direct Workflow Invocation

```python
from graph import create_workflow
from state import PDFDocument
from datetime import datetime

# Create workflow with checkpointing
graph = create_workflow(enable_checkpointing=True)

# Create input
pdf = PDFDocument(
    file_path="./sample.pdf",
    filename="sample.pdf",
    upload_timestamp=datetime.now(),
    file_size_mb=2.5
)

# Run workflow
result = graph.invoke({
    "documents": [pdf],
    # ... rest of state
})

print(f"Extracted: {result['total_guidelines_extracted']} guidelines")
```

## Deployment

Once you're ready for production:

### Option 1: LangGraph Cloud (Easiest)

```bash
# Install CLI
pip install langgraph-cli

# Deploy
langgraph deploy --name geo-seo-kb
```

### Option 2: Docker

```bash
# Build
docker-compose build

# Run
docker-compose up -d

# Scale
docker-compose up -d --scale api=3
```

### Option 3: Kubernetes

See `DEPLOYMENT.md` for complete Kubernetes manifests.

## Learning Resources

### Understanding the Code

1. **Start with state.py**
   - See how data flows through the system
   - Understand TypedDict and Annotated reducers

2. **Read nodes.py**
   - See how each processing step works
   - Learn structured output with Pydantic

3. **Study graph.py**
   - Understand workflow composition
   - See conditional edges in action

4. **Explore parallel_processor.py**
   - Learn Send() API for parallelization
   - Understand memory-aware batching

### Documentation

- **README.md**: High-level overview
- **ARCHITECTURE.md**: Deep technical details
- **DEPLOYMENT.md**: Production deployment
- **SUMMARY.md**: Implementation summary

### Examples

Run `python example_usage.py` and select:
- Example 1: See basic workflow
- Example 2: Learn parallel processing
- Example 3: Understand memory management
- Example 4: Master the API
- Example 5: Handle errors properly
- Example 6: Customize configuration

## Getting Help

### Check Logs

```bash
# API logs (if running in terminal)
# Just watch the terminal output

# Docker logs
docker logs -f qdrant

# Check for errors
grep ERROR api.log
```

### Enable Debug Logging

```python
# Add to api.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

### LangSmith Debugging

```bash
# Enable tracing
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your_key

# View traces at: https://smith.langchain.com
```

## What's Next?

You now have a complete, production-ready system. Consider:

1. **Adding more PDFs**: Build your knowledge base
2. **Customizing categories**: Edit `CATEGORY_DESCRIPTIONS` in config.py
3. **Tuning deduplication**: Adjust `SIMILARITY_THRESHOLD`
4. **Deploying to production**: Use LangGraph Cloud or Docker
5. **Integrating with apps**: Use the REST API in your applications

## Success Checklist

- [x] Python 3.11+ installed
- [x] Dependencies installed (`pip install -r requirements.txt`)
- [x] API keys configured (.env file)
- [x] Qdrant running (Docker or cloud)
- [x] API server started (`python api.py`)
- [x] First PDF uploaded
- [x] Processing completed
- [x] Guidelines searchable

If all checked, you're ready to build your knowledge base!

---

**Need more help?**
- Read: ARCHITECTURE.md for technical details
- Read: DEPLOYMENT.md for production deployment
- Read: README.md for feature overview
- Run: `python example_usage.py` for interactive examples
- Check: http://localhost:8000/docs for API documentation
