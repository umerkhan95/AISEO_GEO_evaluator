# Deployment Guide - GEO/SEO Knowledge Base

Complete deployment guide for all environments: local, Docker, Kubernetes, and LangGraph Cloud.

## Prerequisites

- Python 3.11+
- Docker (optional)
- Kubernetes cluster (optional)
- API Keys: Google (Gemini), OpenAI

## Quick Start (Local Development)

### 1. Clone and Setup

```bash
cd AISEO
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
nano .env  # Edit with your API keys
```

Required variables:
```bash
GOOGLE_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### 3. Start Qdrant

```bash
docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage:z \
  --name qdrant \
  qdrant/qdrant
```

Verify:
```bash
curl http://localhost:6333/health
```

### 4. Run the API

```bash
python api.py
```

Visit: http://localhost:8000/docs for interactive API documentation.

### 5. Test the System

```bash
# Upload a PDF
curl -X POST http://localhost:8000/upload \
  -F "file=@sample.pdf"

# Get response with batch_id
# Check status
curl http://localhost:8000/batch/{batch_id}
```

## Production Deployment Options

---

## Option 1: LangGraph Cloud (Recommended)

Best for: Production deployments with zero infrastructure management.

### Step 1: Install LangGraph CLI

```bash
pip install langgraph-cli
```

### Step 2: Initialize LangGraph Project

```bash
# Already configured in langgraph.json
cat langgraph.json
```

```json
{
  "dependencies": ["."],
  "graphs": {
    "geo_seo_kb": "./graph.py:app"
  },
  "env": ".env"
}
```

### Step 3: Test Locally with LangGraph Dev Server

```bash
langgraph dev
```

Opens: http://localhost:8123

### Step 4: Deploy to LangGraph Cloud

```bash
# Login (if not already)
langgraph login

# Deploy
langgraph deploy --name geo-seo-kb
```

### Step 5: Use the Deployed Graph

```python
from langgraph_sdk import get_client

client = get_client(url="https://your-deployment-url")

# Create a thread
thread = client.threads.create()

# Run the graph
run = client.runs.create(
    thread_id=thread["thread_id"],
    assistant_id="geo_seo_kb",
    input={
        "documents": [
            {
                "file_path": "s3://bucket/paper.pdf",
                "filename": "paper.pdf",
                "upload_timestamp": "2024-01-01T00:00:00",
                "file_size_mb": 3.2
            }
        ],
        # ... rest of initial state
    }
)

# Stream results
for event in client.runs.stream(
    thread_id=thread["thread_id"],
    run_id=run["run_id"]
):
    print(event)
```

**Benefits:**
- ✅ Auto-scaling
- ✅ Built-in monitoring via LangSmith
- ✅ Managed checkpointing
- ✅ Zero infrastructure
- ✅ HTTPS/authentication included

---

## Option 2: Docker Deployment

Best for: Self-hosted production with container orchestration.

### Step 1: Create Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create upload directory
RUN mkdir -p /app/uploads

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/ || exit 1

# Run the API
CMD ["python", "api.py"]
```

### Step 2: Create Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_storage:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
    volumes:
      - ./uploads:/app/uploads
    depends_on:
      qdrant:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  qdrant_storage:
```

### Step 3: Build and Run

```bash
# Build
docker-compose build

# Run
docker-compose up -d

# Check logs
docker-compose logs -f

# Scale API instances
docker-compose up -d --scale api=3

# Stop
docker-compose down
```

### Step 4: Add Nginx Load Balancer (Optional)

```nginx
# nginx.conf
upstream api_backend {
    server api:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name your-domain.com;

    client_max_body_size 100M;

    location / {
        proxy_pass http://api_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 300s;
    }
}
```

Add to docker-compose.yml:
```yaml
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/conf.d/default.conf
    depends_on:
      - api
```

---

## Option 3: Kubernetes Deployment

Best for: Large-scale production with auto-scaling requirements.

### Step 1: Create Kubernetes Manifests

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: geo-seo-kb
```

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: geo-seo-config
  namespace: geo-seo-kb
data:
  QDRANT_HOST: "qdrant-service"
  QDRANT_PORT: "6333"
  MAX_CONCURRENT_DOCUMENTS: "5"
  MAX_MEMORY_MB: "4096"
```

```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: api-keys
  namespace: geo-seo-kb
type: Opaque
stringData:
  GOOGLE_API_KEY: "your_gemini_api_key"
  OPENAI_API_KEY: "your_openai_api_key"
```

```yaml
# k8s/qdrant-statefulset.yaml
apiVersion: v1
kind: Service
metadata:
  name: qdrant-service
  namespace: geo-seo-kb
spec:
  clusterIP: None
  selector:
    app: qdrant
  ports:
    - name: http
      port: 6333
    - name: grpc
      port: 6334
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: qdrant
  namespace: geo-seo-kb
spec:
  serviceName: qdrant-service
  replicas: 1
  selector:
    matchLabels:
      app: qdrant
  template:
    metadata:
      labels:
        app: qdrant
    spec:
      containers:
      - name: qdrant
        image: qdrant/qdrant:latest
        ports:
        - containerPort: 6333
          name: http
        - containerPort: 6334
          name: grpc
        volumeMounts:
        - name: qdrant-storage
          mountPath: /qdrant/storage
  volumeClaimTemplates:
  - metadata:
      name: qdrant-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 50Gi
```

```yaml
# k8s/api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: geo-seo-api
  namespace: geo-seo-kb
spec:
  replicas: 3
  selector:
    matchLabels:
      app: geo-seo-api
  template:
    metadata:
      labels:
        app: geo-seo-api
    spec:
      containers:
      - name: api
        image: your-registry/geo-seo-kb:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: geo-seo-config
        - secretRef:
            name: api-keys
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: api-service
  namespace: geo-seo-kb
spec:
  type: LoadBalancer
  selector:
    app: geo-seo-api
  ports:
  - port: 80
    targetPort: 8000
```

```yaml
# k8s/hpa.yaml (Horizontal Pod Autoscaler)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: geo-seo-api-hpa
  namespace: geo-seo-kb
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: geo-seo-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Step 2: Deploy to Kubernetes

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Create secrets (use kubectl create secret instead for security)
kubectl create secret generic api-keys \
  --from-literal=GOOGLE_API_KEY=your_key \
  --from-literal=OPENAI_API_KEY=your_key \
  -n geo-seo-kb

# Apply configurations
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/qdrant-statefulset.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/hpa.yaml

# Check status
kubectl get pods -n geo-seo-kb
kubectl get svc -n geo-seo-kb

# Get external IP
kubectl get svc api-service -n geo-seo-kb
```

### Step 3: Monitor

```bash
# Logs
kubectl logs -f deployment/geo-seo-api -n geo-seo-kb

# Describe
kubectl describe pod <pod-name> -n geo-seo-kb

# Port forward for testing
kubectl port-forward svc/api-service 8000:80 -n geo-seo-kb
```

---

## Environment-Specific Configurations

### Development

```bash
# .env.development
GOOGLE_API_KEY=dev_key
OPENAI_API_KEY=dev_key
QDRANT_HOST=localhost
MAX_CONCURRENT_DOCUMENTS=2
MAX_MEMORY_MB=1024
LANGCHAIN_TRACING_V2=true  # Enable debugging
```

### Staging

```bash
# .env.staging
GOOGLE_API_KEY=staging_key
OPENAI_API_KEY=staging_key
QDRANT_HOST=qdrant-staging.internal
MAX_CONCURRENT_DOCUMENTS=3
MAX_MEMORY_MB=2048
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=geo-seo-kb-staging
```

### Production

```bash
# .env.production
GOOGLE_API_KEY=prod_key
OPENAI_API_KEY=prod_key
QDRANT_HOST=qdrant-prod.internal
MAX_CONCURRENT_DOCUMENTS=5
MAX_MEMORY_MB=4096
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=geo-seo-kb-production
```

---

## Performance Tuning

### Optimize for High Throughput

```python
# config.py adjustments
MAX_CONCURRENT_DOCUMENTS=5  # More parallel processing
SIMILARITY_THRESHOLD=0.80   # More aggressive deduplication
MAX_PDF_SIZE_MB=100         # Allow larger PDFs
```

### Optimize for Low Latency

```python
# config.py adjustments
MAX_CONCURRENT_DOCUMENTS=3  # Less memory pressure
SIMILARITY_THRESHOLD=0.85   # More accurate, faster
MAX_GUIDELINES_PER_DOCUMENT=300  # Limit extraction
```

### Optimize for Cost

```python
# Use cheaper models where possible
GEMINI_MODEL="gemini-1.5-flash"  # Instead of 2.0
EMBEDDING_MODEL="text-embedding-3-small"  # Already optimal
```

---

## Monitoring Setup

### LangSmith Integration

```bash
# Add to .env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=geo-seo-kb
```

### Prometheus Metrics (Optional)

```python
# Add to api.py
from prometheus_client import Counter, Histogram, generate_latest

pdf_uploads = Counter('pdf_uploads_total', 'Total PDF uploads')
processing_time = Histogram('processing_seconds', 'Processing time')

@api.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

---

## Backup and Recovery

### Qdrant Backup

```bash
# Create snapshot
curl -X POST 'http://localhost:6333/collections/geo_seo_universal/snapshots'

# Download snapshot
curl 'http://localhost:6333/collections/geo_seo_universal/snapshots/snapshot-name' \
  -o backup.snapshot

# Restore snapshot
curl -X PUT 'http://localhost:6333/collections/geo_seo_universal/snapshots/upload' \
  -F 'snapshot=@backup.snapshot'
```

### Automated Backup Script

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/qdrant"

for collection in geo_seo_universal geo_seo_industry geo_seo_technical geo_seo_citation geo_seo_metrics; do
    curl -X POST "http://localhost:6333/collections/$collection/snapshots"
    # Wait for snapshot creation
    sleep 5
    # Download latest snapshot
    curl "http://localhost:6333/collections/$collection/snapshots/latest" \
      -o "$BACKUP_DIR/${collection}_${DATE}.snapshot"
done
```

Add to crontab:
```bash
# Daily backups at 2 AM
0 2 * * * /path/to/backup.sh
```

---

## Troubleshooting

### Issue: API not starting

```bash
# Check logs
docker-compose logs api

# Common fixes:
# 1. Verify environment variables
printenv | grep API_KEY

# 2. Check port conflicts
lsof -i :8000

# 3. Verify Qdrant connection
curl http://localhost:6333/health
```

### Issue: Out of memory

```bash
# Check memory usage
docker stats

# Reduce concurrency
export MAX_CONCURRENT_DOCUMENTS=2

# Increase container memory
# In docker-compose.yml:
services:
  api:
    deploy:
      resources:
        limits:
          memory: 4G
```

### Issue: Slow processing

```bash
# Enable profiling
python -m cProfile api.py

# Check Qdrant performance
curl http://localhost:6333/metrics

# Optimize batch size
export MAX_GUIDELINES_PER_DOCUMENT=200
```

---

## Security Hardening

### 1. API Key Rotation

```bash
# Use secrets manager (AWS Secrets Manager, HashiCorp Vault)
# Rotate keys every 90 days
# Never commit .env files
```

### 2. Rate Limiting

```python
# Add to api.py
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@api.post("/upload")
@limiter.limit("10/minute")  # 10 uploads per minute
async def upload_pdf(...):
    ...
```

### 3. HTTPS/TLS

```bash
# Use Let's Encrypt with nginx
certbot --nginx -d your-domain.com
```

### 4. Authentication

```python
# Add to api.py
from fastapi.security import HTTPBearer

security = HTTPBearer()

@api.post("/upload")
async def upload_pdf(token: str = Depends(security)):
    verify_token(token)
    ...
```

---

## Cost Optimization

### Estimate Monthly Costs

| Component | Usage | Cost |
|-----------|-------|------|
| Gemini API | 1000 PDFs/month (10 pages avg) | ~$50 |
| OpenAI Embeddings | 10,000 guidelines | ~$1 |
| Qdrant Cloud (1GB) | 24/7 | ~$95 |
| LangGraph Cloud | 100 runs/day | ~$100 |
| **Total** | | **~$246/month** |

### Cost Reduction Strategies

1. **Batch Processing**: Group PDFs to reduce API calls
2. **Caching**: Cache embeddings for repeated content
3. **Self-hosted Qdrant**: Use Docker instead of cloud (~$246 → $146)
4. **Model Selection**: Use Gemini Flash instead of Pro

---

## Upgrade Path

### v1.0 → v1.1 (Add Clustering)

1. Update `nodes.py` with clustering logic
2. Modify state schema to include `cluster_id`
3. Deploy without downtime using blue-green deployment

### v1.1 → v2.0 (Multi-language)

1. Add language detection node
2. Update embeddings to multilingual model
3. Create language-specific collections
4. Migrate existing data

---

## Support and Maintenance

- Monitor LangSmith for errors
- Review Qdrant metrics weekly
- Update dependencies monthly
- Backup Qdrant data daily
- Test disaster recovery quarterly

For issues: Check logs, review documentation, create GitHub issue.
