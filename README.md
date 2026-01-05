# AISEO - GEO Website Optimizer & Knowledge Base

**Transform any website content into AI-citation-ready content using Generative Engine Optimization (GEO).**

[![Deploy Backend](https://img.shields.io/badge/Backend-Fly.io-purple)](https://fly.io)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![React 18](https://img.shields.io/badge/React-18-61DAFB.svg)](https://reactjs.org)
[![LangChain](https://img.shields.io/badge/LangChain-Powered-green.svg)](https://langchain.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What is GEO?

**Generative Engine Optimization (GEO)** is the practice of optimizing content to be cited by AI systems like ChatGPT, Claude, Perplexity, and other LLM-powered search engines. Unlike traditional SEO, GEO focuses on:

- **Structured Formatting**: Headers, bullet points, and clear hierarchies
- **Factual Density**: Statistics, citations, and verifiable claims
- **TL;DR Summaries**: Quick-reference sections for AI extraction
- **FAQ Sections**: Question-answer pairs for direct citations
- **Natural Language**: Human-readable, expert-quality writing

---

## Features

### Phase 1: GEO Knowledge Base
- **Scholarly Paper Search**: Uses Tavily to find GEO/SEO research papers
- **PDF Analysis**: Gemini Flash 2.0 multimodal analysis of academic papers
- **Guideline Extraction**: Extracts actionable optimization guidelines
- **Vector Storage**: Qdrant database with 5 specialized collections
- **Semantic Search**: Query guidelines with natural language

### Phase 2: Website Optimizer
- **Intelligent Crawling**: Uses Crawl4AI to extract website content
- **Industry Detection**: Auto-classifies content (Healthcare, Tech, E-commerce, etc.)
- **Parallel Processing**: Optimizes multiple content chunks simultaneously
- **GEO Scoring**: Before/after scoring to measure improvement
- **Anti-Hallucination**: Strict rules prevent AI from inventing statistics
- **Humanization**: Natural language post-processing for authentic tone
- **Real-time Progress**: Live updates via polling API
- **Export Options**: Download as Markdown or HTML

---

## Architecture

```
                                    AISEO System Architecture
    +===================================================================================+
    |                                                                                   |
    |   +-------------------+     +------------------+     +------------------------+   |
    |   |   React Frontend  | --> |   FastAPI v2     | --> |   Website Optimizer    |   |
    |   |   (Vite + TS)     |     |   REST API       |     |   LangChain Agent      |   |
    |   +-------------------+     +------------------+     +------------------------+   |
    |           |                         |                          |                  |
    |           v                         v                          v                  |
    |   +-------------------+     +---------------+          +------------------+       |
    |   |    Crawl4AI       |     |   SQLite DB   |          |    Qdrant DB     |       |
    |   |   Web Crawler     |     | Jobs, Chunks  |          | GEO Guidelines   |       |
    |   +-------------------+     +---------------+          +------------------+       |
    |                                                                                   |
    +===================================================================================+
```

### Website Optimization Workflow

```
1. CRAWL      -->  2. CLASSIFY  -->  3. CHUNK     -->  4. RETRIEVE
   Crawl4AI        LLM Industry       Split into        Qdrant Vector
   Extract         Detection          Sections          Search
       |                                                    |
       +----------------------------------------------------+
                                |
                                v
5. OPTIMIZE   -->  6. HUMANIZE  -->  7. ASSEMBLE  -->  8. DELIVER
   Apply GEO       Natural            Combine           Markdown
   Guidelines      Language           Chunks            + HTML
```

---

## Tech Stack

### Backend
| Technology | Purpose |
|------------|---------|
| Python 3.11+ | Core language |
| FastAPI | REST API framework |
| LangChain | LLM orchestration |
| Crawl4AI | Web crawling |
| Qdrant | Vector database |
| SQLite | Job persistence |
| OpenAI GPT-4 | LLM for optimization |

### Frontend
| Technology | Purpose |
|------------|---------|
| React 18 | UI framework |
| TypeScript | Type safety |
| Vite | Build tool |
| TailwindCSS | Styling |
| Framer Motion | Animations |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker (for Qdrant)
- OpenAI API key

### 1. Clone the Repository

```bash
git clone git@github.com:umerkhan95/AISEO_GEO_evaluator.git
cd AISEO_GEO_evaluator
```

### 2. Set Up Backend

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env

# Edit .env with your API keys
```

### 3. Start Qdrant

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 4. Initialize Database

```bash
# Initialize SQLite database
python database.py

# Populate Qdrant with GEO guidelines (optional)
python src/workflows/nodes/guideline_retriever.py
```

### 5. Start Backend Server

```bash
python api_v2.py
# Server runs on http://localhost:8001
```

### 6. Set Up Frontend

```bash
cd frontend
npm install
npm run dev
# Frontend runs on http://localhost:5173
```

---

## API Reference

### Website Optimization Endpoints

#### Start Optimization
```http
POST /api/v2/optimize
Content-Type: application/json

{
  "url": "https://example.com",
  "settings": {}
}
```

#### Get Job Status
```http
GET /api/v2/jobs/{job_id}
```

Response:
```json
{
  "job_id": "job_abc123",
  "status": "processing",
  "industry": "Technology",
  "total_chunks": 5,
  "completed_chunks": 3,
  "original_geo_score": 4.2,
  "optimized_geo_score": 8.5,
  "crawl_stats": {
    "crawl_time_ms": 2500,
    "word_count": 1500,
    "links_count": 12
  }
}
```

#### Get Results
```http
GET /api/v2/results/{job_id}
```

#### Download Markdown
```http
GET /api/v2/results/{job_id}/markdown
```

#### Download HTML
```http
GET /api/v2/results/{job_id}/html
```

#### Get Applied Guidelines
```http
GET /api/v2/results/{job_id}/guidelines
```

#### Get GEO Guidelines from Knowledge Base
```http
GET /api/v2/guidelines?category=citation&industry=Healthcare&limit=50
```

Query Parameters:
- `category`: Filter by category (`citation`, `universal`, `industry`, `technical`, `metrics`)
- `industry`: Filter by industry (`B2B_SaaS`, `Healthcare`, `E-commerce`, `Finance`, `Legal`, etc.)
- `view`: Set to `sources` to get unique research papers instead of guidelines
- `limit`: Maximum results (default 100)

**Get all research sources:**
```http
GET /api/v2/guidelines?view=sources
```

Response:
```json
{
  "total": 12,
  "sources": [
    {
      "title": "GEO: Generative Engine Optimization",
      "authors": ["Pranjal Aggarwal", "..."],
      "url": "https://arxiv.org/abs/2311.09735",
      "venue": "KDD 2024",
      "guidelines_count": 33
    }
  ]
}
```

#### Get Showcase (Optimized Websites)
```http
GET /api/v2/showcase?limit=50
```

#### Health Check
```http
GET /api/v2/health
```

---

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key | Yes |
| `OPENAI_MODEL` | Model to use (default: gpt-4o) | No |
| `QDRANT_HOST` | Qdrant host (default: localhost) | No |
| `QDRANT_PORT` | Qdrant port (default: 6333) | No |
| `QDRANT_API_KEY` | Qdrant API key (for cloud) | No |
| `LANGCHAIN_TRACING_V2` | Enable LangSmith | No |
| `LANGCHAIN_API_KEY` | LangSmith API key | No |

---

## Deployment

### Backend on Fly.io

```bash
# Install Fly CLI
brew install flyctl  # macOS

# Login
flyctl auth login

# Deploy
flyctl deploy

# Set secrets
flyctl secrets set OPENAI_API_KEY=sk-xxx
flyctl secrets set QDRANT_HOST=your-qdrant-host
flyctl secrets set QDRANT_API_KEY=your-qdrant-key
```

### Frontend Deployment

```bash
cd frontend
npm run build
# Deploy dist/ to Vercel, Netlify, or Fly.io
```

---

## Project Structure

```
AISEO/
├── api_v2.py                       # FastAPI REST API (v2)
├── api.py                          # Legacy API (v1)
├── database.py                     # SQLite operations
├── config.py                       # Configuration
├── requirements.txt                # Python dependencies
├── fly.toml                        # Fly.io config
├── Dockerfile                      # Docker config
├── .env.example                    # Environment template
├── src/
│   └── workflows/
│       ├── website_optimizer.py    # Main orchestrator
│       └── nodes/
│           ├── crawler_node.py     # Crawl4AI integration
│           ├── industry_classifier.py
│           ├── guideline_retriever.py
│           ├── chunk_optimizer.py
│           └── humanizer_node.py
└── frontend/
    ├── src/
    │   ├── pages/
    │   │   └── Optimize.tsx        # Main optimization UI
    │   └── components/
    ├── package.json
    └── vite.config.ts
```

---

## GEO Scoring Criteria

| Criteria | Weight | Description |
|----------|--------|-------------|
| Structure | 20% | Headers, bullet points, clear hierarchy |
| Factual Density | 25% | Statistics, citations, verifiable claims |
| TL;DR Presence | 15% | Summary sections for quick reference |
| FAQ Format | 15% | Question-answer pairs |
| Natural Language | 15% | Readability and authenticity |
| Expert Tone | 10% | Professional, authoritative voice |

---

## Qdrant Collections

5 specialized collections with **123 GEO guidelines** from **12 research sources**:

| Collection | Guidelines | Purpose |
|------------|------------|---------|
| `geo_seo_citation` | 38 | Citation optimization tactics |
| `geo_seo_technical` | 28 | Technical implementation |
| `geo_seo_universal` | 22 | Universal GEO/SEO guidelines |
| `geo_seo_metrics` | 18 | Metrics and analytics |
| `geo_seo_industry` | 17 | Industry-specific guidelines |

### Research Sources

| Source | Guidelines | URL |
|--------|------------|-----|
| AllAboutAI Research | 45 | [allaboutai.com](https://www.allaboutai.com/geo/what-is-generative-engine-optimization/) |
| Princeton GEO (KDD 2024) | 33 | [arxiv.org/abs/2311.09735](https://arxiv.org/abs/2311.09735) |
| AI Search Domination | 9 | [arxiv.org/abs/2509.08919](https://arxiv.org/abs/2509.08919) |
| Omnius GEO Report 2025 | 8 | [omnius.so](https://www.omnius.so/blog/geo-industry-report) |
| Averi AI LLM Guide | 8 | [averi.ai](https://www.averi.ai/breakdowns/the-definitive-guide-to-llm-optimized-content) |
| Superprompt AI Traffic | 6 | [superprompt.com](https://superprompt.com/blog/ai-traffic-up-527-percent-how-to-get-cited-by-chatgpt-claude-perplexity-2025) |
| StoryChief LLM Content | 5 | [storychief.io](https://storychief.io/blog/how-to-structure-your-content-so-llms-are-more-likely-to-cite-you) |
| Profound Citation Patterns | 4 | [tryprofound.com](https://www.tryprofound.com/blog/ai-platform-citation-patterns) |
| Metehan Perplexity Ranking | 3 | [metehan.ai](https://metehan.ai/blog/perplexity-ai-seo-59-ranking-patterns/) |
| Digital Bloom AI Visibility | 1 | [thedigitalbloom.com](https://thedigitalbloom.com/learn/2025-ai-citation-llm-visibility-report/) |
| Passionfruit SERP Analysis | 1 | [getpassionfruit.com](https://www.getpassionfruit.com/blog/why-ai-citations-lean-on-the-top-10) |

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Author

**Umer Khan** - [umerkhan.eu](https://umerkhan.eu)

- GitHub: [@umerkhan95](https://github.com/umerkhan95)
- LinkedIn: [umerkhan261](https://linkedin.com/in/umerkhan261)
- Email: hello@umerkhan.eu

---

## Live Demo

Try the GEO Website Optimizer: [umerkhan.eu/geo](https://umerkhan.eu/geo)
