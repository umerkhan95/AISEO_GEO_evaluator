"""
PDF processing tools for the GEO/SEO Knowledge Base.

These tools enable the Deep Agent to:
1. Download PDFs from URLs
2. Extract text content from PDFs
3. Analyze PDFs using Gemini Flash 2.0 for structure extraction
"""

import os
import json
import hashlib
import base64
from pathlib import Path
from typing import Optional
from datetime import datetime

import httpx
import google.generativeai as genai
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini
_gemini_model = None


def _get_gemini_model():
    """Get or create Gemini model instance."""
    global _gemini_model
    if _gemini_model is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        genai.configure(api_key=api_key)
        _gemini_model = genai.GenerativeModel("gemini-2.0-flash")
    return _gemini_model


def download_pdf(
    url: str,
    filename: Optional[str] = None,
    download_dir: Optional[str] = None
) -> str:
    """
    Download a PDF from a URL and save it locally.

    This tool downloads PDF files from URLs (including arxiv, researchgate, etc.)
    and saves them to the local filesystem for processing.

    Args:
        url: The URL of the PDF to download
        filename: Optional custom filename (default: derived from URL or hash)
        download_dir: Optional directory to save to (default: ./pdfs)

    Returns:
        JSON string with download status, file path, and metadata
    """
    # Set up download directory
    base_dir = download_dir or os.getenv("PDF_DOWNLOAD_DIR", "./pdfs")
    Path(base_dir).mkdir(parents=True, exist_ok=True)

    # Generate filename if not provided
    if not filename:
        # Use URL hash for unique filename
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        # Try to extract meaningful name from URL
        url_parts = url.rstrip("/").split("/")
        if url_parts[-1].endswith(".pdf"):
            filename = url_parts[-1]
        else:
            filename = f"paper_{url_hash}.pdf"

    filepath = Path(base_dir) / filename

    result = {
        "url": url,
        "filename": filename,
        "filepath": str(filepath),
        "status": "pending",
        "file_size_mb": 0,
        "download_timestamp": datetime.now().isoformat(),
    }

    try:
        # Download the PDF
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; GEO-SEO-KB/1.0; +research)",
            "Accept": "application/pdf,*/*",
        }

        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()

            # Verify it's a PDF
            content_type = response.headers.get("content-type", "")
            if "pdf" not in content_type.lower() and not url.endswith(".pdf"):
                # Check if content starts with PDF magic bytes
                if not response.content[:4] == b"%PDF":
                    result["status"] = "error"
                    result["error"] = f"URL does not return a PDF (content-type: {content_type})"
                    return json.dumps(result, indent=2)

            # Save the PDF
            with open(filepath, "wb") as f:
                f.write(response.content)

            # Get file size
            file_size = len(response.content) / (1024 * 1024)  # MB

            result["status"] = "success"
            result["file_size_mb"] = round(file_size, 2)
            result["content_type"] = content_type

    except httpx.HTTPStatusError as e:
        result["status"] = "error"
        result["error"] = f"HTTP error {e.response.status_code}: {str(e)}"
    except httpx.TimeoutException:
        result["status"] = "error"
        result["error"] = "Download timed out after 60 seconds"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return json.dumps(result, indent=2)


def extract_pdf_text(
    filepath: str,
    max_pages: Optional[int] = None
) -> str:
    """
    Extract text content from a PDF file using PyPDF.

    This is a fast text extraction that works well for text-based PDFs.
    For scanned PDFs or complex layouts, use analyze_pdf_with_gemini instead.

    Args:
        filepath: Path to the PDF file
        max_pages: Maximum number of pages to extract (default: all)

    Returns:
        JSON string with extracted text and metadata
    """
    result = {
        "filepath": filepath,
        "status": "pending",
        "total_pages": 0,
        "pages_extracted": 0,
        "text_content": "",
        "metadata": {},
    }

    try:
        if not os.path.exists(filepath):
            result["status"] = "error"
            result["error"] = f"File not found: {filepath}"
            return json.dumps(result, indent=2)

        reader = PdfReader(filepath)
        total_pages = len(reader.pages)
        result["total_pages"] = total_pages

        # Extract metadata
        if reader.metadata:
            result["metadata"] = {
                "title": reader.metadata.get("/Title", ""),
                "author": reader.metadata.get("/Author", ""),
                "subject": reader.metadata.get("/Subject", ""),
                "creator": reader.metadata.get("/Creator", ""),
            }

        # Extract text from pages
        pages_to_extract = min(total_pages, max_pages) if max_pages else total_pages
        text_parts = []

        for i in range(pages_to_extract):
            page = reader.pages[i]
            text = page.extract_text() or ""
            text_parts.append(f"--- Page {i + 1} ---\n{text}")

        result["text_content"] = "\n\n".join(text_parts)
        result["pages_extracted"] = pages_to_extract
        result["status"] = "success"
        result["char_count"] = len(result["text_content"])

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return json.dumps(result, indent=2)


def analyze_pdf_with_gemini(
    filepath: str,
    analysis_type: str = "full"
) -> str:
    """
    Analyze a PDF using Gemini Flash 2.0 for intelligent content extraction.

    This tool uses Gemini's multimodal capabilities to:
    1. Extract document structure (title, authors, sections)
    2. Identify main themes and topics
    3. Detect industry mentions
    4. Extract actionable GEO/SEO guidelines

    Args:
        filepath: Path to the PDF file
        analysis_type: Type of analysis:
            - "structure": Extract document structure only
            - "guidelines": Extract actionable guidelines
            - "full": Complete analysis (structure + guidelines + themes)

    Returns:
        JSON string with analysis results including structure and extracted content
    """
    result = {
        "filepath": filepath,
        "analysis_type": analysis_type,
        "status": "pending",
        "analysis": {},
    }

    try:
        if not os.path.exists(filepath):
            result["status"] = "error"
            result["error"] = f"File not found: {filepath}"
            return json.dumps(result, indent=2)

        model = _get_gemini_model()

        # Read PDF and convert to base64
        with open(filepath, "rb") as f:
            pdf_data = f.read()

        # Upload PDF to Gemini
        pdf_file = genai.upload_file(filepath, mime_type="application/pdf")

        # Build prompt based on analysis type
        if analysis_type == "structure":
            prompt = """Analyze this academic paper/article and extract its structure.
            Return a JSON object with:
            {
                "title": "Paper title",
                "authors": ["Author 1", "Author 2"],
                "abstract": "Abstract text",
                "sections": [
                    {"title": "Section title", "summary": "Brief summary of section content"}
                ],
                "publication_year": "Year if mentioned",
                "source": "Journal/conference if mentioned"
            }
            """
        elif analysis_type == "guidelines":
            prompt = """Extract all actionable GEO (Generative Engine Optimization) and SEO guidelines from this paper.
            Return a JSON object with:
            {
                "guidelines": [
                    {
                        "content": "The specific actionable guideline",
                        "category": "One of: universal_seo_geo, industry_specific, technical, citation_optimization, metrics",
                        "industries": ["List of applicable industries or empty if universal"],
                        "source_section": "Section where this was found",
                        "confidence": 0.0 to 1.0
                    }
                ]
            }
            Focus on specific, actionable recommendations, not general observations.
            """
        else:  # full analysis
            prompt = """Perform a comprehensive analysis of this GEO/SEO academic paper/article.
            Return a JSON object with:
            {
                "structure": {
                    "title": "Paper title",
                    "authors": ["Author 1", "Author 2"],
                    "abstract": "Abstract text",
                    "sections": [{"title": "Section", "summary": "Summary"}],
                    "publication_year": "Year",
                    "source": "Journal/conference"
                },
                "themes": ["Main theme 1", "Main theme 2"],
                "industries_mentioned": ["Industry 1", "Industry 2"],
                "guidelines": [
                    {
                        "content": "Actionable guideline",
                        "category": "universal_seo_geo|industry_specific|technical|citation_optimization|metrics",
                        "industries": [],
                        "source_section": "Section",
                        "confidence": 0.9,
                        "has_quantitative_data": true/false,
                        "has_case_study": true/false
                    }
                ],
                "key_statistics": ["Any notable statistics or data points"],
                "citations_style_examples": ["Examples of citation-worthy content formatting"]
            }
            """

        # Generate analysis
        response = model.generate_content([pdf_file, prompt])

        # Parse response
        response_text = response.text

        # Try to extract JSON from response
        try:
            # Find JSON in response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                analysis = json.loads(json_match.group())
                result["analysis"] = analysis
                result["status"] = "success"
            else:
                result["analysis"] = {"raw_response": response_text}
                result["status"] = "partial"
                result["note"] = "Could not parse structured JSON from response"
        except json.JSONDecodeError:
            result["analysis"] = {"raw_response": response_text}
            result["status"] = "partial"
            result["note"] = "Response was not valid JSON"

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return json.dumps(result, indent=2)


def batch_analyze_pdfs(
    filepaths: list[str],
    analysis_type: str = "full"
) -> str:
    """
    Analyze multiple PDFs in batch.

    Args:
        filepaths: List of PDF file paths to analyze
        analysis_type: Type of analysis for all PDFs

    Returns:
        JSON string with results for all PDFs
    """
    results = {
        "total_files": len(filepaths),
        "successful": 0,
        "failed": 0,
        "analyses": []
    }

    for filepath in filepaths:
        analysis_result = analyze_pdf_with_gemini(filepath, analysis_type)
        parsed = json.loads(analysis_result)

        if parsed.get("status") == "success":
            results["successful"] += 1
        else:
            results["failed"] += 1

        results["analyses"].append(parsed)

    return json.dumps(results, indent=2)
