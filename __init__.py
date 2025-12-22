"""
GEO/SEO Knowledge Base - LangGraph Architecture

A comprehensive system for processing scholarly PDFs about GEO/SEO
and building a queryable vector knowledge base.
"""

__version__ = "1.0.0"

from .graph import app
from .state import GraphState, PDFDocument, Guideline, EnrichedGuideline
from .config import config

__all__ = [
    "app",
    "GraphState",
    "PDFDocument",
    "Guideline",
    "EnrichedGuideline",
    "config",
]
