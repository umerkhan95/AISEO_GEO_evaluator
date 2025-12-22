#!/usr/bin/env python3
"""
GEO/SEO Knowledge Base - Main Entry Point

This script provides a CLI interface to the Deep Agents-powered knowledge base.

Usage:
    python main.py research "citation optimization"
    python main.py process https://arxiv.org/pdf/xxx.pdf
    python main.py query "How to optimize for AI citations?"
    python main.py stats
    python main.py chat
    python main.py serve
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

# Load environment variables
load_dotenv()

console = Console()


def research_papers(topic: str, industry: str = None, max_results: int = 10):
    """Search for GEO/SEO papers using Tavily."""
    from src.tools.web_search import search_geo_seo_papers, search_scholarly_articles

    console.print(f"\n[bold blue]Searching for papers on:[/bold blue] {topic}")
    if industry:
        console.print(f"[dim]Industry filter: {industry}[/dim]")

    with console.status("[bold green]Searching..."):
        if industry:
            result = search_scholarly_articles(topic, industry, max_results=max_results)
        else:
            result = search_geo_seo_papers(topic, max_results=max_results)

    data = json.loads(result)

    if data.get("error"):
        console.print(f"[red]Error: {data['error']}[/red]")
        return

    # Display results
    table = Table(title="Research Results")
    table.add_column("Title", style="cyan", max_width=50)
    table.add_column("URL", style="dim", max_width=40)
    table.add_column("Score", justify="right")

    results = data.get("results") or data.get("articles", [])
    for item in results[:10]:
        table.add_row(
            item.get("title", "")[:50],
            item.get("url", "")[:40],
            str(round(item.get("score", 0), 2)),
        )

    console.print(table)

    if data.get("answer") or data.get("summary"):
        console.print(Panel(
            data.get("answer") or data.get("summary"),
            title="Summary",
            border_style="green",
        ))


def process_pdf(url: str, filename: str = None):
    """Download and process a PDF from URL."""
    from src.tools.pdf_tools import download_pdf, analyze_pdf_with_gemini
    from src.tools.qdrant_tools import store_guidelines

    console.print(f"\n[bold blue]Processing PDF from:[/bold blue] {url}")

    # Download
    with console.status("[bold green]Downloading PDF..."):
        download_result = json.loads(download_pdf(url, filename))

    if download_result.get("status") != "success":
        console.print(f"[red]Download failed: {download_result.get('error')}[/red]")
        return

    filepath = download_result["filepath"]
    console.print(f"[green]Downloaded to: {filepath}[/green]")
    console.print(f"[dim]Size: {download_result['file_size_mb']} MB[/dim]")

    # Analyze
    with console.status("[bold green]Analyzing with Gemini Flash 2.0..."):
        analysis_result = json.loads(analyze_pdf_with_gemini(filepath, "full"))

    if analysis_result.get("status") not in ["success", "partial"]:
        console.print(f"[red]Analysis failed: {analysis_result.get('error')}[/red]")
        return

    analysis = analysis_result.get("analysis", {})

    # Show structure
    structure = analysis.get("structure", {})
    console.print(Panel(
        f"**Title:** {structure.get('title', 'Unknown')}\n"
        f"**Authors:** {', '.join(structure.get('authors', []))}\n"
        f"**Themes:** {', '.join(analysis.get('themes', []))}",
        title="Document Structure",
        border_style="blue",
    ))

    # Extract guidelines
    guidelines = analysis.get("guidelines", [])
    console.print(f"\n[bold]Extracted {len(guidelines)} guidelines[/bold]")

    if guidelines:
        # Store in Qdrant
        with console.status("[bold green]Storing guidelines in Qdrant..."):
            source_paper = {
                "title": structure.get("title", "Unknown"),
                "authors": structure.get("authors", []),
                "url": url,
                "filepath": filepath,
            }
            store_result = json.loads(store_guidelines(guidelines, source_paper))

        console.print(f"[green]Stored {store_result.get('stored', 0)} guidelines[/green]")
        if store_result.get("failed", 0) > 0:
            console.print(f"[yellow]Failed to store {store_result['failed']} guidelines[/yellow]")

        # Show by collection
        by_collection = store_result.get("by_collection", {})
        if by_collection:
            table = Table(title="Guidelines by Collection")
            table.add_column("Collection", style="cyan")
            table.add_column("Count", justify="right")
            for coll, count in by_collection.items():
                table.add_row(coll, str(count))
            console.print(table)


def query_kb(query: str, collection: str = None, industry: str = None, limit: int = 10):
    """Query the knowledge base."""
    from src.tools.qdrant_tools import search_guidelines

    console.print(f"\n[bold blue]Querying:[/bold blue] {query}")

    with console.status("[bold green]Searching knowledge base..."):
        result = json.loads(search_guidelines(query, collection, industry, limit=limit))

    results = result.get("results", [])
    console.print(f"[dim]Found {len(results)} results[/dim]\n")

    for i, item in enumerate(results, 1):
        console.print(Panel(
            f"{item.get('guideline_text', '')}\n\n"
            f"[dim]Category: {item.get('category')} | "
            f"Priority: {item.get('priority')} | "
            f"Confidence: {item.get('confidence_score', 0):.2f} | "
            f"Relevance: {item.get('relevance_score', 0):.2f}[/dim]",
            title=f"[bold]#{i}[/bold] ({item.get('collection', 'unknown')})",
            border_style="green" if item.get("priority") == "high" else "blue",
        ))


def show_stats():
    """Show knowledge base statistics."""
    from src.tools.qdrant_tools import get_collection_stats

    console.print("\n[bold blue]Knowledge Base Statistics[/bold blue]")

    with console.status("[bold green]Fetching statistics..."):
        result = json.loads(get_collection_stats())

    # Total
    console.print(f"\n[bold]Total Guidelines: {result.get('total_guidelines', 0)}[/bold]")
    console.print(f"[dim]Average Confidence: {result.get('average_confidence', 0):.3f}[/dim]")

    # Collections table
    collections = result.get("collections", {})
    if collections:
        table = Table(title="Collections")
        table.add_column("Collection", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("Status")
        for name, info in collections.items():
            table.add_row(
                name,
                str(info.get("count", 0)),
                info.get("status", "unknown"),
            )
        console.print(table)

    # Priority distribution
    by_priority = result.get("by_priority", {})
    if by_priority:
        table = Table(title="By Priority")
        table.add_column("Priority", style="cyan")
        table.add_column("Count", justify="right")
        for priority, count in by_priority.items():
            table.add_row(priority, str(count))
        console.print(table)

    # Industry distribution
    by_industry = result.get("by_industry", {})
    if by_industry:
        table = Table(title="By Industry")
        table.add_column("Industry", style="cyan")
        table.add_column("Count", justify="right")
        for industry, count in sorted(by_industry.items(), key=lambda x: x[1], reverse=True)[:10]:
            table.add_row(industry, str(count))
        console.print(table)


def chat_mode():
    """Interactive chat with the Deep Agent."""
    from src.agents.orchestrator import create_geo_seo_agent

    console.print("\n[bold blue]GEO/SEO Knowledge Base Chat[/bold blue]")
    console.print("[dim]Type 'exit' or 'quit' to end the session[/dim]")
    console.print("[dim]Type 'help' for example queries[/dim]\n")

    with console.status("[bold green]Initializing Deep Agent..."):
        agent = create_geo_seo_agent()

    console.print("[green]Agent ready![/green]\n")

    while True:
        try:
            user_input = console.input("[bold cyan]You:[/bold cyan] ")

            if user_input.lower() in ["exit", "quit", "q"]:
                console.print("[dim]Goodbye![/dim]")
                break

            if user_input.lower() == "help":
                console.print(Panel(
                    "**Example queries:**\n"
                    "- Find papers about citation optimization for healthcare\n"
                    "- What are the best GEO practices for B2B SaaS?\n"
                    "- Show me technical implementation guidelines\n"
                    "- Search for schema markup best practices\n"
                    "- How do I optimize content for AI search engines?",
                    title="Help",
                    border_style="blue",
                ))
                continue

            if not user_input.strip():
                continue

            with console.status("[bold green]Thinking..."):
                messages = [{"role": "user", "content": user_input}]
                result = agent.invoke(messages)

            response_messages = result.get("messages", [])
            if response_messages:
                last_message = response_messages[-1]
                if hasattr(last_message, "content"):
                    content = last_message.content
                elif isinstance(last_message, dict):
                    content = last_message.get("content", str(last_message))
                else:
                    content = str(last_message)

                console.print(f"\n[bold green]Agent:[/bold green] {content}\n")
            else:
                console.print("[yellow]No response generated[/yellow]\n")

        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted. Goodbye![/dim]")
            break
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]\n")


def serve_api(host: str = "0.0.0.0", port: int = 8000):
    """Start the FastAPI server."""
    import uvicorn
    from api import api

    console.print(f"\n[bold blue]Starting API server on http://{host}:{port}[/bold blue]")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    uvicorn.run(api, host=host, port=port, log_level="info")


def main():
    parser = argparse.ArgumentParser(
        description="GEO/SEO Knowledge Base CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py research "citation optimization"
  python main.py research "healthcare GEO" --industry Healthcare
  python main.py process https://arxiv.org/pdf/2311.xxxxx.pdf
  python main.py query "How to optimize for AI citations?"
  python main.py query "schema markup" --collection technical
  python main.py stats
  python main.py chat
  python main.py serve --port 8000
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Research command
    research_parser = subparsers.add_parser("research", help="Search for GEO/SEO papers")
    research_parser.add_argument("topic", help="Research topic")
    research_parser.add_argument("--industry", "-i", help="Industry filter")
    research_parser.add_argument("--max", "-m", type=int, default=10, help="Max results")

    # Process command
    process_parser = subparsers.add_parser("process", help="Process a PDF from URL")
    process_parser.add_argument("url", help="PDF URL")
    process_parser.add_argument("--filename", "-f", help="Custom filename")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the knowledge base")
    query_parser.add_argument("query", help="Search query")
    query_parser.add_argument("--collection", "-c", help="Collection filter")
    query_parser.add_argument("--industry", "-i", help="Industry filter")
    query_parser.add_argument("--limit", "-l", type=int, default=10, help="Max results")

    # Stats command
    subparsers.add_parser("stats", help="Show knowledge base statistics")

    # Chat command
    subparsers.add_parser("chat", help="Interactive chat with Deep Agent")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host address")
    serve_parser.add_argument("--port", "-p", type=int, default=8000, help="Port number")

    args = parser.parse_args()

    if args.command == "research":
        research_papers(args.topic, args.industry, args.max)
    elif args.command == "process":
        process_pdf(args.url, args.filename)
    elif args.command == "query":
        query_kb(args.query, args.collection, args.industry, args.limit)
    elif args.command == "stats":
        show_stats()
    elif args.command == "chat":
        chat_mode()
    elif args.command == "serve":
        serve_api(args.host, args.port)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
