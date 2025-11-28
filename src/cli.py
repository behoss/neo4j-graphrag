"""Command-line interface for GraphRAG pipeline."""

# Suppress Vertex AI deprecation warnings BEFORE any imports
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="vertexai")
warnings.filterwarnings("ignore", message=".*deprecated.*")

import asyncio
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.prompt import Prompt, Confirm

from .config import PipelineConfig
from .database import DatabaseManager
from .pipeline import GraphRAGPipeline, set_progress_callback
from .logging_config import setup_logging

console = Console()

# Global for progress updates
_current_progress_task = None
_current_progress = None


def get_config() -> PipelineConfig:
    """Load configuration from environment."""
    try:
        return PipelineConfig.from_env()
    except ValueError as e:
        console.print(f"[red]Configuration Error:[/red] {e}")
        console.print("\nPlease ensure you have a .env.local file with:")
        console.print("  - NEO4J_URI")
        console.print("  - NEO4J_USERNAME")
        console.print("  - NEO4J_PASSWORD")
        console.print("  - GOOGLE_API_KEY")
        sys.exit(1)


def cmd_ingest(file_path: str, title: Optional[str] = None, overwrite: bool = False):
    """Ingest a document into the knowledge graph."""
    global _current_progress, _current_progress_task

    config = get_config()
    path = Path(file_path)

    if not path.exists():
        console.print(f"[red]Error:[/red] File not found: {file_path}")
        sys.exit(1)

    # Read file content
    if path.suffix == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    elif path.suffix == ".md":
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        console.print(f"[red]Error:[/red] Unsupported file type: {path.suffix}")
        console.print("Supported types: .txt, .md")
        sys.exit(1)

    # Generate document ID and title
    doc_id = path.stem.lower().replace(" ", "-").replace("_", "-")
    doc_title = title or path.stem.replace("-", " ").replace("_", " ").title()

    # Estimate chunks for display (SimpleKGPipeline uses ~4000 chars per chunk)
    estimated_chunks = max(1, len(text) // 4000)

    console.print(
        Panel.fit(
            f"[bold]Document Ingestion[/bold]\n\n"
            f"File: {path.name}\n"
            f"Title: {doc_title}\n"
            f"ID: {doc_id}\n"
            f"Size: {len(text):,} characters\n"
            f"Est. chunks: ~{estimated_chunks}",
            title="📚 GraphRAG",
        )
    )

    def progress_callback(message: str, current: int, total: int):
        """Update progress display."""
        global _current_progress, _current_progress_task
        if _current_progress and _current_progress_task is not None:
            _current_progress.update(_current_progress_task, description=f"{message}")

    with GraphRAGPipeline(config) as pipeline:
        # Check if document exists
        docs = pipeline.db.get_documents()
        existing = [d for d in docs if d.get("id") == doc_id]

        if existing and not overwrite:
            if not Confirm.ask(f"Document '{doc_id}' already exists. Overwrite?"):
                console.print("[yellow]Ingestion cancelled[/yellow]")
                return

            # Delete existing document
            console.print("[dim]Removing existing document...[/dim]")
            pipeline.db.delete_document(doc_id)

        # Initialize database
        console.print("[dim]Initializing database...[/dim]")
        pipeline.db.initialize()

        # Set up progress callback
        set_progress_callback(progress_callback)

        # Ingest document with live progress
        console.print()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=False,
        ) as progress:
            _current_progress = progress
            _current_progress_task = progress.add_task("Starting ingestion...", total=None)

            try:
                result = asyncio.run(
                    pipeline.ingest_text(
                        text=text,
                        document_id=doc_id,
                        document_title=doc_title,
                        metadata={"source_file": str(path)},
                    )
                )
                progress.update(_current_progress_task, description="[green]Complete![/green]")
            except Exception as e:
                progress.update(
                    _current_progress_task, description=f"[red]Failed: {str(e)[:50]}[/red]"
                )
                console.print(f"\n[red]Error during ingestion:[/red] {e}")
                console.print(f"\n📝 Check log file for details: {pipeline.md_logger.log_file}")
                raise
            finally:
                _current_progress = None
                _current_progress_task = None
                set_progress_callback(None)

        # Show results
        console.print("\n[green]✓ Ingestion complete![/green]\n")

        table = Table(title="Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Document ID", result.get("document_id", doc_id))
        table.add_row("Chunks", str(result.get("chunks", 0)))
        table.add_row("Entities", str(result.get("entities", 0)))
        table.add_row("Relationships", str(result.get("relationships", 0)))
        if result.get("elapsed_seconds"):
            table.add_row("Time", f"{result['elapsed_seconds']:.1f} seconds")
        console.print(table)

        console.print(f"\n📝 Log file: {pipeline.md_logger.log_file}")
        console.print(
            "[dim]View the log file for detailed chunks, entities, and relationships.[/dim]"
        )


def cmd_query(document_id: Optional[str] = None, show_context: bool = True):
    """Interactive query session."""
    config = get_config()

    with GraphRAGPipeline(config) as pipeline:
        # List available documents
        docs = pipeline.db.get_documents()

        if not docs:
            console.print("[yellow]No documents found in the database.[/yellow]")
            console.print("Use 'graphrag ingest <file>' to add documents.")
            return

        # Show available documents
        console.print(
            Panel.fit(
                "[bold]Available Documents[/bold]",
                title="📚 GraphRAG Query",
            )
        )

        table = Table()
        table.add_column("#", style="dim")
        table.add_column("Title", style="cyan")
        table.add_column("ID", style="dim")
        table.add_column("Entities", style="green")
        table.add_column("Chunks", style="blue")

        for i, doc in enumerate(docs, 1):
            table.add_row(
                str(i),
                doc.get("title", "Untitled"),
                doc.get("id", ""),
                str(doc.get("entity_count", 0)),
                str(doc.get("chunk_count", 0)),
            )

        console.print(table)

        # Select document (optional)
        if document_id is None and len(docs) > 1:
            choice = Prompt.ask(
                "\nSelect document number (or 'all' for all documents)",
                default="all",
            )
            if choice.lower() != "all":
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(docs):
                        document_id = docs[idx].get("id")
                except ValueError:
                    pass

        # Query loop
        console.print("\n[bold]Enter your questions (type 'exit' to quit)[/bold]\n")

        while True:
            try:
                question = Prompt.ask("[cyan]Question[/cyan]")

                if question.lower() in ["exit", "quit", "q"]:
                    console.print("\n👋 Goodbye!")
                    break

                if not question.strip():
                    continue

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Thinking...", total=None)
                    result = asyncio.run(
                        pipeline.query(
                            question,
                            document_id,
                            return_context=True,
                            include_citations=False,  # Pure answer without [1], [2] markers
                        )
                    )
                    progress.update(task, completed=True)

                # Handle different result types
                if isinstance(result, dict):
                    # Display retrieved context with citation markers
                    if show_context:
                        _display_query_context(
                            result.get("chunks", []),
                            result.get("graph_context", []),
                            result.get("citations", []),
                        )
                    answer = result.get("answer", "")
                else:
                    answer = str(result)

                console.print(Panel(answer, title="💡 Answer", border_style="green"))
                console.print()

            except KeyboardInterrupt:
                console.print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                console.print(f"\n[red]Error:[/red] {e}\n")


def _display_query_context(chunks: list, graph_context: list, citations: Optional[list] = None):
    """Display retrieved chunks and graph context in a beautiful format."""
    # Display retrieved chunks
    if chunks:
        console.print(
            Panel.fit("[bold]📄 Retrieved Chunks[/bold]", border_style="blue")
        )

        chunk_table = Table(show_header=True, header_style="bold blue", box=None)
        chunk_table.add_column("#", style="dim", width=3)
        chunk_table.add_column("Score", style="cyan", width=8)
        chunk_table.add_column("Source", style="magenta", width=10)
        chunk_table.add_column("Content Preview", style="white")

        for i, chunk in enumerate(chunks[:5], 1):
            score = chunk.get("score", "N/A")
            score_str = f"{score:.3f}" if isinstance(score, float) else str(score)

            # Determine source type
            search_type = chunk.get("search_type", chunk.get("source", "vector"))
            if search_type == "vector":
                source_display = "🔍 vector"
            elif search_type == "fulltext":
                source_display = "📝 text"
            else:
                source_display = "🔀 hybrid"

            text = chunk.get("text", "")
            preview = (
                text[:150].replace("\n", " ") + "..."
                if len(text) > 150
                else text.replace("\n", " ")
            )
            chunk_table.add_row(str(i), score_str, source_display, preview)

        console.print(chunk_table)
        console.print()

    # Display graph context
    if graph_context:
        console.print(
            Panel.fit("[bold]🔗 Knowledge Graph Relationships[/bold]", border_style="magenta")
        )

        rel_table = Table(show_header=True, header_style="bold magenta", box=None)
        rel_table.add_column("Source", style="cyan")
        rel_table.add_column("Relationship", style="yellow")
        rel_table.add_column("Target", style="green")

        for rel in graph_context[:15]:
            source = rel.get("source", "?")
            relationship = rel.get("relationship", "?")
            target = rel.get("target", "?")
            rel_table.add_row(source, f"--[{relationship}]-->", target)

        console.print(rel_table)

        if len(graph_context) > 15:
            console.print(f"[dim]... and {len(graph_context) - 15} more relationships[/dim]")
        console.print()


def cmd_list():
    """List all documents in the database."""
    config = get_config()

    with DatabaseManager(config.neo4j) as db:
        docs = db.get_documents()

        if not docs:
            console.print("[yellow]No documents found in the database.[/yellow]")
            return

        console.print(
            Panel.fit(
                f"[bold]Documents ({len(docs)} total)[/bold]",
                title="📚 GraphRAG",
            )
        )

        table = Table()
        table.add_column("Title", style="cyan")
        table.add_column("ID", style="dim")
        table.add_column("Entities", style="green")
        table.add_column("Chunks", style="blue")

        for doc in docs:
            table.add_row(
                doc.get("title", "Untitled"),
                doc.get("id", ""),
                str(doc.get("entity_count", 0)),
                str(doc.get("chunk_count", 0)),
            )

        console.print(table)


def cmd_info(document_id: str):
    """Show detailed information about a document."""
    config = get_config()

    with DatabaseManager(config.neo4j) as db:
        docs = db.get_documents()
        doc = next((d for d in docs if d.get("id") == document_id), None)

        if not doc:
            console.print(f"[red]Document not found:[/red] {document_id}")
            return

        console.print(
            Panel.fit(
                f"[bold]{doc.get('title', 'Untitled')}[/bold]\n\n"
                f"ID: {doc.get('id')}\n"
                f"Path: {doc.get('path', 'N/A')}\n"
                f"Entities: {doc.get('entity_count', 0)}\n"
                f"Chunks: {doc.get('chunk_count', 0)}",
                title="📄 Document Info",
            )
        )

        # Show sample entities
        entities = db.get_document_entities(document_id, limit=20)
        if entities:
            console.print("\n[bold]Sample Entities:[/bold]")
            table = Table()
            table.add_column("Name", style="cyan")
            table.add_column("Type", style="green")
            table.add_column("Description", style="dim")

            for entity in entities:
                labels = entity.get("labels", [])
                entity_type = [l for l in labels if l != "__Entity__"]
                table.add_row(
                    entity.get("name", ""),
                    entity_type[0] if entity_type else "Entity",
                    (entity.get("description", "") or "")[:50],
                )

            console.print(table)


def cmd_delete(document_id: str, force: bool = False):
    """Delete a document from the database."""
    config = get_config()

    with DatabaseManager(config.neo4j) as db:
        docs = db.get_documents()
        doc = next((d for d in docs if d.get("id") == document_id), None)

        if not doc:
            console.print(f"[red]Document not found:[/red] {document_id}")
            return

        if not force:
            if not Confirm.ask(
                f"Delete document '{doc.get('title', document_id)}'? This cannot be undone."
            ):
                console.print("[yellow]Deletion cancelled[/yellow]")
                return

        deleted = db.delete_document(document_id)
        console.print(f"[green]✓ Deleted {deleted} nodes[/green]")


def cmd_stats():
    """Show database statistics."""
    config = get_config()

    with DatabaseManager(config.neo4j) as db:
        stats = db.get_statistics()

        console.print(
            Panel.fit(
                "[bold]Database Statistics[/bold]",
                title="📊 GraphRAG",
            )
        )

        # Node counts
        if stats.get("nodes_by_label"):
            console.print("\n[bold]Nodes by Label:[/bold]")
            table = Table()
            table.add_column("Label", style="cyan")
            table.add_column("Count", style="green")

            for label, count in stats["nodes_by_label"].items():
                table.add_row(label, str(count))

            console.print(table)

        # Relationship counts
        if stats.get("relationships_by_type"):
            console.print("\n[bold]Relationships by Type:[/bold]")
            table = Table()
            table.add_column("Type", style="cyan")
            table.add_column("Count", style="green")

            for rel_type, count in stats["relationships_by_type"].items():
                table.add_row(rel_type, str(count))

            console.print(table)

        # Totals
        console.print(f"\n[bold]Total Nodes:[/bold] {stats.get('total_nodes', 0)}")
        console.print(f"[bold]Total Relationships:[/bold] {stats.get('total_relationships', 0)}")


def cmd_purge(force: bool = False):
    """Purge all data from the database."""
    config = get_config()

    if not force:
        console.print("[red bold]⚠️  WARNING: This will delete ALL data![/red bold]")
        if not Confirm.ask("Are you absolutely sure?"):
            console.print("[yellow]Purge cancelled[/yellow]")
            return

        confirmation = Prompt.ask("Type 'DELETE EVERYTHING' to confirm")
        if confirmation != "DELETE EVERYTHING":
            console.print("[yellow]Purge cancelled[/yellow]")
            return

    with DatabaseManager(config.neo4j) as db:
        db.clear_all()
        console.print("[green]✓ Database purged[/green]")


def cmd_init():
    """Initialize database with constraints and indexes."""
    config = get_config()

    console.print(
        Panel.fit(
            "[bold]Database Initialization[/bold]",
            title="🔧 GraphRAG",
        )
    )

    with DatabaseManager(config.neo4j) as db:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Setting up constraints...", total=None)
            db.setup_constraints()
            progress.update(task, description="Setting up indexes...")
            db.setup_indexes()
            progress.update(task, description="Setting up vector index...")
            db.setup_vector_index(
                index_name=config.vector_index_name,
                dimensions=config.vector_dimensions,
            )
            progress.update(task, completed=True)

        console.print("[green]✓ Database initialized[/green]")


def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="GraphRAG - Knowledge Graph RAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a document")
    ingest_parser.add_argument("file", help="Path to the document file")
    ingest_parser.add_argument("--title", "-t", help="Document title")
    ingest_parser.add_argument("--overwrite", "-o", action="store_true", help="Overwrite existing")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the knowledge graph")
    query_parser.add_argument("--document", "-d", help="Document ID to query")

    # List command
    subparsers.add_parser("list", help="List all documents")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show document info")
    info_parser.add_argument("document_id", help="Document ID")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a document")
    delete_parser.add_argument("document_id", help="Document ID")
    delete_parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation")

    # Stats command
    subparsers.add_parser("stats", help="Show database statistics")

    # Purge command
    purge_parser = subparsers.add_parser("purge", help="Purge all data")
    purge_parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation")

    # Init command
    subparsers.add_parser("init", help="Initialize database")

    args = parser.parse_args()

    # Set up logging
    setup_logging()

    if args.command == "ingest":
        cmd_ingest(args.file, args.title, args.overwrite)
    elif args.command == "query":
        cmd_query(args.document)
    elif args.command == "list":
        cmd_list()
    elif args.command == "info":
        cmd_info(args.document_id)
    elif args.command == "delete":
        cmd_delete(args.document_id, args.force)
    elif args.command == "stats":
        cmd_stats()
    elif args.command == "purge":
        cmd_purge(args.force)
    elif args.command == "init":
        cmd_init()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
