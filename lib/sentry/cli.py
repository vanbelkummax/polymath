#!/usr/bin/env python3
"""
Literature Sentry CLI - Autonomous polymathic resource curator.

Usage:
    python3 -m lib.sentry.cli "spatial transcriptomics" --max 15
    python3 -m lib.sentry.cli "transformer attention" --source github --min-stars 100
    python3 -m lib.sentry.cli --show-tickets
"""

import sys
from pathlib import Path
from typing import Optional, List
from enum import Enum

# Ensure lib is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import typer
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
except ImportError:
    print("Installing required packages: typer rich")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "typer", "rich", "-q"])
    import typer
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(
    name="literature-sentry",
    help="Autonomous polymathic resource curator - discovers and ingests best-of-breed content.",
    add_completion=False,
)
console = Console()


class SourceType(str, Enum):
    all = "all"
    europepmc = "europepmc"
    arxiv = "arxiv"
    biorxiv = "biorxiv"
    github = "github"
    youtube = "youtube"


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query for discovering resources"),
    source: SourceType = typer.Option(
        SourceType.all, "--source", "-s", help="Limit to specific source"
    ),
    max_results: int = typer.Option(
        20, "--max", "-n", help="Maximum items to discover per source"
    ),
    min_score: float = typer.Option(
        0.4, "--min-score", help="Minimum priority score to ingest (0.0-1.0)"
    ),
    min_stars: int = typer.Option(
        50, "--min-stars", help="Minimum GitHub stars (only for github source)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", "-d", help="Score and display but don't ingest"
    ),
    show_hidden_gems: bool = typer.Option(
        True, "--gems/--no-gems", help="Include hidden gems (low popularity, high relevance)"
    ),
):
    """
    Search for and ingest high-quality polymathic resources.

    Examples:
        literature-sentry "spatial transcriptomics visium"
        literature-sentry "colibactin pks E. coli" --source europepmc
        literature-sentry "attention mechanism" --source github --min-stars 100
    """
    from lib.sentry.sentry import Sentry

    console.print(f"\n[bold blue]Literature Sentry[/bold blue] - Query: [green]{query}[/green]")
    console.print(f"Sources: {source.value} | Max: {max_results} | Min Score: {min_score}")

    if dry_run:
        console.print("[yellow]DRY RUN - will not ingest[/yellow]")

    console.print()

    # Initialize sentry
    sentry = Sentry()

    # Determine which sources to query
    sources = [source.value] if source != SourceType.all else None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Discovery phase
        task = progress.add_task("Discovering resources...", total=None)

        results = sentry.discover(
            query=query,
            sources=sources,
            max_per_source=max_results,
            min_stars=min_stars if source == SourceType.github else None,
        )

        progress.update(task, description=f"Found {len(results)} candidates")

        # Scoring phase
        progress.update(task, description="Scoring candidates...")
        scored = sentry.score_all(results, query)

        # Filter by min_score
        qualified = [r for r in scored if r["priority_score"] >= min_score]
        hidden_gems = [r for r in scored if r.get("is_hidden_gem") and show_hidden_gems]

        # Combine, dedupe
        to_process = {r["external_id"]: r for r in qualified + hidden_gems}

        progress.update(task, description=f"Qualified: {len(to_process)} items")

    # Display results table
    _display_results(list(to_process.values()), dry_run)

    if dry_run:
        console.print("\n[yellow]Dry run complete. Use without --dry-run to ingest.[/yellow]")
        return

    # Ingestion phase
    if not to_process:
        console.print("\n[yellow]No items qualified for ingestion.[/yellow]")
        return

    console.print(f"\n[bold]Ingesting {len(to_process)} items...[/bold]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting...", total=len(to_process))

        ingested = 0
        paywalled = 0
        failed = 0

        for item in to_process.values():
            progress.update(task, description=f"Processing: {item['title'][:50]}...")

            result = sentry.fetch_and_ingest(item)

            if result["status"] == "ingested":
                ingested += 1
            elif result["status"] == "paywalled":
                paywalled += 1
            else:
                failed += 1

            progress.advance(task)

    # Summary
    console.print(f"\n[bold green]Ingestion Complete[/bold green]")
    console.print(f"  Ingested: {ingested}")
    console.print(f"  Paywalled (flagged): {paywalled}")
    console.print(f"  Failed: {failed}")

    if paywalled > 0:
        console.print(f"\n[yellow]Run 'literature-sentry --show-tickets' to see paywalled items.[/yellow]")


@app.command()
def show_tickets(
    limit: int = typer.Option(20, "--limit", "-n", help="Max tickets to show"),
    resolve: Optional[str] = typer.Option(None, "--resolve", help="Mark ticket ID as resolved"),
):
    """
    Show items that need manual download (paywalled, login required, etc.)
    """
    from lib.sentry.sentry import Sentry

    sentry = Sentry()

    if resolve:
        sentry.resolve_ticket(resolve)
        console.print(f"[green]Ticket {resolve} marked as resolved.[/green]")
        return

    tickets = sentry.get_pending_tickets(limit=limit)

    if not tickets:
        console.print("[green]No pending tickets! All caught up.[/green]")
        return

    table = Table(title="Paywalled Items - Manual Download Required")
    table.add_column("ID", style="dim")
    table.add_column("Title", style="cyan", max_width=50)
    table.add_column("DOI", style="green")
    table.add_column("Reason")
    table.add_column("Created")

    for t in tickets:
        table.add_row(
            str(t["id"]),
            t["title"][:50] + "..." if len(t["title"]) > 50 else t["title"],
            t.get("doi", "-"),
            t["reason"],
            t["created_at"].strftime("%Y-%m-%d") if t.get("created_at") else "-",
        )

    console.print(table)
    console.print(f"\n[dim]After downloading, place PDFs in staging and run:[/dim]")
    console.print(f"[dim]  python3 lib/unified_ingest.py /path/to/file.pdf[/dim]")
    console.print(f"[dim]  literature-sentry --show-tickets --resolve <ID>[/dim]")


@app.command()
def stats():
    """Show sentry statistics - items discovered, ingested, pending."""
    from lib.sentry.sentry import Sentry

    sentry = Sentry()
    stats = sentry.get_stats()

    console.print("\n[bold blue]Literature Sentry Statistics[/bold blue]\n")

    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="green", justify="right")

    table.add_row("Total Discovered", str(stats["total_discovered"]))
    table.add_row("Ingested", str(stats["ingested"]))
    table.add_row("Pending", str(stats["pending"]))
    table.add_row("Paywalled (needs user)", str(stats["paywalled"]))
    table.add_row("Failed", str(stats["failed"]))
    table.add_row("Hidden Gems Found", str(stats["hidden_gems"]))

    console.print(table)

    console.print("\n[bold]By Source:[/bold]")
    for source, count in stats.get("by_source", {}).items():
        console.print(f"  {source}: {count}")


def _display_results(results: List[dict], dry_run: bool):
    """Display scored results in a table."""
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(title="Discovered Resources" + (" (DRY RUN)" if dry_run else ""))
    table.add_column("Score", style="cyan", justify="right")
    table.add_column("Source", style="blue")
    table.add_column("Title", max_width=50)
    table.add_column("Concepts", style="green", max_width=30)
    table.add_column("Gem?", justify="center")

    # Sort by score
    results = sorted(results, key=lambda x: x.get("priority_score", 0), reverse=True)

    for r in results[:30]:  # Limit display
        score = f"{r.get('priority_score', 0):.2f}"
        source = r.get("source", "?")
        title = r.get("title", "Untitled")
        if len(title) > 50:
            title = title[:47] + "..."
        concepts = ", ".join(r.get("concept_domains", [])[:3])
        gem = "[yellow]*[/yellow]" if r.get("is_hidden_gem") else ""

        table.add_row(score, source, title, concepts, gem)

    console.print(table)

    if len(results) > 30:
        console.print(f"[dim]... and {len(results) - 30} more[/dim]")


if __name__ == "__main__":
    app()
