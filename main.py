"""ModelHub Downloader CLI - Download models from HuggingFace and ModelScope.

A user-friendly CLI tool for downloading public AI models without authentication.
Supports HuggingFace and ModelScope with ModelScope as the recommended source.

Usage:
    python main.py --model Qwen/Qwen3-ASR-1.7B --source ms --output ./models
"""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table

from core.downloader import ModelDownloader, Source, create_downloader

# Create Typer app
app = typer.Typer(
    name="modelhub",
    help="ModelHub Downloader - Download AI models from HuggingFace and ModelScope",
    add_completion=False,
    rich_markup_mode="rich",
)

# Default output directory
DEFAULT_OUTPUT = "./models"


def version_callback(value: bool):
    """Print version and exit."""
    if value:
        rprint("[bold cyan]ModelHub Downloader[/bold cyan] v2.0.0")
        rprint("[dim]Built with Typer + Rich[/dim]")
        raise typer.Exit()


@app.command("download", help="Download a model from HuggingFace or ModelScope")
def download_model(
    model_id: str = typer.Argument(
        ...,
        help="Model ID in format 'namespace/model_name' (e.g., Qwen/Qwen3-ASR-1.7B)",
    ),
    source: str = typer.Option(
        "ms",  # ModelScope as default
        "--source",
        "-s",
        help="Download source: hf=HuggingFace, ms=ModelScope (recommended)",
        case_sensitive=False,
    ),
    output: str = typer.Option(
        DEFAULT_OUTPUT,
        "--output",
        "-o",
        help="Output directory for downloaded models",
    ),
    verify: bool = typer.Option(
        True,
        "--verify/--no-verify",
        help="Verify download after completion",
    ),
):
    """Download a model from HuggingFace or ModelScope.

    Examples:
        [dim]# Download from ModelScope (recommended)[/dim]
        modelhub download Qwen/Qwen3-ASR-1.7B

        [dim]# Download from HuggingFace[/dim]
        modelhub download Qwen/Qwen3-ASR-1.7B --source hf

        [dim]# Download to custom directory[/dim]
        modelhub download Qwen/Qwen3-ASR-1.7B -o D:/ComfyUI/models
    """
    # Validate source
    source = source.lower()
    if source not in ["hf", "ms", "auto"]:
        rprint("[red]❌ Invalid source. Use 'hf' or 'ms'[/red]")
        raise typer.Exit(1)

    # Create downloader and run
    downloader = create_downloader(
        output_dir=output,
        source=source,
    )

    # Download
    success = downloader.download(model_id)

    # Verify if requested
    if verify and success:
        downloader.verify_download(model_id)

    # Exit with appropriate code
    raise typer.Exit(0 if success else 1)


@app.command("list", help="List supported sources and their status")
def list_sources():
    """List available download sources."""
    table = Table(title="Download Sources")
    table.add_column("Source", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Status", style="yellow")

    table.add_row(
        "ms",
        "ModelScope (Recommended) - China's AI model hub",
        "[green][OK][/green]",
    )
    table.add_row(
        "hf",
        "HuggingFace - Global AI model hub",
        "[green][OK][/green]",
    )
    table.add_row(
        "auto",
        "Auto-detect - Try HF first, fallback to MS",
        "[green][OK][/green]",
    )

    from rich.console import Console

    console = Console()
    console.print(table)


@app.command("clean", help="Clean cache directories")
def clean_cache(
    hf: bool = typer.Option(False, "--hf", help="Clean HuggingFace cache"),
    ms: bool = typer.Option(False, "--ms", help="Clean ModelScope cache"),
    all: bool = typer.Option(False, "--all", "-a", help="Clean all caches"),
):
    """Clean model cache directories."""
    from pathlib import Path
    import shutil

    console = typer.get_console()

    hf_cache = Path.home() / ".cache" / "huggingface"
    ms_cache = Path.home() / ".cache" / "modelscope"

    cleaned = []

    if all or hf:
        if hf_cache.exists():
            try:
                shutil.rmtree(hf_cache)
                cleaned.append("HuggingFace")
                rprint(f"[green]✅ Cleaned HuggingFace cache[/green]")
            except Exception as e:
                rprint(f"[red]❌ Failed to clean HF cache: {e}[/red]")
        else:
            rprint("[dim]HuggingFace cache not found[/dim]")

    if all or ms:
        if ms_cache.exists():
            try:
                shutil.rmtree(ms_cache)
                cleaned.append("ModelScope")
                rprint(f"[green]✅ Cleaned ModelScope cache[/green]")
            except Exception as e:
                rprint(f"[red]❌ Failed to clean MS cache: {e}[/red]")
        else:
            rprint("[dim]ModelScope cache not found[/dim]")

    if not cleaned:
        rprint("[yellow]⚠️  No caches cleaned. Use --hf, --ms, or --all[/yellow]")


@app.command("interactive", help="Interactive mode for downloading models")
def interactive_mode():
    """Run in interactive mode with guided prompts."""
    rprint(
        Panel.fit(
            "[bold cyan]ModelHub Downloader[/bold cyan]\n"
            "[green]Download AI models from HuggingFace & ModelScope[/green]",
            border_style="cyan",
        )
    )

    # Get model ID
    model_id = Prompt.ask(
        "[bold yellow]?[/bold yellow] Enter model ID",
        default="Qwen/Qwen3-ASR-1.7B",
    ).strip()

    if not model_id:
        rprint("[red]❌ Model ID cannot be empty[/red]")
        raise typer.Exit(1)

    # Select source
    rprint("\n[bold yellow]?[/bold yellow] Select download source:")
    rprint("  1) [cyan]ModelScope[/cyan] (Recommended - faster in China)")
    rprint("  2) [cyan]HuggingFace[/cyan] (Global)")
    rprint("  3) [cyan]Auto-detect[/cyan] (Try HF first)")

    choice = Prompt.ask("Choose", choices=["1", "2", "3"], default="1")
    source_map = {"1": "ms", "2": "hf", "3": "auto"}
    source = source_map[choice]

    # Get output directory
    output = Prompt.ask(
        "[bold yellow]?[/bold yellow] Output directory",
        default=DEFAULT_OUTPUT,
    ).strip()

    # Confirm
    rprint("\n[blue]─[/blue]" * 40)
    rprint(f"[bold]Download Summary[/bold]")
    rprint(f"  Model:    {model_id}")
    rprint(f"  Source:   {source.upper()}")
    rprint(f"  Output:   {output}")
    rprint("[blue]─[/blue]" * 40)

    if not Confirm.ask("Start download?", default=True):
        rprint("[yellow]Cancelled[/yellow]")
        raise typer.Exit()

    # Download
    downloader = create_downloader(output_dir=output, source=source)
    success = downloader.download(model_id)

    if success:
        downloader.verify_download(model_id)
        rprint("\n[bold green]🎉 Download completed![/bold green]")
    else:
        rprint("\n[bold red]❌ Download failed[/bold red]")
        raise typer.Exit(1)


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
):
    """ModelHub Downloader - Download AI models from HuggingFace and ModelScope.

    ModelScope is the recommended source for users in China.
    No authentication required for public models.
    """
    pass


def run_cli():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    run_cli()
