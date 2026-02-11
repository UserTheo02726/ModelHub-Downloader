"""Progress bar utilities using Rich."""

from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from rich.console import Console

console = Console()


def create_download_progress() -> Progress:
    """Create a progress bar for downloads.

    Returns:
        Configured Progress instance
    """
    return Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def cleanup_progress() -> Progress:
    """Create a progress bar for cleanup operations.

    Returns:
        Configured Progress instance
    """
    return Progress(
        SpinnerColumn(style="yellow"),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    )
