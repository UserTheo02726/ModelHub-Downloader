"""Core downloader module for HuggingFace and ModelScope.

This module provides download functionality for public models without requiring
authentication. It uses huggingface_hub and modelscope libraries directly.
"""

import os
import sys
from pathlib import Path
from typing import Optional
from enum import Enum

try:
    from huggingface_hub import snapshot_download, HfApi
except ImportError:
    print(
        "Error: 'huggingface_hub' is required. Install with: pip install huggingface_hub"
    )
    sys.exit(1)

try:
    from modelscope import snapshot_download as ms_snapshot_download
except ImportError:
    print("Error: 'modelscope' is required. Install with: pip install modelscope")
    sys.exit(1)


class Source(str, Enum):
    """Download source enum."""

    HUGGINGFACE = "hf"
    MODELSCOPE = "ms"
    AUTO = "auto"


class DownloadError(Exception):
    """Base exception for download errors."""

    pass


class ModelDownloader:
    """Model downloader supporting HuggingFace and ModelScope.

    This class handles downloading public models without requiring
    authentication. It uses the official libraries but bypasses
    authentication checks for public repositories.
    """

    SOURCE_HF = "hf"
    SOURCE_MS = "ms"
    SOURCE_AUTO = "auto"

    def __init__(
        self,
        output_dir: str = "./models",
        source: str = SOURCE_MS,  # ModelScope as default
        retry_count: int = 3,
        timeout: int = 300,
    ):
        """Initialize the downloader.

        Args:
            output_dir: Target directory for downloaded models
            source: Download source (hf/ms/auto)
            retry_count: Number of retry attempts on failure
            timeout: Timeout in seconds for each download
        """
        self.output_dir = Path(output_dir)
        self.source = source
        self.retry_count = retry_count
        self.timeout = timeout

        # Cache directories (respect environment variables)
        self.hf_cache_dir = Path(
            os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
        )
        self.ms_cache_dir = Path(
            os.environ.get("MODELSCOPE_CACHE", Path.home() / ".cache" / "modelscope")
        )

    def validate_model_id(self, model_id: str) -> bool:
        """Validate model ID format.

        Args:
            model_id: Model ID in format 'namespace/model_name'

        Returns:
            True if valid, False otherwise
        """
        if not model_id or "/" not in model_id:
            return False
        parts = model_id.split("/")
        return len(parts) == 2 and all(part.strip() for part in parts)

    def get_model_path(self, model_id: str) -> Path:
        """Get the target path for a model.

        Args:
            model_id: Model ID

        Returns:
            Target Path for the model
        """
        model_name = model_id.split("/")[-1]
        return self.output_dir / model_name

    def _format_size(self, size_bytes: int) -> str:
        """Format bytes to human-readable string."""
        if size_bytes >= 1024**3:
            return f"{size_bytes / (1024**3):.2f} GB"
        elif size_bytes >= 1024**2:
            return f"{size_bytes / (1024**2):.2f} MB"
        elif size_bytes >= 1024:
            return f"{size_bytes / 1024:.2f} KB"
        return f"{size_bytes} B"

    def get_model_size(self, model_id: str, source: str) -> Optional[int]:
        """Get model size from metadata.

        Args:
            model_id: Model ID
            source: Source (hf/ms)

        Returns:
            Total size in bytes or None
        """
        try:
            if source == self.SOURCE_HF:
                api = HfApi()
                info = api.model_info(model_id, files_metadata=True)
                total = 0
                for sibling in info.siblings:
                    lfs = sibling.lfs
                    if isinstance(lfs, dict):
                        total += lfs.get("size", 0)
                    else:
                        total += sibling.size or 0
                return total
        except Exception:
            pass
        return None

    def download_hf(self, model_id: str) -> bool:
        """Download from HuggingFace without authentication.

        Args:
            model_id: Model ID

        Returns:
            True on success, False on failure
        """
        from huggingface_hub import hf_hub_download
        import httpx

        target_path = self.get_model_path(model_id)

        print(f"\n[cyan]⬇️  Downloading from HuggingFace:[/cyan] {model_id}")
        print(f"   [dim]Target: {target_path}[/dim]")

        try:
            # Create directory if needed
            target_path.mkdir(parents=True, exist_ok=True)

            # Get model info
            try:
                size = self.get_model_size(model_id, self.SOURCE_HF)
                if size:
                    print(f"   [dim]Model size: {self._format_size(size)}[/dim]")
            except Exception:
                pass

            # Download using snapshot_download (no auth needed for public repos)
            snapshot_download(
                repo_id=model_id,
                local_dir=str(target_path),
                local_dir_use_symlinks=False,
                resume_download=True,
            )

            print(f"   [green]✅ Success: {target_path}[/green]")
            return True

        except httpx.HTTPError as e:
            print(f"   [red]❌ HTTP Error: {e}[/red]")
            return False
        except KeyboardInterrupt:
            print(f"\n   [yellow]⚠️  Download interrupted[/yellow]")
            raise
        except Exception as e:
            print(f"   [red]❌ Error: {e}[/red]")
            return False

    def download_ms(self, model_id: str) -> bool:
        """Download from ModelScope without authentication.

        Args:
            model_id: Model ID

        Returns:
            True on success, False on failure
        """
        import httpx

        target_path = self.get_model_path(model_id)

        print(f"\n[cyan]⬇️  Downloading from ModelScope:[/cyan] {model_id}")
        print(f"   [dim]Target: {target_path}[/dim]")

        try:
            # Create directory if needed
            target_path.mkdir(parents=True, exist_ok=True)

            # Get model info
            try:
                size = self.get_model_size(model_id, self.SOURCE_MS)
                if size:
                    print(f"   [dim]Model size: {self._format_size(size)}[/dim]")
            except Exception:
                pass

            # Download using ModelScope (no auth needed for public repos)
            ms_snapshot_download(
                model_id,
                local_dir=str(target_path),
            )

            print(f"   [green]✅ Success: {target_path}[/green]")
            return True

        except httpx.HTTPError as e:
            print(f"   [red]❌ HTTP Error: {e}[/red]")
            return False
        except KeyboardInterrupt:
            print(f"\n   [yellow]⚠️  Download interrupted[/yellow]")
            raise
        except Exception as e:
            print(f"   [red]❌ Error: {e}[/red]")
            return False

    def download(self, model_id: str) -> bool:
        """Download model from specified source.

        Args:
            model_id: Model ID

        Returns:
            True on success, False on failure
        """
        if not self.validate_model_id(model_id):
            print(f"[red]❌ Invalid model ID format: {model_id}[/red]")
            print("   [dim]Expected format: 'namespace/model_name'[/dim]")
            return False

        # Auto mode: try HF first, then MS
        if self.source == self.SOURCE_AUTO:
            print("[yellow]🔄 Auto mode: Trying HuggingFace first...[/yellow]")
            if self.download_hf(model_id):
                return True
            print("\n[yellow]⚠️  HuggingFace failed, trying ModelScope...[/yellow]")
            return self.download_ms(model_id)

        # Specific source
        if self.source == self.SOURCE_HF:
            return self.download_hf(model_id)
        elif self.source == self.SOURCE_MS:
            return self.download_ms(model_id)
        else:
            print(f"[red]❌ Unknown source: {self.source}[/red]")
            return False

    def verify_download(self, model_id: str) -> bool:
        """Verify downloaded files.

        Args:
            model_id: Model ID

        Returns:
            True if files exist, False otherwise
        """
        target_path = self.get_model_path(model_id)

        if not target_path.exists():
            print(f"[red]❌ Directory not found: {target_path}[/red]")
            return False

        files = list(target_path.rglob("*"))
        file_count = sum(1 for f in files if f.is_file())

        if file_count > 0:
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            print(
                f"\n[green]✅ Verification passed:[/green] {file_count} files, {self._format_size(total_size)}"
            )
            return True
        else:
            print(f"\n[red]❌ Verification failed: directory is empty[/red]")
            return False


def create_downloader(
    output_dir: str = "./models",
    source: str = "ms",  # ModelScope as default
    retry_count: int = 3,
) -> ModelDownloader:
    """Factory function to create a downloader instance.

    Args:
        output_dir: Target directory for models
        source: Download source (hf/ms/auto)
        retry_count: Retry attempts

    Returns:
        Configured ModelDownloader instance
    """
    return ModelDownloader(
        output_dir=output_dir,
        source=source,
        retry_count=retry_count,
    )
