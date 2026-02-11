"""ModelHub Downloader - Download models from HuggingFace & ModelScope.

Usage:
    python main.py download <model_id>
    python main.py clean --all
    python main.py (interactive mode)
"""

import os
import sys
import shutil
import re
import logging
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table

try:
    from huggingface_hub import snapshot_download, HfApi
except ImportError:
    print("Error: pip install huggingface_hub")
    sys.exit(1)

try:
    from modelscope import snapshot_download as ms_snapshot_download
except ImportError:
    print("Error: pip install modelscope")
    sys.exit(1)


# === 配置 ===
DEFAULT_OUTPUT = "./models"
SOURCE_HF = "hf"
SOURCE_MS = "ms"
SOURCE_AUTO = "auto"


# === 异常类 ===
class ModelDownloadError(Exception):
    """模型下载基异常"""

    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self):
        if self.details:
            return f"{self.message} (details: {self.details})"
        return self.message


class ValidationError(ModelDownloadError):
    """验证失败异常"""

    def __init__(self, field: str, value: str, reason: str):
        self.field = field
        self.value = value
        self.reason = reason
        super().__init__(
            f"Validation failed for '{field}': {reason}",
            {"field": field, "value": value, "reason": reason},
        )


class DownloadError(ModelDownloadError):
    """下载过程异常"""

    def __init__(self, model_id: str, source: str, original_error: Exception):
        self.model_id = model_id
        self.source = source
        self.original_error = original_error
        super().__init__(
            f"Download failed for {model_id} from {source}: {original_error}",
            {"model_id": model_id, "source": source},
        )


# === 路径验证器 ===
def path_validator(path: str) -> bool:
    """
    验证路径安全性

    规则:
    1. 不允许路径遍历 (../ 或 ..\)
    2. 不允许特殊字符 (空格、制表符等)
    3. 不允许绝对路径遍历

    Args:
        path: 待验证的路径

    Returns:
        bool: 路径是否安全
    """
    # 检查路径遍历
    if ".." in path.replace("\\", "/"):
        return False

    # 检查特殊字符
    if re.search(r"[\s\t\n\r]", path):
        return False

    # 检查绝对路径遍历
    normalized = path.replace("\\", "/")
    if normalized.startswith("/"):
        # 允许简单绝对路径，不允许遍历
        parts = [p for p in normalized.split("/") if p]
        if ".." in parts:
            return False

    return True


# === 日志配置 ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("ModelDownloader")


class ModelDownloader:
    def __init__(self, output_dir: str = DEFAULT_OUTPUT, source: str = SOURCE_MS):
        self.output_dir = Path(output_dir)
        self.source = source

    def validate(self, model_id: str) -> bool:
        return bool(model_id and "/" in model_id and len(model_id.split("/")) == 2)

    def get_path(self, model_id: str) -> Path:
        return self.output_dir / model_id.split("/")[-1]

    def _fmt_size(self, size: int) -> str:
        if size >= 1024**3:
            return f"{size / 1024**3:.1f} GB"
        if size >= 1024**2:
            return f"{size / 1024**2:.1f} MB"
        if size >= 1024:
            return f"{size / 1024:.1f} KB"
        return f"{size} B"

    def _get_size(self, model_id: str) -> Optional[int]:
        try:
            api = HfApi()
            info = api.model_info(model_id, files_metadata=True)
            return sum(
                s.lfs.get("size", 0) if isinstance(s.lfs, dict) else s.size or 0
                for s in info.siblings
            )
        except:
            return None

    def download_hf(self, model_id: str) -> bool:
        """
        从 HuggingFace 下载模型

        Args:
            model_id: 模型 ID (格式: org/model-name)

        Returns:
            bool: 下载是否成功

        Raises:
            DownloadError: 下载过程发生错误
        """
        target = self.get_path(model_id)
        rprint(f"\n[cyan]Download from HF:[/cyan] {model_id}")
        rprint(f"  [dim]→ {target}[/dim]")
        try:
            target.mkdir(parents=True, exist_ok=True)
            if size := self._get_size(model_id):
                rprint(f"  [dim]Size: {self._fmt_size(size)}[/dim]")
            snapshot_download(
                repo_id=model_id,
                local_dir=str(target),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            rprint(f"  [green]OK[/green]")
            logger.info(f"Successfully downloaded {model_id} from HF")
            return True
        except Exception as e:
            rprint(f"  [red]Error: {e}[/red]")
            logger.error(f"Download failed for {model_id} from HF", exc_info=True)
            raise DownloadError(model_id, "HF", e) from e

    def download_ms(self, model_id: str) -> bool:
        """
        从 ModelScope 下载模型

        Args:
            model_id: 模型 ID

        Returns:
            bool: 下载是否成功

        Raises:
            DownloadError: 下载过程发生错误
        """
        target = self.get_path(model_id)
        rprint(f"\n[cyan]Download from MS:[/cyan] {model_id}")
        rprint(f"  [dim]→ {target}[/dim]")
        try:
            target.mkdir(parents=True, exist_ok=True)
            ms_snapshot_download(model_id, local_dir=str(target))
            rprint(f"  [green]OK[/green]")
            logger.info(f"Successfully downloaded {model_id} from MS")
            return True
        except Exception as e:
            rprint(f"  [red]Error: {e}[/red]")
            logger.error(f"Download failed for {model_id} from MS", exc_info=True)
            raise DownloadError(model_id, "MS", e) from e

    def download(self, model_id: str) -> bool:
        """
        下载模型（自动选择源）

        Args:
            model_id: 模型 ID

        Returns:
            bool: 下载是否成功

        Raises:
            ValidationError: model_id 格式无效
        """
        # 验证
        if not self.validate(model_id):
            msg = f"Invalid model ID format: {model_id}"
            rprint(f"[red]{msg}[/red]")
            logger.warning(f"Validation failed: {model_id}")
            raise ValidationError("model_id", model_id, "Must be in 'org/model' format")

        if self.source == SOURCE_AUTO:
            rprint("[yellow]Auto mode: Try HF → MS[/yellow]")
            return self.download_hf(model_id) or self.download_ms(model_id)
        return (
            self.download_hf(model_id)
            if self.source == SOURCE_HF
            else self.download_ms(model_id)
        )

    def verify(self, model_id: str) -> bool:
        target = self.get_path(model_id)
        if not target.exists():
            rprint(f"[red]Directory not found: {target}[/red]")
            return False
        files = list(target.rglob("*"))
        count = sum(1 for f in files if f.is_file())
        if count > 0:
            size = sum(f.stat().st_size for f in files if f.is_file())
            rprint(f"\n[green]Verified:[/green] {count} files, {self._fmt_size(size)}")
            return True
        rprint(f"\n[red]Directory is empty[/red]")
        return False


# === CLI ===
app = typer.Typer(
    name="modelhub",
    help="ModelHub - HuggingFace & ModelScope 下载工具",
    add_completion=False,
)


@app.command("download")
def download_cmd(
    model_id: str = typer.Argument(..., help="Model ID, e.g. Qwen/Qwen3-ASR-1.7B"),
    source: str = typer.Option(
        "ms", "-s", "--source", help="hf=HF, ms=MS (default)", case_sensitive=False
    ),
    output: str = typer.Option(
        DEFAULT_OUTPUT, "-o", "--output", help="Output directory"
    ),
    verify: bool = typer.Option(True, "--verify/--no-verify", help="Verify download"),
):
    """Download a model"""
    d = ModelDownloader(output_dir=output, source=source)
    if d.download(model_id) and verify:
        d.verify(model_id)


@app.command("clean")
def clean_cmd(
    hf: bool = typer.Option(False, "--hf", help="Clean HF cache"),
    ms: bool = typer.Option(False, "--ms", help="Clean MS cache"),
    all: bool = typer.Option(False, "-a", "--all", help="Clean all caches"),
):
    """Clean cache directories"""
    hf_cache = Path.home() / ".cache" / "huggingface"
    ms_cache = Path.home() / ".cache" / "modelscope"
    if all or hf and hf_cache.exists():
        shutil.rmtree(hf_cache, ignore_errors=True)
        rprint("[green]HF cache cleaned[/green]")
    if all or ms and ms_cache.exists():
        shutil.rmtree(ms_cache, ignore_errors=True)
        rprint("[green]MS cache cleaned[/green]")
    if not hf and not ms and not all:
        rprint("[yellow]No cache to clean[/yellow]")


@app.command("list")
def list_cmd():
    """List available sources"""
    t = Table(title="Download Sources")
    t.add_column("Source")
    t.add_column("Status")
    t.add_row("ms (推荐)", "[green]OK[/green]")
    t.add_row("hf", "[green]OK[/green]")
    t.add_row("auto", "[green]OK[/green]")
    rprint(t)


@app.command("interactive")
def interactive_cmd():
    """Interactive mode"""
    rprint(Panel.fit("[bold cyan]ModelHub Downloader[/bold cyan]", border_style="cyan"))
    # Model ID
    model_id = Prompt.ask(
        "[yellow]?[/yellow] Model ID", default="Qwen/Qwen3-ASR-1.7B"
    ).strip()
    if not model_id:
        raise typer.Exit(1)
    # Source
    rprint("  1) [cyan]ModelScope[/cyan] (推荐)")
    rprint("  2) [cyan]HuggingFace[/cyan]")
    rprint("  3) [cyan]Auto[/cyan]")
    choice = Prompt.ask("Choose", choices=["1", "2", "3"], default="1")
    source = {"1": SOURCE_MS, "2": SOURCE_HF, "3": SOURCE_AUTO}[choice]
    # Output
    output = Prompt.ask("[yellow]?[/yellow] Output", default=DEFAULT_OUTPUT).strip()
    # Confirm - 单行分隔符
    rprint(f"\n[cyan]=== Summary ===[/cyan]")
    rprint(f"  Model:  {model_id}")
    rprint(f"  Source: {source.upper()}")
    rprint(f"  Output: {output}")
    rprint("[cyan]=============[/cyan]")
    if not Confirm.ask("Start download?", default=True):
        rprint("[yellow]Cancelled[/yellow]")
        raise typer.Exit()
    # Download
    d = ModelDownloader(output_dir=output, source=source)
    if d.download(model_id):
        d.verify(model_id)
        rprint("\n[bold green]Done[/bold green]")
    else:
        rprint("\n[bold red]Failed[/bold red]")
        raise typer.Exit(1)


@app.callback()
def main():
    """ModelHub - HuggingFace & ModelScope 下载工具"""
    pass


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        interactive_cmd()
    else:
        app()
