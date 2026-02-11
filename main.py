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
    from huggingface_hub.errors import GatedRepoError
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


# === 下载配置 ===
from dataclasses import dataclass


@dataclass
class DownloadConfig:
    """下载配置数据类"""

    model_id: str = ""
    source: str = SOURCE_MS
    output_dir: str = DEFAULT_OUTPUT

    def is_complete(self) -> bool:
        """
        检查配置是否完整

        规则: model_id 非空且包含 "/"
        """
        return bool(self.model_id and self.model_id.strip() and "/" in self.model_id)

    def validate(self) -> bool:
        """
        验证配置有效性

        Returns:
            bool: True 表示有效
        """
        return self.is_complete()

    def reset(self):
        """重置为默认值"""
        self.model_id = ""
        self.source = SOURCE_MS
        self.output_dir = DEFAULT_OUTPUT


# === 认证检查器 ===
from typing import Tuple
from httpx import HTTPStatusError


class AuthChecker:
    """
    认证状态检查器

    在下载前检查模型是否需要认证
    使用 HuggingFace 官方异常类型进行精确检测
    """

    def check_model_access(self, model_id: str, source: str) -> Tuple[bool, str]:
        """
        检查模型访问权限

        Args:
            model_id: 模型 ID
            source: 下载源 (hf/ms/auto)

        Returns:
            Tuple[需要认证, 错误信息]
        """
        # 只检查 HF 源
        if source != "hf":
            return False, ""

        try:
            # 轻量级 API 调用
            HfApi().model_info(model_id, files_metadata=False, timeout=10)
            return False, ""
        except GatedRepoError as e:
            return True, str(e)
        except HTTPStatusError as e:
            if e.response.status_code == 401:
                return True, f"401 Unauthorized: 可能需要认证或已达到下载限速"
            return False, ""
        except Exception:
            # 网络错误、超时等不视为认证问题
            return False, ""


# === 下载会话 ===
class DownloadSession:
    """
    下载会话管理类

    负责协调整个交互式下载流程:
    - 主菜单显示和导航
    - 下载配置流程
    - 下载执行
    - 下载后继续选项
    """

    def __init__(self):
        self.config = DownloadConfig()
        self.download_history: list[str] = []

    def add_to_history(self, model_id: str):
        """
        添加模型到下载历史

        Args:
            model_id: 成功下载的模型 ID
        """
        self.download_history.append(model_id)

    def show_main_menu(self) -> str:
        """
        显示主菜单并获取用户选择

        Returns:
            str: 用户选择的菜单项 ("1", "2", "3", "4")
        """
        from rich.prompt import Prompt
        from rich.panel import Panel

        menu = Panel.fit(
            "[bold cyan]ModelHub Downloader[/bold cyan]\n\n"
            "  1) 开始下载\n"
            "  2) 查看当前配置\n"
            "  3) 清理缓存\n"
            "  4) 退出",
            border_style="cyan",
        )
        rprint(menu)

        return Prompt.ask("选择", choices=["1", "2", "3", "4"], default="1")

    def step_input(
        self, title: str, prompt: str, default: str = "", validate=None
    ) -> str | None:
        """
        通用步骤输入方法（两步确认）

        Args:
            title: 步骤标题（显示在面板顶部）
            prompt: 提示文本
            default: 默认值
            validate: 验证函数，返回 True 表示有效

        Returns:
            str: 用户确认的输入值
            None: 用户选择返回
            (永不返回): 用户选择取消
        """
        from rich.prompt import Prompt
        from rich.panel import Panel

        # 构建步骤面板
        panel = Panel.fit(
            f"[bold]Step {title}[/bold]\n\n"
            f"? [yellow]{prompt}[/yellow]: [dim]{default}[/dim]\n\n"
            "[b] 返回  [c] 取消",
            border_style="cyan",
        )
        rprint(panel)

        # 获取用户输入
        value = Prompt.ask(prompt, default=default).strip()

        # 处理特殊输入
        if value.lower() == "b":
            rprint("[yellow]已返回[/yellow]")
            return None

        if value.lower() == "c":
            rprint("[yellow]已取消[/yellow]")
            raise typer.Exit(0)

        # 验证输入
        if validate and not validate(value):
            rprint("[red]输入无效，请重新输入[/red]")
            return self.step_input(title, prompt, default, validate)

        return value

    def show_config_summary(self) -> str:
        """
        显示配置汇总页

        Returns:
            str: 用户操作
            - "": 开始下载
            - "1"/"2"/"3": 修改对应配置项
            - "m": 返回主菜单
            - "q": 退出
        """
        from rich.prompt import Prompt
        from rich.table import Table

        # 创建配置表格
        t = Table(title="下载配置", show_header=False, box=None)
        t.add_column("选项", style="cyan", width=18)
        t.add_column("值", style="green")

        # Model ID
        model_display = self.config.model_id or "[未设置]"
        t.add_row("1) Model ID", model_display)

        # Source
        source_display_map = {
            "ms": "ModelScope (推荐)",
            "hf": "HuggingFace",
            "auto": "Auto (自动尝试)",
        }
        source_display = source_display_map.get(self.config.source, self.config.source)
        t.add_row("2) Source", source_display)

        # Output
        t.add_row("3) Output", self.config.output_dir)

        rprint("\n")
        rprint(t)
        rprint("\n")

        # 返回操作
        return Prompt.ask(
            "[Enter] 开始下载  [1/2/3] 修改  [m] 主菜单  [q] 退出", default=""
        )

    def run_download_setup(self) -> bool:
        """
        运行下载配置流程

        Returns:
            bool: True 表示完成配置并准备下载，False 表示用户返回
        """
        from rich.prompt import Prompt
        from rich import print as rprint

        # 进入配置循环
        while True:
            action = self.show_config_summary()

            if action == "":
                # 开始下载
                if self.config.validate():
                    return True
                else:
                    rprint("[red]配置无效，请检查[/red]")

            elif action == "1":
                # 修改 Model ID
                new_id = self.step_input(
                    "Model ID",
                    "输入 Model ID",
                    default=self.config.model_id or "",
                    validate=lambda x: bool(x and "/" in x),
                )
                if new_id is not None:
                    self.config.model_id = new_id

            elif action == "2":
                # 修改 Source
                rprint("\n  1) [cyan]ModelScope[/cyan] (推荐)")
                rprint("  2) [cyan]HuggingFace[/cyan]")
                rprint("  3) [cyan]Auto[/cyan]")
                choice = Prompt.ask("选择", choices=["1", "2", "3"], default="1")
                source_map = {"1": SOURCE_MS, "2": SOURCE_HF, "3": SOURCE_AUTO}
                self.config.source = source_map[choice]

            elif action == "3":
                # 修改 Output
                new_output = self.step_input(
                    "Output",
                    "输入输出目录",
                    default=self.config.output_dir or DEFAULT_OUTPUT,
                    validate=lambda x: bool(x.strip()),
                )
                if new_output is not None:
                    self.config.output_dir = new_output

            elif action == "m":
                # 返回主菜单
                return False

            elif action.lower() == "q":
                # 退出
                raise typer.Exit(0)

    def execute_download(self) -> bool:
        """
        执行下载

        Returns:
            bool: True 表示下载成功，False 表示失败
        """
        from rich import print as rprint

        d = ModelDownloader(
            output_dir=self.config.output_dir or DEFAULT_OUTPUT,
            source=self.config.source,
        )

        if d.download(self.config.model_id):
            d.verify(self.config.model_id)
            self.add_to_history(self.config.model_id)
            return True
        return False

    def show_current_config(self):
        """显示当前配置"""
        from rich.table import Table

        t = Table(title="当前配置", show_header=False, box=None)
        t.add_column("选项", style="cyan", width=18)
        t.add_column("值", style="green")

        model_display = self.config.model_id or "[未设置]"
        t.add_row("Model ID", model_display)

        source_display_map = {
            "ms": "ModelScope (推荐)",
            "hf": "HuggingFace",
            "auto": "Auto (自动尝试)",
        }
        source_display = source_display_map.get(self.config.source, self.config.source)
        t.add_row("Source", source_display)

        t.add_row("Output", self.config.output_dir or DEFAULT_OUTPUT)

        rprint("\n")
        rprint(t)
        rprint("\n")

    def run_clean_cache(self):
        """清理缓存"""
        from rich.prompt import Prompt

        choice = Prompt.ask(
            "清理所有缓存? ([y] 是  [n] 否)", choices=["y", "n"], default="n"
        )
        if choice == "y":
            _do_clean_cache(clean_all=True)
            rprint("[green]缓存已清理[/green]")

    def show_continue_menu(self) -> str:
        """
        显示下载后继续菜单

        Returns:
            str: 用户选择的操作
        """
        from rich.prompt import Prompt
        from rich.panel import Panel

        menu = Panel.fit(
            "[bold cyan]下载完成[/bold cyan]\n\n"
            "  1) 继续下载其他模型\n"
            "  2) 修改当前配置后下载\n"
            "  3) 查看下载历史\n"
            "  4) 退出",
            border_style="green",
        )
        rprint(menu)

        return Prompt.ask("选择", choices=["1", "2", "3", "4"], default="1")

    def ask_reuse_config(self) -> str:
        """
        询问是否复用当前配置

        Returns:
            str: "y" 复用，"n" 新建，"m" 主菜单
        """
        from rich import print as rprint
        from rich.prompt import Prompt

        rprint("[yellow]检测到已有配置，是否复用?[/yellow]")
        choice = Prompt.ask(
            "([y] 复用  [n] 新建  [m] 主菜单)",
            choices=["y", "n", "m"],
            default="y",
        )

        return choice


# === 异常类 ===
class ModelDownloadError(Exception):
    """模型下载基异常"""

    def __init__(self, message: str, details: dict | None = None):
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
    1. 不允许路径遍历 (../ 或 ..\\)
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


# === 模型 ID 验证器 ===
def model_id_validator(model_id: str) -> bool:
    """
    验证模型 ID 格式

    规则:
    1. 必须包含 exactly one "/"
    2. 组织名和模型名只能包含字母、数字、连字符、下划线、点
    3. 长度限制 (组织 <= 100, 模型 <= 200)
    4. 不能以连字符或点开头/结尾

    Args:
        model_id: 待验证的模型 ID

    Returns:
        bool: 格式是否有效
    """
    # 空值检查
    if not model_id or not isinstance(model_id, str):
        return False

    # 必须包含 exactly one "/"
    parts = model_id.split("/")
    if len(parts) != 2:
        return False

    org, model = parts

    # 长度限制
    if len(org) > 100 or len(model) > 200:
        return False

    # 不能为空
    if not org or not model:
        return False

    # 字符白名单 (字母、数字、连字符、下划线、点)
    valid_pattern = r"^[a-zA-Z0-9_\-\.]+$"
    if not (re.match(valid_pattern, org) and re.match(valid_pattern, model)):
        return False

    # 不能以连字符或点开头/结尾
    if org[0] in ("-", ".") or org[-1] in ("-", "."):
        return False
    if model[0] in ("-", ".") or model[-1] in ("-", "."):
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
        """验证模型 ID 格式"""
        return model_id_validator(model_id)

    def get_path(self, model_id: str) -> Path:
        """
        获取模型下载目标路径

        Args:
            model_id: 模型 ID

        Returns:
            Path: 下载目标路径

        Raises:
            ValidationError: 输出路径不安全
        """
        # 验证 model_id
        if not self.validate(model_id):
            raise ValidationError("model_id", model_id, "Invalid format")

        # 获取模型名
        model_name = model_id.split("/")[-1]

        # 构建完整路径
        full_path = self.output_dir / model_name

        # 验证路径安全
        normalized = full_path.resolve()

        # 获取输出目录的绝对路径
        output_dir_resolved = self.output_dir.resolve()

        # 确保目标路径在输出目录内
        try:
            normalized.relative_to(output_dir_resolved)
        except ValueError:
            raise ValidationError(
                "output_dir", str(full_path), "Path traversal not allowed"
            )

        # 额外检查相对路径中的 ".."
        path_str = str(full_path).replace("\\", "/")
        if "/../" in path_str or path_str.startswith("../") or path_str.endswith("/.."):
            raise ValidationError(
                "output_dir", str(full_path), "Path traversal not allowed"
            )

        return full_path

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
            if not info.siblings:
                return 0
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
                local_dir_use_symlinks=False,  # type: ignore[arg-type]
            )
            rprint(f"  [green]OK[/green]")
            logger.info(f"Successfully downloaded {model_id} from HF")
            return True
        except GatedRepoError as e:
            # Gated model 需要用户认证
            rprint(f"\n[red]错误: 模型 {model_id} 是受限制模型 (Gated Model)[/red]")
            rprint("[yellow]解决方案:[/yellow]")
            rprint(f"  1. 访问 https://huggingface.co/{model_id}")
            rprint("  2. 登录 HuggingFace 账户并接受使用协议")
            rprint("  3. 设置环境变量 HF_TOKEN 或运行: huggingface-cli login")
            raise DownloadError(
                model_id,
                "HF",
                Exception(f"Gated model: {model_id}. Please authenticate first."),
            ) from e
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


def _do_clean_cache(
    clean_all: bool = False, clean_hf: bool = False, clean_ms: bool = False
):
    """
    执行缓存清理

    Args:
        clean_all: 清理所有缓存
        clean_hf: 只清理 HuggingFace 缓存
        clean_ms: 只清理 ModelScope 缓存
    """
    hf_cache = Path.home() / ".cache" / "huggingface"
    ms_cache = Path.home() / ".cache" / "modelscope"
    if clean_all or clean_hf and hf_cache.exists():
        shutil.rmtree(hf_cache, ignore_errors=True)
        rprint("[green]HF cache cleaned[/green]")
    if clean_all or clean_ms and ms_cache.exists():
        shutil.rmtree(ms_cache, ignore_errors=True)
        rprint("[green]MS cache cleaned[/green]")
    if not clean_all and not clean_hf and not clean_ms:
        rprint("[yellow]No cache to clean[/yellow]")


@app.command("clean")
def clean_cmd(
    hf: bool = typer.Option(False, "--hf", help="Clean HF cache"),
    ms: bool = typer.Option(False, "--ms", help="Clean MS cache"),
    all: bool = typer.Option(False, "-a", "--all", help="Clean all caches"),
):
    """Clean cache directories"""
    _do_clean_cache(clean_all=all, clean_hf=hf, clean_ms=ms)


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
    """
    交互式下载模式

    使用 DownloadSession 管理整个交互流程:
    - 主菜单导航
    - 下载配置和执行
    - 下载后继续选项
    """
    session = DownloadSession()

    try:
        while True:
            choice = session.show_main_menu()

            if choice == "1":
                # 开始下载流程
                if session.run_download_setup():
                    rprint(f"正在下载 {session.config.model_id}...")
                    if session.execute_download():
                        rprint("\n[bold green]下载完成![/bold green]")

                        # 显示继续菜单
                        while True:
                            cont_choice = session.show_continue_menu()

                            if cont_choice == "1":
                                # 继续下载 - 重置配置
                                session.config.reset()
                                break
                            elif cont_choice == "2":
                                # 修改配置后下载
                                if session.run_download_setup():
                                    rprint(f"正在下载 {session.config.model_id}...")
                                    if session.execute_download():
                                        rprint("\n[bold green]下载完成![/bold green]")
                                    else:
                                        rprint("\n[bold red]下载失败[/bold red]")
                            elif cont_choice == "3":
                                # 查看下载历史
                                if session.download_history:
                                    rprint("\n下载历史:")
                                    for i, model_id in enumerate(
                                        session.download_history, 1
                                    ):
                                        rprint(f"  {i}) {model_id}")
                                else:
                                    rprint("\n暂无下载历史")
                            else:
                                # 退出
                                rprint("[yellow]再见！[/yellow]")
                                sys.exit(0)

            elif choice == "2":
                # 查看当前配置
                session.show_current_config()

            elif choice == "3":
                # 清理缓存
                session.run_clean_cache()

            elif choice == "4":
                # 退出
                rprint("[yellow]再见！[/yellow]")
                sys.exit(0)

    except (KeyboardInterrupt, typer.Exit):
        rprint("\n[yellow]再见！[/yellow]")
        sys.exit(0)


@app.callback()
def main():
    """ModelHub - HuggingFace & ModelScope 下载工具"""
    pass


if __name__ == "__main__":
    import sys

    try:
        if len(sys.argv) == 1:
            interactive_cmd()
        else:
            app()
    except KeyboardInterrupt:
        # 优雅处理用户中断 (Ctrl+C)
        rprint("\n[yellow]Cancelled by user[/yellow]")
        sys.exit(0)
