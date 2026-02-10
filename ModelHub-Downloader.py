# -*- coding: utf-8 -*-

"""Requirements

- tqdm>=4.60
- rich>=13.0
- huggingface_hub>=0.20
- modelscope>=1.7
"""

"""
ModelHub Downloader - 公开模型快速下载工具
支持 HuggingFace 和 ModelScope 平台
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

# ──────── 依赖检查 ────────

try:
    import tqdm as _tqdm_module  # noqa: F401  验证 tqdm 已安装
except ImportError:
    print("请安装 tqdm: pip install tqdm")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.prompt import Prompt, Confirm
    from rich.panel import Panel
except ImportError:
    print("请安装 rich: pip install rich")
    sys.exit(1)

try:
    from huggingface_hub import snapshot_download, scan_cache_dir, HfApi
except ImportError:
    print("请安装 huggingface_hub: pip install huggingface_hub")
    sys.exit(1)

try:
    from modelscope import snapshot_download as ms_snapshot_download
except ImportError:
    print("请安装 modelscope: pip install modelscope")
    sys.exit(1)


def _format_size(size_bytes: int) -> str:
    """将字节数格式化为人类可读字符串"""
    if size_bytes >= 1024**3:
        return f"{size_bytes / (1024**3):.2f} GB"
    elif size_bytes >= 1024**2:
        return f"{size_bytes / (1024**2):.2f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KB"
    return f"{size_bytes} B"


class ModelDownloader:
    """模型下载器主类"""

    SOURCE_HF = "hf"
    SOURCE_MS = "ms"
    SOURCE_AUTO = "auto"

    def __init__(self):
        self.console = Console()
        self.logger = self._setup_logger()

        self.model_id: str = ""
        self.source: str = ""
        self.output_dir: str = ""
        self.non_interactive: bool = False

        # 缓存目录（尊重环境变量）
        self.hf_cache_dir = Path(
            os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
        )
        self.ms_cache_dir = Path(
            os.environ.get(
                "MODELSCOPE_CACHE", Path.home() / ".cache" / "modelscope"
            )
        )

    # ──────────────────────────── 基础工具 ────────────────────────────

    @staticmethod
    def _setup_logger() -> logging.Logger:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    @staticmethod
    def _validate_model_id_format(model_id: str) -> bool:
        """验证模型ID格式: 非空, 恰好两段, 每段非空"""
        if not model_id or "/" not in model_id:
            return False
        parts = model_id.split("/")
        return len(parts) == 2 and all(part.strip() for part in parts)

    def _get_model_dir(self) -> Path:
        """
        计算模型的本地存储目录。
        统一使用模型名（'/' 后半部分）作为目录名，
        与 ModelScope 默认行为保持一致。
        
        例: "Qwen/Qwen3-ASR-1.7B" → <output_dir>/Qwen3-ASR-1.7B
        """
        model_name = self.model_id.split("/")[-1]
        return Path(self.output_dir) / model_name

    # ──────────────────────────── 交互界面 ────────────────────────────

    def show_welcome(self):
        self.console.print(
            Panel.fit(
                "[bold blue]ModelHub-Downloader v1.3[/bold blue]\n"
                "[green]公开模型快速下载工具[/green]",
                border_style="blue",
            )
        )
        self.console.print(
            "支持的平台: [cyan]HuggingFace[/cyan] | [cyan]ModelScope[/cyan]\n"
        )

    def get_model_id(self) -> str:
        while True:
            self.model_id = Prompt.ask(
                "[bold yellow]?[/bold yellow] 请输入模型ID "
                "(例如: Qwen/Qwen3-ASR-1.7B)"
            ).strip()
            if not self.model_id:
                self.console.print("[red]ERROR 模型ID不能为空[/red]")
                continue
            if not self._validate_model_id_format(self.model_id):
                self.console.print(
                    "[red]ERROR 模型ID需为 'namespace/model-name' 格式[/red]"
                )
                continue
            break
        return self.model_id

    def select_source(self) -> str:
        self.console.print("[bold yellow]?[/bold yellow] 选择下载源:")
        self.console.print("  1) HuggingFace (huggingface.co)")
        self.console.print("  2) ModelScope  (modelscope.cn)")
        self.console.print("  3) 自动检测    (推荐)")

        choice = Prompt.ask("请选择", choices=["1", "2", "3"], default="3")
        source_map = {
            "1": self.SOURCE_HF,
            "2": self.SOURCE_MS,
            "3": self.SOURCE_AUTO,
        }
        self.source = source_map[choice]
        return self.source

    def get_output_dir(self) -> str:
        default_dir = "D:\\ComfyUI\\models" if os.name == "nt" else "./models"
        while True:
            self.output_dir = Prompt.ask(
                "[bold yellow]?[/bold yellow] 输入存储目录",
                default=default_dir,
            ).strip()
            path = Path(self.output_dir)
            if path.exists():
                if not path.is_dir():
                    self.console.print("[red]路径存在但不是目录[/red]")
                    continue
                break
            if self.non_interactive or Confirm.ask(
                "目录不存在，是否创建?", default=True
            ):
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    break
                except OSError as e:
                    self.console.print(f"[red]创建目录失败: {e}[/red]")
                    continue
        return self.output_dir

    def confirm_download(self) -> bool:
        if self.non_interactive:
            return True

        source_labels = {
            self.SOURCE_HF: "HuggingFace",
            self.SOURCE_MS: "ModelScope",
            self.SOURCE_AUTO: "自动检测",
        }

        self.console.print("\n" + "[blue]─[/blue]" * 50)
        self.console.print("[bold]下载信息确认[/bold]")
        self.console.print("[blue]─[/blue]" * 50)
        self.console.print(f"  模型ID:    {self.model_id}")
        self.console.print(
            f"  下载源:    {source_labels.get(self.source, self.source)}"
        )
        self.console.print(f"  目标目录:  {self._get_model_dir()}")
        return Confirm.ask("\n是否开始下载?")

    # ──────────────────────────── 验证与缓存 ────────────────────────────

    def _verify_download(self, model_dir: Path) -> bool:
        """验证下载目录的基本完整性"""
        self.console.print("\n[blue]正在验证下载完整性...[/blue]")

        file_count = 0
        total_size = 0
        for p in model_dir.rglob("*"):
            if p.is_file() and not p.name.startswith("."):
                file_count += 1
                total_size += p.stat().st_size

        self.console.print(f"  文件数量: {file_count}")
        self.console.print(f"  总大小:   {_format_size(total_size)}")

        if file_count > 0:
            self.console.print(
                "[green]✓ 验证通过[/green]"
            )
            return True
        else:
            self.console.print("[red]❌ 验证失败 (目录为空)[/red]")
            return False

    def _auto_clean_hf_cache(self, repo_id: str):
        """下载验证通过后，自动清理该 repo 的 HF 缓存"""
        self.console.print(
            f"[dim]正在自动清理 HuggingFace 缓存 ({repo_id})...[/dim]"
        )
        try:
            hf_cache_info = scan_cache_dir()
            for repo in hf_cache_info.repos:
                if repo.repo_id == repo_id:
                    revisions = [rev.commit_hash for rev in repo.revisions]
                    strategy = hf_cache_info.delete_revisions(*revisions)
                    strategy.execute()
                    self.console.print(
                        f"[green]✓ 缓存已自动清理 "
                        f"(释放 {strategy.expected_freed_size_str})[/green]"
                    )
                    return
            self.console.print("[dim]缓存中无残留，无需清理[/dim]")
        except Exception as e:
            self.console.print(
                f"[yellow]⚠️  自动清理缓存失败 (不影响已下载文件): {e}[/yellow]"
            )

    def _auto_clean_ms_cache(self, model_id: str):
        """下载验证通过后，自动清理该模型的 ModelScope 缓存"""
        # ModelScope 缓存结构: ~/.cache/modelscope/hub/<namespace>/<model>
        cache_model_dir = self.ms_cache_dir / "hub" / model_id.replace("/", os.sep)
        if not cache_model_dir.exists():
            self.console.print("[dim]ModelScope 缓存中无残留，无需清理[/dim]")
            return

        self.console.print(
            f"[dim]正在自动清理 ModelScope 缓存 ({model_id})...[/dim]"
        )
        try:
            cache_size = sum(
                f.stat().st_size
                for f in cache_model_dir.rglob("*")
                if f.is_file()
            )
            shutil.rmtree(cache_model_dir)
            self.console.print(
                f"[green]✓ 缓存已自动清理 "
                f"(释放 {_format_size(cache_size)})[/green]"
            )
        except Exception as e:
            self.console.print(
                f"[yellow]⚠️  自动清理缓存失败 (不影响已下载文件): {e}[/yellow]"
            )

    def clean_all_cache(self):
        """手动清理所有缓存（通过 --clean-cache 调用）"""
        self.console.print(
            "\n[bold red]⚠️  警告: 此操作将删除所有已下载的缓存！[/bold red]"
        )
        self.console.print(f"  HuggingFace 缓存: {self.hf_cache_dir}")
        self.console.print(f"  ModelScope 缓存:  {self.ms_cache_dir}")

        if not self.non_interactive:
            if not Confirm.ask("\n确认要清空所有缓存吗?", default=False):
                self.console.print("[yellow]操作已取消[/yellow]")
                return

        for name, cache_dir in [
            ("HuggingFace", self.hf_cache_dir),
            ("ModelScope", self.ms_cache_dir),
        ]:
            if not cache_dir.exists():
                self.console.print(f"[dim]{name} 缓存目录不存在，跳过[/dim]")
                continue
            try:
                self.console.print(f"[cyan]正在清理 {name} 缓存...[/cyan]")
                shutil.rmtree(cache_dir)
                self.console.print(f"[green]✓ {name} 缓存已清除[/green]")
            except OSError as e:
                self.console.print(
                    f"[red]❌ 清理 {name} 缓存失败: {e}[/red]"
                )

        self.console.print("\n[bold green]缓存清理完成！[/bold green]\n")

    # ──────────────────────────── 下载核心 ────────────────────────────
    #
    # 关键设计：不禁用 huggingface_hub / modelscope 的原生进度条。
    # 这两个库底层都使用 tqdm 显示每个文件的下载进度，
    # 直接让它们输出即可，无需自己扫描目录大小来模拟。
    #

    def _download_from_hf(self) -> bool:
        """从 HuggingFace 下载模型（使用库内置 tqdm 进度条）"""
        self.console.print("\n[green]▶ 正在从 HuggingFace 下载...[/green]")
        self.logger.info("开始下载 HuggingFace 模型 %s", self.model_id)

        try:
            model_dir = self._get_model_dir()

            if model_dir.exists():
                self.console.print(
                    f"[yellow]  目标目录已存在，将增量更新: {model_dir}[/yellow]"
                )
            else:
                self.console.print(
                    f"[cyan]  下载目标: {model_dir}[/cyan]"
                )
                model_dir.mkdir(parents=True, exist_ok=True)

            # 获取模型大小（可选，仅用于显示信息）
            try:
                api = HfApi()
                info = api.model_info(self.model_id, files_metadata=True)
                total_size = sum(
                    (s.lfs.get("size", 0) if s.lfs and isinstance(s.lfs, dict) else s.size or 0)
                    for s in info.siblings
                )
                if total_size > 0:
                    self.console.print(
                        f"[dim]  模型大小: {_format_size(total_size)}[/dim]"
                    )
            except Exception:
                pass  # 获取大小失败不影响下载

            self.console.print("")  # 空行，让进度条输出更清晰

            # ★ 核心：直接调用 snapshot_download
            #   huggingface_hub 内部使用 tqdm 逐文件显示进度条
            #   不设置 HF_HUB_DISABLE_PROGRESS_BARS，让原生进度条正常工作
            snapshot_download(
                repo_id=self.model_id,
                local_dir=str(model_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
            )

            self.console.print(
                f"\n[green]✅ HuggingFace 下载成功![/green]"
            )
            self.console.print(f"[green]   保存路径: {model_dir}[/green]")
            self.logger.info("HuggingFace 下载成功，路径 %s", model_dir)

            # 验证 → 通过后自动清理缓存
            if self._verify_download(model_dir):
                self._auto_clean_hf_cache(self.model_id)

            return True

        except KeyboardInterrupt:
            raise
        except Exception as e:
            self.console.print(
                f"\n[red]❌ HuggingFace 下载失败: {e}[/red]"
            )
            self.logger.error("HuggingFace 下载出错: %s", e, exc_info=True)
            return False

    def _download_from_ms(self) -> bool:
        """从 ModelScope 下载模型（使用库内置 tqdm 进度条）"""
        self.console.print("\n[green]▶ 正在从 ModelScope 下载...[/green]")
        self.logger.info("开始下载 ModelScope 模型 %s", self.model_id)

        try:
            model_dir = self._get_model_dir()

            if model_dir.exists():
                self.console.print(
                    f"[yellow]  目标目录已存在，将增量更新: {model_dir}[/yellow]"
                )
            else:
                self.console.print(
                    f"[cyan]  下载目标: {model_dir}[/cyan]"
                )
                model_dir.mkdir(parents=True, exist_ok=True)

            self.console.print("")  # 空行

            # ★ 核心：直接调用 ms_snapshot_download
            #   modelscope 内部使用 tqdm 显示进度条
            ms_snapshot_download(
                self.model_id,
                local_dir=str(model_dir),
            )

            self.console.print(
                f"\n[green]✅ ModelScope 下载成功![/green]"
            )
            self.console.print(f"[green]   保存路径: {model_dir}[/green]")
            self.logger.info("ModelScope 下载成功，路径 %s", model_dir)

            # 验证 → 通过后自动清理缓存
            if self._verify_download(model_dir):
                self._auto_clean_ms_cache(self.model_id)

            return True

        except KeyboardInterrupt:
            raise
        except Exception as e:
            self.console.print(
                f"\n[red]❌ ModelScope 下载失败: {e}[/red]"
            )
            self.logger.error("ModelScope 下载出错: %s", e, exc_info=True)
            return False

    def download(self) -> bool:
        """根据选定的下载源执行下载"""
        if not self._validate_model_id_format(self.model_id):
            self.console.print(
                "[red]ERROR 模型ID格式错误，需为 'namespace/model-name'[/red]"
            )
            return False

        if self.source == self.SOURCE_HF:
            return self._download_from_hf()
        elif self.source == self.SOURCE_MS:
            return self._download_from_ms()
        else:
            # 自动检测: 先 HF 后 MS
            self.console.print(
                "[yellow]自动检测模式: 优先尝试 HuggingFace[/yellow]"
            )
            if self._download_from_hf():
                return True
            self.console.print(
                "\n[yellow]HuggingFace 不可用，切换到 ModelScope...[/yellow]"
            )
            return self._download_from_ms()

    # ──────────────────────────── 主循环 ────────────────────────────

    def _ensure_params(self):
        """确保所有必需参数已就绪"""
        if not self.model_id:
            self.get_model_id()
        elif not self._validate_model_id_format(self.model_id):
            self.console.print(
                f"[red]模型ID格式无效: {self.model_id}[/red]"
            )
            if self.non_interactive:
                sys.exit(1)
            self.model_id = ""
            self.get_model_id()

        if not self.source:
            if self.non_interactive:
                self.source = self.SOURCE_AUTO
            else:
                self.select_source()

        if not self.output_dir:
            if self.non_interactive:
                self.output_dir = (
                    "D:\\ComfyUI\\models" if os.name == "nt" else "./models"
                )
                Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            else:
                self.get_output_dir()

    def run(self):
        """运行主程序"""
        try:
            self.show_welcome()

            while True:
                self._ensure_params()

                if not self.confirm_download():
                    self.console.print("[yellow]下载已取消[/yellow]")
                    break

                success = self.download()

                if success:
                    self.console.print(
                        f"\n[bold green]{'═' * 50}[/bold green]"
                    )
                    self.console.print(
                        "[bold green]  下载流程完成[/bold green]"
                    )
                    self.console.print(
                        f"[bold green]{'═' * 50}[/bold green]"
                    )

                    if self.non_interactive:
                        break

                    # 简化菜单：移除了缓存清理选项（已自动完成）
                    self.console.print(
                        "\n[bold yellow]?[/bold yellow] 下一步操作:"
                    )
                    self.console.print("  1) 继续下载其他模型")
                    self.console.print("  2) 更换输出目录后下载")
                    self.console.print("  3) 退出程序")

                    choice = Prompt.ask(
                        "请选择",
                        choices=["1", "2", "3"],
                        default="3",
                    )

                    if choice == "1":
                        self.model_id = ""
                    elif choice == "2":
                        self.model_id = ""
                        self.output_dir = ""
                    else:
                        self.console.print(
                            "[yellow]感谢使用，再见！[/yellow]"
                        )
                        break
                else:
                    self.console.print(
                        "\n[red]下载失败，请检查模型ID和网络连接[/red]"
                    )

                    if self.non_interactive:
                        sys.exit(1)

                    if Confirm.ask(
                        "\n[bold yellow]?[/bold yellow] 是否重新尝试?",
                        default=True,
                    ):
                        continue
                    else:
                        break

        except KeyboardInterrupt:
            self.console.print("\n[yellow]操作已取消[/yellow]")
            sys.exit(130)
        except EOFError:
            self.console.print("\n[yellow]输入结束，退出程序[/yellow]")


def main():
    parser = argparse.ArgumentParser(
        description="ModelHub Downloader - 公开模型下载工具",
        epilog="示例: %(prog)s -m Qwen/Qwen3-ASR-1.7B -s auto -o ./models",
    )
    parser.add_argument(
        "-m", "--model-id", help="模型ID (格式: namespace/model)"
    )
    parser.add_argument(
        "-s",
        "--source",
        choices=["hf", "ms", "auto"],
        help="下载源: hf=HuggingFace, ms=ModelScope, auto=自动检测",
    )
    parser.add_argument("-o", "--output-dir", help="输出目录")
    parser.add_argument(
        "--clean-cache",
        action="store_true",
        help="手动清理所有缓存并退出",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="跳过确认提示（非交互模式）",
    )

    args = parser.parse_args()
    downloader = ModelDownloader()

    # 仅清理缓存模式
    if args.clean_cache:
        downloader.non_interactive = args.yes
        downloader.clean_all_cache()
        return

    # 判断非交互模式
    has_all_params = all([args.model_id, args.source, args.output_dir])
    downloader.non_interactive = args.yes or has_all_params

    if args.model_id:
        downloader.model_id = args.model_id.strip()
    if args.source:
        downloader.source = args.source
    if args.output_dir:
        downloader.output_dir = args.output_dir.strip()

    downloader.run()


if __name__ == "__main__":
    main()