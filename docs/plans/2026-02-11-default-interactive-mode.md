# 无参数自动进入交互模式实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**目标:** 修改入口逻辑，使 `python main.py` 无参数时直接进入交互模式而不是显示帮助。

**架构:** 使用 Typer 的 `@app.callback(invoke_without_command=True)` 配置回调函数，在 `ctx.invoked_subcommand is None` 时调用 `interactive_mode()`。

**技术栈:** Python 3.8+, Typer, Rich

---

## 背景分析

### 当前问题
- **main.py 第 250-256 行**: `@app.callback()` 没有任何参数配置
- **无参数行为**: 默认显示帮助信息而不是进入交互模式
- **用户期望**: 运行 `python main.py` 直接进入交互模式

### Typer 解决方案
```python
@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """CLI入口回调"""
    if ctx.invoked_subcommand is None:
        interactive_mode()  # 无参数时进入交互模式
```

---

## Task 1: 修改 @app.callback() 装饰器添加 invoke_without_command=True

**文件:**
- 修改: `main.py:250-256`

**Step 1: 查看当前代码**

```python
@app.callback()
def main():
    """ModelHub Downloader - Download AI models from HuggingFace and ModelScope.

    ModelScope is the recommended source for users in China.
    No authentication required for public models.
    """
```

**Step 2: 修改回调函数**

```python
from typing import Optional


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        is_flag=True,
        callback=version_callback,
        help="显示版本信息",
    ),
):
    """ModelHub Downloader - Download AI models from HuggingFace and ModelScope.

    无参数时自动进入交互模式。
    ModelScope is the recommended source for users in China.
    No authentication required for public models.
    """
    # 无子命令时进入交互模式
    if ctx.invoked_subcommand is None:
        interactive_mode()
```

**Step 3: 运行 LSP 诊断验证**

运行: `lsp_diagnostics file="D:\TRAE_Script\ModelHub-Downloader\main.py"`
预期: 无错误

**Step 4: 提交**

```bash
git add main.py
git commit -m "feat: add invoke_without_command=True for automatic interactive mode"
```

---

## Task 2: 添加 --version 支持到回调函数

**文件:**
- 修改: `main.py:34-39` (version_callback)
- 修改: `main.py:250-268` (main 回调)

**Step 1: 查看当前 version_callback**

```python
def version_callback(value: bool):
    """Print version and exit."""
    if value:
        rprint("[bold cyan]ModelHub Downloader[/bold cyan] v2.0.0")
        rprint("[dim]Built with Typer + Rich[/dim]")
        raise typer.Exit()
```

**Step 2: 在回调函数中添加 --version 选项**

修改 `main.py` 的 `@app.callback()` 装饰器：

```python
from typing import Optional


@app.callback(
    invoke_without_command=True,
    epilog="无参数时自动进入交互模式。",
)
def main(
    ctx: typer.Context,
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        is_flag=True,
        help="显示版本信息并退出",
    ),
):
    """ModelHub Downloader - Download AI models from HuggingFace and ModelScope.

    无参数时自动进入交互模式。
    ModelScope is the recommended source for users in China.
    No authentication required for public models.
    """
    # 处理 --version 参数
    if version:
        rprint("[bold cyan]ModelHub Downloader[/bold cyan] v2.0.0")
        rprint("[dim]Built with Typer + Rich[/dim]")
        raise typer.Exit()

    # 无子命令时进入交互模式
    if ctx.invoked_subcommand is None:
        interactive_mode()
```

**Step 3: 验证 --version 参数**

运行: `python main.py --version`
预期输出:
```
ModelHub Downloader v2.0.0
Built with Typer + Rich
```

**Step 4: 验证无参数行为**

运行: `python main.py` (需要模拟用户输入来验证进入交互模式)

预期: 应该显示欢迎信息并提示输入模型 ID

**Step 5: 提交**

```bash
git add main.py
git commit -m "feat: add --version flag and default interactive mode"
```

---

## Task 3: 更新 README.md 文档

**文件:**
- 修改: `README.md`

**Step 1: 更新使用方法**

```markdown
## 使用方法

### 🚀 交互式模式 (推荐)

直接运行脚本，无需任何参数：

```bash
python main.py
```

**行为变化：**
- 无参数时自动进入交互模式
- 支持 Ctrl+C 安全退出
- 下载完成后询问是否继续

### 🛠️ 命令行模式 (自动化)

适合脚本调用或熟练用户：

```bash
# 查看版本
python main.py --version

# 下载模型
python main.py download Qwen/Qwen3-ASR-1.7B --source ms --output ./models

# 清理缓存
python main.py clean --all
```
```

**Step 2: 提交**

```bash
git add README.md
git commit -m "docs: update README for automatic interactive mode"
```

---

## Task 4: 添加测试验证入口行为

**文件:**
- 新建: `tests/test_entry_point.py`

**Step 1: 创建测试文件**

```python
"""Tests for CLI entry point behavior."""
from typer.testing import CliRunner
from unittest.mock import patch
from io import StringIO


def test_main_without_args_enters_interactive_mode():
    """Test that main.py without args enters interactive mode."""
    from main import app
    runner = CliRunner()

    # Mock inputs: model_id=Qwen/Qwen2.5-0.5B, source=1, output=./models, continue=n
    with patch("sys.stdin", StringIO("1\n\nn\n")):
        result = runner.invoke(app, [])
        # Should show welcome message and enter interactive flow
        assert "ModelHub Downloader" in result.output
        assert "Enter model ID" in result.output


def test_main_with_version_flag():
    """Test that main.py --version shows version info."""
    from main import app
    runner = CliRunner()
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "ModelHub Downloader" in result.output
    assert "v2.0.0" in result.output


def test_main_with_download_command():
    """Test that main.py download works normally."""
    from main import app
    runner = CliRunner()
    result = runner.invoke(app, ["download", "Qwen/Qwen2.5-0.5B"])
    # Should not enter interactive mode
    assert "Enter model ID" not in result.output
```

**Step 2: 运行测试**

运行: `pytest tests/test_entry_point.py -v`
预期: 全部通过

**Step 3: 提交**

```bash
git add tests/test_entry_point.py
git commit -m "test: add entry point tests for automatic interactive mode"
```

---

## 验证清单

- [ ] `python main.py` 直接进入交互模式（无参数时）
- [ ] `python main.py --version` 显示版本信息
- [ ] `python main.py download Qwen/Qwen3-ASR-1.7B` 正常执行下载命令
- [ ] `python main.py --help` 显示帮助信息
- [ ] pytest tests/test_entry_point.py 全部通过
- [ ] README.md 文档已更新

---

## 行为变化对照表

| 命令 | 修改前 | 修改后 |
|-----|-------|-------|
| `python main.py` | 显示帮助菜单 | 进入交互模式 |
| `python main.py --version` | 显示帮助 | 显示版本信息 |
| `python main.py download X` | 正常下载 | 正常下载 |
| `python main.py --help` | 显示帮助 | 显示帮助 |

---

## 相关文档

- **Typer Context**: https://typer.tiangolo.com/tutorial/commands/context/
- **Typer Callback**: https://typer.tiangolo.com/tutorial/commands/callback/
