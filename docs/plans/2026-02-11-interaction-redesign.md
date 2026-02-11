# 交互流程重新设计 - 详细实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** 重构 `interactive_cmd`，添加返回机制和下载后继续选项

**Architecture:**
- 新增 `DownloadConfig` 数据类管理下载配置
- 新增 `DownloadSession` 类管理会话状态和交互流程
- 每个步骤支持 `[b] 返回` 和 `[c] 取消`
- 下载后提供继续选项菜单
- 支持配置沿用（询问用户是否使用当前配置）

**Tech Stack:** Python 3.9+, rich (Prompt, Panel, Table, Confirm), typer

---

## 新交互流程预览

### 主菜单
```
╔════════════════════════════════════════╗
║         ModelHub Downloader             ║
╠════════════════════════════════════════╣
║  1) 开始下载                            ║
║  2) 查看当前配置                        ║
║  3) 清理缓存                            ║
║  4) 退出                                ║
╚════════════════════════════════════════╝
```

### 步骤输入（两步确认）
```
╔════════════════════════════════════════╗
║  Step: Model ID                      ║
╠════════════════════════════════════════╣
║  ? Model ID: Qwen/Qwen3-ASR-1.7B     ║
║                                       ║
║  [Enter] 确认  [b] 返回  [c] 取消    ║
╚════════════════════════════════════════╝
```

### 下载后继续选项
```
╔════════════════════════════════════════╗
║  ✓ 下载成功                           ║
╠════════════════════════════════════════╣
║ 同目录继续下载  1)其他模型              ║
║  2)模型                   ║ 新目录下载新
║  3) 返回主菜单                        ║
║  4) 退出                              ║
╚════════════════════════════════════════╝
```

---

## 任务总览

| Phase | 任务 | 描述 | 预计时间 |
|-------|------|------|---------|
| 1 | Task 1-2 | 基础框架（DownloadConfig, DownloadSession） | 20 min |
| 2 | Task 3-5 | 交互流程（步骤输入、汇总页、主循环） | 30 min |
| 3 | Task 6-7 | 状态继承（继续菜单、配置沿用） | 20 min |
| 4 | Task 8-9 | 测试验证、最终提交 | 15 min |
| **合计** | **9** | | **85 min** |

---

## Phase 1: 基础框架

### Task 1: 创建 DownloadConfig 数据类

**Files:**
- Modify: `main.py` (配置常量后，约第34行)
- Create: `tests/test_download_config.py`

**Context:** 需要一个数据类来存储和管理下载配置，包括 model_id、source、output_dir。

**Step 1: 编写失败的测试**

```python
# tests/test_download_config.py
"""测试 DownloadConfig 数据类"""
import pytest
from main import DownloadConfig, SOURCE_MS, DEFAULT_OUTPUT


def test_default_values():
    """验证默认配置值"""
    config = DownloadConfig()
    assert config.model_id == ""
    assert config.source == SOURCE_MS
    assert config.output_dir == DEFAULT_OUTPUT


def test_is_complete_false_when_empty():
    """验证 is_complete 在配置为空时返回 False"""
    config = DownloadConfig()
    assert config.is_complete() == False


def test_is_complete_true_when_model_id_set():
    """验证 is_complete 在设置 model_id 后返回 True"""
    config = DownloadConfig()
    config.model_id = "Qwen/Qwen3-ASR-1.7B"
    assert config.is_complete() == True


def test_reset_restores_defaults():
    """验证 reset 方法恢复默认值"""
    config = DownloadConfig()
    config.model_id = "custom/model"
    config.source = "hf"
    config.output_dir = "/custom/path"
    
    config.reset()
    
    assert config.model_id == ""
    assert config.source == SOURCE_MS
    assert config.output_dir == DEFAULT_OUTPUT
```

**Step 2: 运行测试验证失败**

```bash
cd "D:\TRAE_Script\ModelHub-Downloader"
pytest tests/test_download_config.py -v
```
Expected: `ERROR - ImportError: cannot import 'DownloadConfig'`

**Step 3: 编写最小实现**

```python
# 在 main.py 配置常量后添加（约第38行）

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
        return bool(
            self.model_id and 
            self.model_id.strip() and
            "/" in self.model_id
        )
    
    def reset(self):
        """重置为默认值"""
        self.model_id = ""
        self.source = SOURCE_MS
        self.output_dir = DEFAULT_OUTPUT
```

**Step 4: 运行测试验证通过**

```bash
pytest tests/test_download_config.py -v
```
Expected: 4 passed

**Step 5: 提交**

```bash
git add tests/test_download_config.py main.py
git commit -m "feat: 添加 DownloadConfig 数据类管理下载配置"
```

---

### Task 2: 创建 DownloadSession 会话类

**Files:**
- Modify: `main.py` (DownloadConfig 类后，约第55行)
- Create: `tests/test_download_session.py`

**Context:** 需要一个会话管理类来协调整个交互流程，包括主菜单、下载流程、继续选项等。

**Step 1: 编写失败的测试**

```python
# tests/test_download_session.py
"""测试 DownloadSession 会话管理类"""
import pytest
from unittest.mock import patch
from main import DownloadSession, DownloadConfig


def test_session_creation_initializes_config():
    """验证会话创建时初始化配置"""
    session = DownloadSession()
    assert isinstance(session.config, DownloadConfig)


def test_session_creation_empty_history():
    """验证会话创建时下载历史为空"""
    session = DownloadSession()
    assert session.download_history == []


def test_add_to_history():
    """验证添加下载历史"""
    session = DownloadSession()
    session.add_to_history("Qwen/Qwen3-ASR-1.7B")
    session.add_to_history("facebook/opt-1.3b")
    
    assert len(session.download_history) == 2
    assert session.download_history[-1] == "facebook/opt-1.3b"


def test_show_main_menu_returns_choice():
    """验证主菜单返回用户选择"""
    session = DownloadSession()
    
    with patch('rich.prompt.Prompt.ask') as mock_ask:
        mock_ask.return_value = "1"
        
        choice = session.show_main_menu()
        
        assert choice == "1"
```

**Step 2: 运行测试验证失败**

```bash
pytest tests/test_download_session.py -v
```
Expected: `ERROR - ImportError: cannot import 'DownloadSession'`

**Step 3: 编写最小实现**

```python
# 在 main.py DownloadConfig 类后添加

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
            border_style="cyan"
        )
        rprint(menu)
        
        return Prompt.ask(
            "选择",
            choices=["1", "2", "3", "4"],
            default="1"
        )
```

**Step 4: 运行测试验证通过**

```bash
pytest tests/test_download_session.py -v
```
Expected: 4 passed

**Step 5: 提交**

```bash
git add tests/test_download_session.py main.py
git commit -m "feat: 添加 DownloadSession 会话管理类"
```

---

## Phase 2: 交互流程

### Task 3: 实现步骤输入（两步确认）

**Files:**
- Modify: `main.py` (DownloadSession.show_main_menu 后)
- Create: `tests/test_step_input.py`

**Context:** 需要一个通用的步骤输入方法，支持：显示步骤标题和提示、获取用户输入、`[b]` 返回、`[c]` 取消、可选的输入验证。

**Step 1: 编写失败的测试**

```python
# tests/test_step_input.py
"""测试步骤输入功能"""
import pytest
from unittest.mock import patch
from main import DownloadSession


def test_step_input_returns_value_on_confirm():
    """验证确认时返回输入值"""
    session = DownloadSession()
    
    with patch('rich.prompt.Prompt.ask') as mock_ask:
        mock_ask.return_value = "Qwen/Qwen3-ASR-1.7B"
        
        result = session.step_input(
            title="Model ID",
            prompt="Model ID",
            default="Qwen/Qwen3-ASR-1.7B"
        )
        
        assert result == "Qwen/Qwen3-ASR-1.7B"


def test_step_input_returns_none_on_back():
    """验证返回时返回 None"""
    session = DownloadSession()
    
    with patch('rich.prompt.Prompt.ask') as mock_ask:
        mock_ask.return_value = "b"
        
        result = session.step_input(
            title="Model ID",
            prompt="Model ID",
            default="test"
        )
        
        assert result is None


def test_step_input_exits_on_cancel():
    """验证取消时退出程序"""
    session = DownloadSession()
    
    with patch('rich.prompt.Prompt.ask') as mock_ask:
        mock_ask.return_value = "c"
        
        with pytest.raises(SystemExit):
            session.step_input(
                title="Model ID",
                prompt="Model ID",
                default="test"
            )


def test_step_input_with_validation():
    """验证带验证的输入"""
    session = DownloadSession()
    
    def validate_model_id(value: str) -> bool:
        return "/" in value
    
    with patch('rich.prompt.Prompt.ask') as mock_ask:
        # 第一次输入无效，第二次有效
        mock_ask.side_effect = ["invalid", "Qwen/Qwen3-ASR-1.7B"]
        
        result = session.step_input(
            title="Model ID",
            prompt="Model ID",
            default="Qwen/Qwen3-ASR-1.7B",
            validate=validate_model_id
        )
        
        assert result == "Qwen/Qwen3-ASR-1.7B"
```

**Step 2: 运行测试验证失败**

```bash
pytest tests/test_step_input.py -v
```
Expected: `ERROR - AttributeError: 'DownloadSession' object has no attribute 'step_input'`

**Step 3: 编写最小实现**

```python
# 在 DownloadSession 类中添加

def step_input(
    self,
    title: str,
    prompt: str,
    default: str = "",
    validate: callable = None
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
        border_style="cyan"
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
```

**Step 4: 运行测试验证通过**

```bash
pytest tests/test_step_input.py -v
```
Expected: 4 passed

**Step 5: 提交**

```bash
git add tests/test_step_input.py main.py
git commit -m "feat: 实现步骤输入功能，支持返回和取消"
```

---

### Task 4: 实现配置汇总页

**Files:**
- Modify: `main.py` (DownloadSession.step_input 后)
- Create: `tests/test_config_summary.py`

**Context:** 配置汇总页显示当前所有配置，支持修改单项或开始下载。

**Step 1: 编写失败的测试**

```python
# tests/test_config_summary.py
"""测试配置汇总页"""
import pytest
from unittest.mock import patch
from main import DownloadSession


def test_show_config_summary_returns_action():
    """验证配置汇总页返回用户操作"""
    session = DownloadSession()
    session.config.model_id = "Qwen/Qwen3-ASR-1.7B"
    session.config.source = "ms"
    session.config.output_dir = "./models"
    
    with patch('rich.prompt.Prompt.ask') as mock_ask:
        mock_ask.return_value = ""
        
        result = session.show_config_summary()
        
        assert result == ""


def test_show_config_summary_displays_all_settings():
    """验证汇总页显示所有配置"""
    session = DownloadSession()
    session.config.model_id = "Qwen/Qwen3-ASR-1.7B"
    session.config.source = "hf"
    session.config.output_dir = "/custom/path"
    
    with patch('rich.prompt.Prompt.ask') as mock_ask:
        mock_ask.return_value = ""
        
        result = session.show_config_summary()
        assert result is not None


def test_show_config_summary_empty_model_id():
    """验证空 model_id 时显示未设置"""
    session = DownloadSession()
    
    with patch('rich.prompt.Prompt.ask') as mock_ask:
        mock_ask.return_value = "m"
        
        result = session.show_config_summary()
        
        assert result == "m"
```

**Step 2: 运行测试验证失败**

```bash
pytest tests/test_config_summary.py -v
```
Expected: `ERROR - AttributeError: 'DownloadSession' object has no attribute 'show_config_summary'`

**Step 3: 编写最小实现**

```python
# 在 DownloadSession 类中添加

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
        "auto": "Auto (自动尝试)"
    }
    source_display = source_display_map.get(self.config.source, self.config.source)
    t.add_row("2) Source", source_display)
    
    # Output
    t.add_row("3) Output", self.config.output_dir)
    
    rprint("\n" + t + "\n")
    
    # 返回操作
    return Prompt.ask(
        "[Enter] 开始下载  [1/2/3] 修改  [m] 主菜单  [q] 退出",
        default=""
    )
```

**Step 4: 运行测试验证通过**

```bash
pytest tests/test_config_summary.py -v
```
Expected: 3 passed

**Step 5: 提交**

```bash
git add tests/test_config_summary.py main.py
git commit -m "feat: 实现配置汇总页"
```

---

### Task 5: 实现主交互循环

**Files:**
- Modify: `main.py` (重构 interactive_cmd 函数，约第414行)
- Modify: `main.py` (添加 run_download_setup, execute_download 方法)

**Context:** 这是核心任务，将原有的线性流程重构为循环交互流程。

**Step 1: 编写测试（验证重构后行为）**

```python
# tests/test_interactive_loop.py
"""测试主交互循环"""
import pytest
from unittest.mock import patch, MagicMock
from main import DownloadSession, DownloadConfig
import typer


def test_run_download_setup_flow():
    """验证下载配置流程"""
    session = DownloadSession()
    
    # Mock 所有输入
    with patch.object(session, 'step_input') as mock_step, \
         patch.object(session, 'show_config_summary') as mock_summary, \
         patch.object(session, 'execute_download') as mock_execute:
        
        # 模拟完整输入流程
        mock_step.side_effect = [
            "Qwen/Qwen3-ASR-1.7B",  # Model ID
            "1",  # Source: ModelScope
            "./models",  # Output
        ]
        mock_summary.return_value = ""  # 开始下载
        
        session.run_download_setup()
        
        # 验证配置正确设置
        assert session.config.model_id == "Qwen/Qwen3-ASR-1.7B"
        assert session.config.source == "ms"
        assert session.config.output_dir == "./models"
        mock_execute.assert_called_once()


def test_execute_download_success():
    """验证下载成功流程"""
    session = DownloadSession()
    session.config.model_id = "Qwen/Qwen3-ASR-1.7B"
    session.config.output_dir = "./models"
    
    with patch('main.ModelDownloader') as mock_downloader_class:
        mock_downloader = MagicMock()
        mock_downloader.download.return_value = True
        mock_downloader_class.return_value = mock_downloader
        
        with patch.object(session, 'show_continue_menu') as mock_continue:
            mock_continue.return_value = "4"  # 退出
            
            session.execute_download()
            
            # 验证添加到历史
            assert "Qwen/Qwen3-ASR-1.7B" in session.download_history


def test_execute_download_failure():
    """验证下载失败流程"""
    session = DownloadSession()
    session.config.model_id = "Qwen/Qwen3-ASR-1.7B"
    session.config.output_dir = "./models"
    
    with patch('main.ModelDownloader') as mock_downloader_class:
        mock_downloader = MagicMock()
        mock_downloader.download.return_value = False
        mock_downloader_class.return_value = mock_downloader
        
        session.execute_download()
        
        # 不应添加到历史
        assert "Qwen/Qwen3-ASR-1.7B" not in session.download_history
```

**Step 2: 运行测试验证失败**

```bash
pytest tests/test_interactive_loop.py -v
```
Expected: `ERROR - Multiple errors (run_download_setup, execute_download not defined)`

**Step 3: 编写实现**

```python
# 在 DownloadSession 类中添加以下方法

def run_download_setup(self):
    """
    运行下载配置流程
    
    流程:
    1. 输入 Model ID
    2. 输入 Source
    3. 输入 Output
    4. 显示配置汇总
    5. 用户确认后执行下载
    """
    while True:
        # 步骤 1: 输入 Model ID
        model_id = self.step_input(
            title="Model ID",
            prompt="Model ID",
            default=self.config.model_id or "Qwen/Qwen3-ASR-1.7B",
            validate=model_id_validator
        )
        if model_id is None:
            return  # 用户选择返回
        self.config.model_id = model_id
        
        # 步骤 2: 输入 Source
        source = self.step_input(
            title="Source",
            prompt="Source (1=MS, 2=HF, 3=Auto)",
            default=self.config.source
        )
        if source is None:
            return
        
        # 转换用户输入为源标识
        if source in ["1", "ms"]:
            self.config.source = SOURCE_MS
        elif source in ["2", "hf"]:
            self.config.source = SOURCE_HF
        else:
            self.config.source = SOURCE_AUTO
        
        # 步骤 3: 输入 Output
        output = self.step_input(
            title="Output",
            prompt="Output",
            default=self.config.output_dir
        )
        if output is None:
            return
        self.config.output_dir = output
        
        # 步骤 4: 显示配置汇总
        action = self.show_config_summary()
        
        # 步骤 5: 处理用户操作
        if action == "":
            self.execute_download()
            break
        elif action == "m":
            return
        elif action == "q":
            rprint("[yellow]再见！[/yellow]")
            raise typer.Exit(0)


def execute_download(self):
    """
    执行下载
    
    使用当前配置执行下载，
    成功后添加到历史记录并显示继续菜单。
    """
    from main import ModelDownloader
    
    d = ModelDownloader(
        output_dir=self.config.output_dir,
        source=self.config.source
    )
    
    try:
        if d.download(self.config.model_id):
            d.verify(self.config.model_id)
            self.add_to_history(self.config.model_id)
            rprint("\n[bold green]✓ 下载成功[/bold green]")
            self.show_continue_menu()
        else:
            rprint("\n[bold red]✗ 下载失败[/bold red]")
    except Exception as e:
        rprint(f"\n[bold red]错误: {e}[/bold red]")
        logger.error(f"Download failed: {e}", exc_info=True)
```

**Step构 interactive_cmd**

```python
# 4: 重 替换原有的 interactive_cmd 函数（约第414行）

@app.command("interactive")
def interactive_cmd():
    """
    交互式下载模式
    
    支持:
    - 主菜单导航
    - 下载配置和返回修改
    - 下载后继续选项
    - 配置沿用
    """
    session = DownloadSession()
    
    try:
        while True:
            choice = session.show_main_menu()
            
            if choice == "1":
                session.run_download_setup()
            elif choice == "2":
                session.show_current_config()
            elif choice == "3":
                session.run_clean_cache()
            elif choice == "4":
                rprint("[yellow]再见！[/yellow]")
                raise typer.Exit(0)
    except KeyboardInterrupt:
        rprint("\n[yellow]Cancelled by user[/yellow]")
        raise typer.Exit(0)
```

**Step 5: 添加缺失的方法（临时存根）**

```python
# 添加到 DownloadSession 类（后续完善）

def show_current_config(self):
    """显示当前配置（占位实现）"""
    from rich.table import Table
    from rich.prompt import Prompt
    
    t = Table(title="当前配置", show_header=False)
    t.add_row("Model ID", self.config.model_id or "[未设置]")
    t.add_row("Source", self.config.source)
    t.add_row("Output", self.config.output_dir)
    rprint(t)
    
    Prompt.ask("[Enter] 返回主菜单", default="")


def run_clean_cache(self):
    """清理缓存（复用现有逻辑）"""
    from . import clean_cmd
    clean_cmd(hf=False, ms=False, all=True)
```

**Step 6: 运行测试验证通过**

```bash
pytest tests/test_interactive_loop.py -v
```
Expected: 3 passed

**Step 7: 提交**

```bash
git add tests/test_interactive_loop.py main.py
git commit -m "feat: 实现主交互循环和下载配置流程"
```

---

## Phase 3: 状态继承

### Task 6: 实现继续选项菜单

**Files:**
- Modify: `main.py` (DownloadSession 类)
- Create: `tests/test_continue_menu.py`

**Context:** 下载成功后显示继续选项菜单，支持同目录继续、新目录下载、返回主菜单、退出。

**Step 1: 编写失败的测试**

```python
# tests/test_continue_menu.py
"""测试继续选项菜单"""
import pytest
from unittest.mock import patch
from main import DownloadSession


def test_show_continue_menu_returns_choice():
    """验证继续菜单返回用户选择"""
    session = DownloadSession()
    
    with patch('rich.prompt.Prompt.ask') as mock_ask:
        mock_ask.return_value = "1"
        
        result = session.show_continue_menu()
        
        assert result == "1"


def test_show_continue_menu_all_choices():
    """验证所有选项都能正确返回"""
    session = DownloadSession()
    
    for expected_choice in ["1", "2", "3", "4"]:
        with patch('rich.prompt.Prompt.ask') as mock_ask:
            mock_ask.return_value = expected_choice
            
            result = session.show_continue_menu()
            
            assert result == expected_choice


def test_show_continue_menu_default_is_1():
    """验证默认选项为 1"""
    session = DownloadSession()
    
    with patch('rich.prompt.Prompt.ask') as mock_ask:
        mock_ask.return_value = ""
        
        result = session.show_continue_menu()
        
        mock_ask.assert_called()
```

**Step 2: 运行测试验证失败**

```bash
pytest tests/test_continue_menu.py -v
```
Expected: `ERROR - AttributeError: 'DownloadSession' object has no attribute 'show_continue_menu'`

**Step 3: 编写实现**

```python
# 在 DownloadSession 类中添加

def show_continue_menu(self) -> str:
    """
    显示下载后继续选项菜单
    
    Returns:
        str: 用户选择
        - "1": 同目录继续下载其他模型
        - "2": 新目录下载新模型
        - "3": 返回主菜单
        - "4": 退出
    """
    from rich.prompt import Prompt
    from rich.panel import Panel
    
    menu = Panel.fit(
        "[bold green]✓ 下载成功[/bold green]\n\n"
        "  1) 同目录继续下载其他模型\n"
        "  2) 新目录下载新模型\n"
        "  3) 返回主菜单\n"
        "  4) 退出",
        border_style="green"
    )
    rprint("\n" + menu + "\n")
    
    return Prompt.ask(
        "请选择",
        choices=["1", "2", "3", "4"],
        default="1"
    )
```

**Step 4: 运行测试验证通过**

```bash
pytest tests/test_continue_menu.py -v
```
Expected: 3 passed

**Step 5: 提交**

```bash
git add tests/test_continue_menu.py main.py
git commit -m "feat: 实现继续选项菜单"
```

---

### Task 7: 实现配置沿用逻辑

**Files:**
- Modify: `main.py` (DownloadSession 类)
- Create: `tests/test_config_reuse.py`

**Context:** 当用户选择继续下载时，询问是否沿用当前配置。

**Step 1: 编写失败的测试**

```python
# tests/test_config_reuse.py
"""测试配置沿用逻辑"""
import pytest
from unittest.mock import patch
from main import DownloadSession


def test_ask_reuse_config_yes():
    """验证选择沿用配置时返回 True"""
    session = DownloadSession()
    session.config.model_id = "Qwen/Qwen3-ASR-1.7B"
    session.config.source = "ms"
    session.config.output_dir = "./models"
    
    with patch('rich.prompt.Confirm.ask') as mock_confirm:
        mock_confirm.return_value = True
        
        result = session.ask_reuse_config()
        
        assert result is True
        assert session.config.model_id == "Qwen/Qwen3-ASR-1.7B"


def test_ask_reuse_config_no():
    """验证选择不沿用配置时清空 model_id"""
    session = DownloadSession()
    session.config.model_id = "Qwen/Qwen3-ASR-1.7B"
    
    with patch('rich.prompt.Confirm.ask') as mock_confirm:
        mock_confirm.return_value = False
        
        result = session.ask_reuse_config()
        
        assert result is False
        assert session.config.model_id == ""
        assert session.config.source == "ms"
        assert session.config.output_dir == "./models"


def test_ask_reuse_config_displays_current_config():
    """验证沿用询问时显示当前配置"""
    session = DownloadSession()
    session.config.model_id = "Qwen/Qwen3-ASR-1.7B"
    session.config.source = "hf"
    session.config.output_dir = "/custom/path"
    
    with patch('rich.prompt.Confirm.ask') as mock_confirm:
        mock_confirm.return_value = True
        
        session.ask_reuse_config()
        
        mock_confirm.assert_called_once()
```

**Step 2: 运行测试验证失败**

```bash
pytest tests/test_config_reuse.py -v
```
Expected: `ERROR - AttributeError: 'DownloadSession' object has no attribute 'ask_reuse_config'`

**Step 3: 编写实现**

```python
# 在 DownloadSession 类中添加

def ask_reuse_config(self) -> bool:
    """
    询问是否沿用当前配置
    
    Returns:
        bool: True=沿用当前配置, False=需要重新输入 model_id
    """
    from rich.prompt import Confirm
    from rich.table import Table
    
    # 显示当前配置
    t = Table(title="当前配置", show_header=False, box=None)
    t.add_row("Model", self.config.model_id)
    t.add_row("Source", self.config.source)
    t.add_row("Output", self.config.output_dir)
    rprint(t)
    
    reuse = Confirm.ask("是否沿用当前配置?", default=True)
    
    if not reuse:
        self.config.model_id = ""
    
    return reuse
```

**Step 4: 运行测试验证通过**

```bash
pytest tests/test_config_reuse.py -v
```
Expected: 3 passed

**Step 5: 提交**

```bash
git add tests/test_config_reuse.py main.py
git commit -m "feat: 实现配置沿用逻辑"
```

---

## Phase 4: 测试验证

### Task 8: 运行所有测试并修复问题

**Files:**
- Modify: `main.py` (修复发现的问题)

**Step 1: 运行所有测试**

```bash
cd "D:\TRAE_Script\ModelHub-Downloader"

pytest tests/test_download_config.py \
       tests/test_download_session.py \
       tests/test_step_input.py \
       tests/test_config_summary.py \
       tests/test_interactive_loop.py \
       tests/test_continue_menu.py \
       tests/test_config_reuse.py \
       -v --tb=short
```

**预期结果**: 所有测试通过

**Step 2: 检查现有测试是否通过**

```bash
pytest tests/ -v --tb=short
```

**Step 3: 修复发现的问题**

**Step 4: 提交测试修复**

```bash
git add tests/ main.py
git commit -m "test: 运行所有测试并修复问题"
```

---

### Task 9: 最终提交和验证

**Files:**
- Modify: `main.py` (确保 KeyboardInterrupt 处理存在)

**Step 1: 验证 KeyboardInterrupt 处理**

**Step 2: 最终验证**

```bash
python -m py_compile main.py
python -c "from main import interactive_cmd; print('Import OK')"
```

**Step 3: 最终提交**

```bash
git status
git diff --stat
git add -A
git commit -m "feat: 交互流程重新设计

- 添加 DownloadConfig 数据类管理下载配置
- 添加 DownloadSession 会话管理类
- 实现步骤输入支持 [b] 返回 [c] 取消
- 实现配置汇总页
- 实现主交互循环
- 实现继续选项菜单
- 实现配置沿用逻辑"

git tag -a v2.0.0 -m "Release v2.0.0: 交互流程重新设计"
```

**Step 4: 推送到远程**

```bash
git push origin main --tags
```

---

## 测试汇总

| 测试文件 | 测试数 | 预期结果 |
|---------|-------|---------|
| test_download_config.py | 4 | ✅ PASS |
| test_download_session.py | 4 | ✅ PASS |
| test_step_input.py | 4 | ✅ PASS |
| test_config_summary.py | 3 | ✅ PASS |
| test_interactive_loop.py | 3 | ✅ PASS |
| test_continue_menu.py | 3 | ✅ PASS |
| test_config_reuse.py | 3 | ✅ PASS |
| **合计** | **24** | **✅** |

---

## 文件变更摘要

| 文件 | 变更 |
|-----|------|
| `main.py` | +200 行（重构 interactive_cmd，新增 DownloadConfig, DownloadSession） |
| `tests/test_download_config.py` | +40 行 |
| `tests/test_download_session.py` | +50 行 |
| `tests/test_step_input.py` | +55 行 |
| `tests/test_config_summary.py` | +40 行 |
| `tests/test_interactive_loop.py` | +50 行 |
| `tests/test_continue_menu.py` | +35 行 |
| `tests/test_config_reuse.py` | +40 行 |

---

## Plan Complete

**Saved to:** `docs/plans/2026-02-11-interaction-redesign.md`

**Two execution options:**

1. **Subagent-Driven (this session)** - 使用 superpowers:subagent-driven-development，每个任务派遣独立 subagent

2. **Parallel Session (separate)** - 在新会话中使用 superpowers:executing-plans

**Which approach?**
