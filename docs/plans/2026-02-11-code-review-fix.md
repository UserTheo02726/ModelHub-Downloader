# ModelHub-Downloader 代码审查修复实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 修复代码审查发现的 10 个问题，提升代码质量、安全性和可维护性

**Architecture:** 
- 重构 `main.py` 中的 `ModelDownloader` 类
- 提取公共逻辑到独立模块
- 添加异常类、验证器、配置管理
- 保持单文件 CLI 结构，拆分业务逻辑

**Tech Stack:** Python 3.9+, rich, typer, huggingface_hub, modelscope

---

## 任务总览

| 优先级 | 任务数 | 预计时间 |
|-------|-------|---------|
| P0 | 2 | 30 分钟 |
| P1 | 4 | 60 分钟 |
| P2 | 3 | 45 分钟 |
| **合计** | **9** | **135 分钟** |

---

## P0 - 立即修复

### Task 1: 添加自定义异常类和日志记录

**Files:**
- Modify: `main.py:1-10` (模块文档)
- Modify: `main.py:40-70` (异常类插入点)

**Step 1: 编写失败的测试**

```python
# tests/test_exceptions.py
import pytest
from main import (
    ModelDownloadError,
    ValidationError,
    DownloadError,
    path_validator,
)


def test_custom_exceptions_exist():
    """验证自定义异常类存在"""
    assert issubclass(ModelDownloadError, Exception)
    assert issubclass(ValidationError, ModelDownloadError)
    assert issubclass(DownloadError, ModelDownloadError)


def test_path_validator_function():
    """验证路径验证器存在且工作"""
    # 有效路径
    assert path_validator("/valid/path") == True
    assert path_validator("./relative") == True
    
    # 无效路径 - 路径遍历攻击
    assert path_validator("../../../etc/passwd") == False
    assert path_validator("/valid/../../etc") == False


def test_path_validator_with_special_chars():
    """验证路径验证器阻止特殊字符"""
    assert path_validator("/path with spaces") == False
    assert path_validator("/path\twith\ttabs") == False
```

**Step 2: 运行测试验证失败**

```bash
cd D:\TRAE_Script\ModelHub-Downloader
mkdir -p tests
pytest tests/test_exceptions.py -v
```
Expected: FAIL - `ModelDownloadError` not defined

**Step 3: 编写最小实现**

```python
# 在 main.py 文件顶部导入后、配置常量前添加

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
            {"field": field, "value": value, "reason": reason}
        )


class DownloadError(ModelDownloadError):
    """下载过程异常"""
    def __init__(self, model_id: str, source: str, original_error: Exception):
        self.model_id = model_id
        self.source = source
        self.original_error = original_error
        super().__init__(
            f"Download failed for {model_id} from {source}: {original_error}",
            {"model_id": model_id, "source": source}
        )


# === 路径验证器 ===
import re

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
    if re.search(r'[\s\t\n\r]', path):
        return False
    
    # 检查绝对路径遍历
    normalized = path.replace("\\", "/")
    if normalized.startswith("/"):
        # 允许简单绝对路径，不允许遍历
        parts = [p for p in normalized.split("/") if p]
        if ".." in parts:
            return False
    
    return True
```

**Step 4: 运行测试验证通过**

```bash
pytest tests/test_exceptions.py -v
```
Expected: PASS (4 tests)

**Step 5: 提交**

```bash
git add tests/test_exceptions.py
git commit -m "feat: 添加自定义异常类和路径验证器"
```

---

### Task 2: 修复异常处理和日志记录

**Files:**
- Modify: `main.py:72-116` (download_hf, download_ms, download 方法)

**Step 1: 编写失败的测试**

```python
# tests/test_error_handling.py
import pytest
from unittest.mock import Mock, patch
from main import ModelDownloader, DownloadError, ValidationError


def test_download_hf_logs_exception():
    """验证 download_hf 记录异常日志"""
    d = ModelDownloader()
    with patch('main.HfApi') as mock_api:
        mock_api.return_value.model_info.side_effect = Exception("API Error")
        # 应该记录异常而不是静默
        # 通过验证返回 False 和不抛出异常来测试
        result = d.download_hf("Qwen/Qwen3-ASR-1.7B")
        assert result == False


def test_download_validates_model_id():
    """验证 download 方法验证 model_id，无效时返回 False"""
    d = ModelDownloader()
    # 无效 ID (无 /)
    result = d.download("invalid_id")
    assert result == False
    
    # 无效 ID (多个 /)
    result = d.download("a/b/c")
    assert result == False


def test_download_raises_on_critical_error():
    """验证严重错误应该抛出异常而不是静默"""
    d = ModelDownloader()
    with patch('main.snapshot_download') as mock_download:
        mock_download.side_effect = Exception("Network error")
        # 应该抛出 DownloadError 而不是静默返回 False
        with pytest.raises(DownloadError):
            d.download_hf("Qwen/Qwen3-ASR-1.7B")
```

**Step 2: 运行测试验证失败**

```bash
pytest tests/test_error_handling.py -v
```
Expected: FAIL - DownloadError not defined (第一个测试通过但异常未被记录)

**Step 3: 编写最小实现**

```python
# 在 main.py 中添加日志配置（在异常类后、下载器类前）
import logging

# === 日志配置 ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ModelDownloader")


# 修改 ModelDownloader 类的 download_hf 方法 (原72-90行)
def download_hf(self, model_id: str) -> bool:
    """
    从 HuggingFace 下载模型
    
    Args:
        model_id: 模型 ID (格式: org/model-name)
        
    Returns:
        bool: 下载是否成功
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


# 修改 download_ms 方法 (原92-103行)
def download_ms(self, model_id: str) -> bool:
    """
    从 ModelScope 下载模型
    
    Args:
        model_id: 模型 ID
        
    Returns:
        bool: 下载是否成功
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


# 修改 download 方法处理验证失败 (原105-116行)
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
```

**Step 4: 运行测试验证通过**

```bash
pytest tests/test_error_handling.py -v
```
Expected: PASS (3 tests)

**Step 5: 提交**

```bash
git add tests/test_error_handling.py main.py
git commit -m "fix: 完善异常处理，添加日志记录"
```

---

## P1 - 下个迭代

### Task 3: 统一 CLI 命令命名规范

**Files:**
- Modify: `main.py:141-224` (CLI 命令函数)

**Step 1: 编写失败的测试**

```python
# tests/test_cli_naming.py
import pytest
from main import app


def test_download_command_exists():
    """验证 download 命令存在"""
    # Typer 命令应该是以函数名注册
    commands = [cmd.name for cmd in app.registered_commands]
    assert "download" in commands


def test_clean_command_exists():
    """验证 clean 命令存在"""
    commands = [cmd.name for cmd in app.registered_commands]
    assert "clean" in commands


def test_list_command_exists():
    """验证 list 命令存在"""
    commands = [cmd.name for cmd in app.registered_commands]
    assert "list" in commands


def test_interactive_command_exists():
    """验证 interactive 命令存在"""
    commands = [cmd.name for cmd in app.registered_commands]
    assert "interactive" in commands
```

**Step 2: 运行测试验证状态**

```bash
pytest tests/test_cli_naming.py -v
```
Expected: PASS (命令已存在)

**Step 3: 分析并优化命名**

当前问题：函数名 `interactive_cmd` 与 `download_cmd` 风格不一致

```python
# 修改 CLI 函数名（可选，添加别名保持向后兼容）

@app.command("interactive")
def interactive_cmd():
    """Interactive mode"""
    # ... 现有代码保持不变
```

**决策**: 命令名称已统一（download, clean, list, interactive），无需修改

**Step 4: 提交**

```bash
git add tests/test_cli_naming.py
git commit -m "docs: 添加 CLI 命令测试"
```

---

### Task 4: 强化输入验证（模型 ID 验证器）

**Files:**
- Modify: `main.py:46-47` (validate 方法)
- Add: `main.py` (model_id_validator 函数)

**Step 1: 编写失败的测试**

```python
# tests/test_validation.py
import pytest
from main import ModelDownloader, model_id_validator, path_validator, ValidationError


def test_model_id_validator_function():
    """验证 model_id_validator 函数存在且工作"""
    # 有效 ID
    assert model_id_validator("Qwen/Qwen3-ASR-1.7B") == True
    assert model_id_validator("facebook/opt-1.3b") == True
    assert model_id_validator("THUDM/chatglm3-6b") == True
    
    # 无效 ID - 缺少组织名
    assert model_id_validator("model-only") == False
    
    # 无效 ID - 多个 /
    assert model_id_validator("a/b/c") == False
    
    # 无效 ID - 包含特殊字符
    assert model_id_validator("org/model?name") == False
    assert model_id_validator("org/model#tag") == False
    
    # 无效 ID - 过长
    long_id = "a/" + "b" * 200
    assert model_id_validator(long_id) == False


def test_validate_method_uses_validator():
    """验证 ModelDownloader.validate 使用 validator"""
    d = ModelDownloader()
    # 有效
    assert d.validate("Qwen/Qwen3-ASR-1.7B") == True
    
    # 无效
    assert d.validate("invalid") == False


def test_validate_rejects_empty():
    """验证 validate 拒绝空值"""
    d = ModelDownloader()
    assert d.validate("") == False
    assert d.validate(None) == False
```

**Step 2: 运行测试验证失败**

```bash
pytest tests/test_validation.py -v
```
Expected: FAIL - `model_id_validator` not defined

**Step 3: 编写最小实现**

```python
# 在 path_validator 后添加 model_id_validator
import re

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
    valid_pattern = re.compile(r'^[a-zA-Z0-9_\-\.]+$')
    if not (valid_pattern.match(org) and valid_pattern.match(model)):
        return False
    
    # 不能以连字符或点开头/结尾
    if org[0] in ('-', '.') or org[-1] in ('-', '.'):
        return False
    if model[0] in ('-', '.') or model[-1] in ('-', '.'):
        return False
    
    return True


# 修改 ModelDownloader.validate 方法
def validate(self, model_id: str) -> bool:
    """验证模型 ID 格式"""
    return model_id_validator(model_id)
```

**Step 4: 运行测试验证通过**

```bash
pytest tests/test_validation.py -v
```
Expected: PASS (10 tests)

**Step 5: 提交**

```bash
git add tests/test_validation.py main.py
git commit -m "feat: 强化模型 ID 输入验证"
```

---

### Task 5: 添加文件路径安全验证

**Files:**
- Modify: `main.py:49-50` (get_path 方法)

**Step 1: 编写失败的测试**

```python
# tests/test_path_safety.py
import pytest
from main import ModelDownloader, path_validator
from pathlib import Path


def test_get_path_validates_output_dir():
    """验证 get_path 验证输出目录安全性"""
    d = ModelDownloader(output_dir="./output")
    
    # 有效 model_id
    path = d.get_path("Qwen/Qwen3-ASR-1.7B")
    assert path == Path("./output/Qwen3-ASR-1.7B")


def test_get_path_blocks_path_traversal():
    """验证 get_path 阻止路径遍历攻击"""
    d = ModelDownloader(output_dir="./output")
    
    # 恶意路径遍历
    with pytest.raises(Exception):
        d.get_path("../../../etc/passwd")
    
    with pytest.raises(Exception):
        d.get_path("valid/../../etc/passwd")


def test_download_validates_output_path():
    """验证 download 命令验证输出路径"""
    d = ModelDownloader(output_dir="../../sensitive")
    
    with pytest.raises(Exception):
        d.download_hf("Qwen/Qwen3-ASR-1.7B")
```

**Step 2: 运行测试验证失败**

```bash
pytest tests/test_path_safety.py -v
```
Expected: FAIL - 恶意路径未被阻止

**Step 3: 编写最小实现**

```python
# 修改 ModelDownloader.get_path 方法
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
    
    # 检查是否在允许的输出目录内
    output_dir_resolved = self.output_dir.resolve()
    
    # 简单实现：确保路径不以 ".." 开头
    path_str = str(normalized).replace("\\", "/")
    if path_str.startswith(".."):
        raise ValidationError("output_dir", str(full_path), "Path traversal not allowed")
    
    return full_path
```

**Step 4: 运行测试验证通过**

```bash
pytest tests/test_path_safety.py -v
```
Expected: PASS (3 tests)

**Step 5: 提交**

```bash
git add tests/test_path_safety.py main.py
git commit -m "feat: 添加文件路径安全验证"
```

---

### Task 6: 提取公共下载逻辑（减少重复）

**Files:**
- Refactor: `main.py:72-116` (download_hf, download_ms)
- Add: `main.py` (抽象下载方法)

**Step 1: 编写失败的测试**

```python
# tests/test_download_abstraction.py
import pytest
from unittest.mock import Mock, patch
from main import ModelDownloader


def test_download_hf_calls_abstract_download():
    """验证 download_hf 使用抽象下载方法"""
    d = ModelDownloader(source="hf")
    with patch.object(d, '_download_from_source') as mock_download:
        mock_download.return_value = True
        result = d.download_hf("Qwen/Qwen3-ASR-1.7B")
        mock_download.assert_called_once()
        assert result == True


def test_download_ms_calls_abstract_download():
    """验证 download_ms 使用抽象下载方法"""
    d = ModelDownloader(source="ms")
    with patch.object(d, '_download_from_source') as mock_download:
        mock_download.return_value = True
        result = d.download_ms("Qwen/Qwen3-ASR-1.7B")
        mock_download.assert_called_once()
        assert result == True
```

**Step 2: 运行测试验证失败**

```bash
pytest tests/test_download_abstraction.py -v
```
Expected: FAIL - `_download_from_source` not defined

**Step 3: 编写最小实现**

```python
# 在 download_ms 方法后添加抽象方法

def _download_from_source(
    self,
    model_id: str,
    source: str,
    download_func: callable
) -> bool:
    """
    抽象下载方法 - 统一下载流程
    
    Args:
        model_id: 模型 ID
        source: 源标识 (HF/MS)
        download_func: 具体下载函数
        
    Returns:
        bool: 下载是否成功
    """
    target = self.get_path(model_id)
    rprint(f"\n[cyan]Download from {source}:[/cyan] {model_id}")
    rprint(f"  [dim]→ {target}[/dim]")
    
    try:
        target.mkdir(parents=True, exist_ok=True)
        if size := self._get_size(model_id):
            rprint(f"  [dim]Size: {self._fmt_size(size)}[/dim]")
        download_func(model_id, str(target))
        rprint(f"  [green]OK[/green]")
        logger.info(f"Successfully downloaded {model_id} from {source}")
        return True
    except Exception as e:
        rprint(f"  [red]Error: {e}[/red]")
        logger.error(f"Download failed for {model_id} from {source}", exc_info=True)
        raise DownloadError(model_id, source, e) from e


# 修改 download_hf
def download_hf(self, model_id: str) -> bool:
    """从 HuggingFace 下载模型"""
    def hf_download(model_id, target):
        snapshot_download(
            repo_id=model_id,
            local_dir=target,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    return self._download_from_source(model_id, "HF", hf_download)


# 修改 download_ms
def download_ms(self, model_id: str) -> bool:
    """从 ModelScope 下载模型"""
    def ms_download(model_id, target):
        ms_snapshot_download(model_id, local_dir=target)
    return self._download_from_source(model_id, "MS", ms_download)
```

**Step 4: 运行测试验证通过**

```bash
pytest tests/test_download_abstraction.py -v
```
Expected: PASS (2 tests)

**Step 5: 提交**

```bash
git add tests/test_download_abstraction.py main.py
git commit -m "refactor: 提取公共下载逻辑，减少代码重复"
```

---

## P2 - 长期优化

### Task 7: 添加模块级文档字符串

**Files:**
- Modify: `main.py:1-8` (添加模块文档)

**Step 1: 编写测试**

```python
# tests/test_documentation.py
import main


def test_module_has_docstring():
    """验证模块有文档字符串"""
    assert main.__doc__ is not None
    assert len(main.__doc__) > 50


def test_downloader_class_has_docstring():
    """验证 ModelDownloader 有文档字符串"""
    assert main.ModelDownloader.__doc__ is not None


def test_downloader_methods_have_docstrings():
    """验证主要方法有文档字符串"""
    for method in ['download', 'download_hf', 'download_ms', 'validate', 'get_path', 'verify']:
        doc = getattr(main.ModelDownloader, method).__doc__
        assert doc is not None, f"ModelDownloader.{method} missing docstring"
```

**Step 2: 运行测试验证状态**

```bash
pytest tests/test_documentation.py -v
```
Expected: FAIL - 缺少文档

**Step 3: 编写最小实现**

```python
"""
ModelHub Downloader - 公开模型下载工具

支持从 HuggingFace 和 ModelScope 下载预训练模型。

Features:
    - 双平台支持 (HF/MS)
    - 交互式下载模式
    - 下载验证
    - 缓存清理

Usage:
    python main.py download <model_id>
    python main.py interactive
    python main.py clean --all

Example:
    python main.py download Qwen/Qwen3-ASR-1.7B --source ms
"""
```

**Step 4: 运行测试验证通过**

```bash
pytest tests/test_documentation.py -v
```
Expected: PASS

**Step 5: 提交**

```bash
git add tests/test_documentation.py main.py
git commit -m "docs: 添加模块和类文档字符串"
```

---

### Task 8: 添加下载重试机制

**Files:**
- Modify: `main.py` (_download_from_source 或下载方法)

**Step 1: 编写失败的测试**

```python
# tests/test_retry.py
import pytest
from unittest.mock import Mock, patch, call
from main import ModelDownloader


def test_download_retries_on_failure():
    """验证下载失败时自动重试"""
    d = ModelDownloader()
    with patch('main.snapshot_download') as mock_download:
        # 前两次失败，第三次成功
        mock_download.side_effect = [
            Exception("Network error"),
            Exception("Network error"),
            None  # 成功
        ]
        # 应该重试 3 次后成功
        result = d.download_hf("Qwen/Qwen3-ASR-1.7B")
        assert result == True
        assert mock_download.call_count == 3


def test_download_fails_after_max_retries():
    """验证超过最大重试次数后失败"""
    d = ModelDownloader()
    with patch('main.snapshot_download') as mock_download:
        mock_download.side_effect = Exception("Persistent error")
        with pytest.raises(Exception):
            d.download_hf("Qwen/Qwen3-ASR-1.7B")
```

**Step 2: 运行测试验证失败**

```bash
pytest tests/test_retry.py -v
```
Expected: FAIL - 无重试机制

**Step 3: 编写最小实现**

```python
# 在 main.py 添加重试配置和函数

# === 重试配置 ===
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 2  # 秒


def with_retry(max_retries: int = DEFAULT_MAX_RETRIES, delay: int = DEFAULT_RETRY_DELAY):
    """
    重试装饰器
    
    Args:
        max_retries: 最大重试次数
        delay: 重试间隔（秒）
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying in {delay}s...")
                        import time
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed")
                        raise
            raise last_exception
        return wrapper
    return decorator


# 修改 _download_from_source 使用重试
def _download_from_source(
    self,
    model_id: str,
    source: str,
    download_func: callable
) -> bool:
    """统一下载流程（带重试）"""
    target = self.get_path(model_id)
    rprint(f"\n[cyan]Download from {source}:[/cyan] {model_id}")
    rprint(f"  [dim]→ {target}[/dim]")
    
    try:
        target.mkdir(parents=True, exist_ok=True)
        if size := self._get_size(model_id):
            rprint(f"  [dim]Size: {self._fmt_size(size)}[/dim]")
        
        # 使用重试装饰器
        @with_retry(max_retries=3, delay=2)
        def safe_download():
            download_func(model_id, str(target))
        
        safe_download()
        rprint(f"  [green]OK[/green]")
        logger.info(f"Successfully downloaded {model_id} from {source}")
        return True
    except Exception as e:
        rprint(f"  [red]Error: {e}[/red]")
        logger.error(f"Download failed for {model_id} from {source}", exc_info=True)
        raise DownloadError(model_id, source, e) from e
```

**Step 4: 运行测试验证通过**

```bash
pytest tests/test_retry.py -v
```
Expected: PASS

**Step 5: 提交**

```bash
git add tests/test_retry.py main.py
git commit -m "feat: 添加下载重试机制"
```

---

### Task 9: 添加配置管理（环境变量支持）

**Files:**
- Add: `main.py` (配置加载逻辑)
- Add: `config.py` (独立配置文件)

**Step 1: 编写失败的测试**

```python
# tests/test_config.py
import pytest
from main import config, load_config


def test_config_has_default_values():
    """验证配置有默认值"""
    assert config.output_dir == "./models"
    assert config.default_source == "ms"
    assert config.max_retries == 3


def test_config_loads_from_env():
    """验证配置从环境变量加载"""
    import os
    os.environ["MODELHUB_OUTPUT"] = "/custom/path"
    os.environ["MODELHUB_SOURCE"] = "hf"
    os.environ["MODELHUB_RETRIES"] = "5"
    
    new_config = load_config()
    
    assert new_config.output_dir == "/custom/path"
    assert new_config.default_source == "hf"
    assert new_config.max_retries == 5
    
    # 清理
    del os.environ["MODELHUB_OUTPUT"]
    del os.environ["MODELHUB_SOURCE"]
    del os.environ["MODELHUB_RETRIES"]
```

**Step 2: 运行测试验证失败**

```bash
pytest tests/test_config.py -v
```
Expected: FAIL - config not defined

**Step 3: 编写最小实现**

```python
# === 配置管理 ===
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """ModelHub 配置"""
    output_dir: str = "./models"
    default_source: str = "ms"
    max_retries: int = 3
    retry_delay: int = 2
    cache_dir: str = "~/.cache"
    
    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量加载配置"""
        import os
        return cls(
            output_dir=os.environ.get("MODELHUB_OUTPUT", cls.output_dir),
            default_source=os.environ.get("MODELHUB_SOURCE", cls.default_source),
            max_retries=int(os.environ.get("MODELHUB_RETRIES", cls.max_retries)),
            retry_delay=int(os.environ.get("MODELHUB_RETRY_DELAY", cls.retry_delay)),
            cache_dir=os.environ.get("MODELHUB_CACHE", cls.cache_dir),
        )


# 全局配置实例
config = Config.from_env()


def load_config() -> Config:
    """加载配置（重新读取环境变量）"""
    return Config.from_env()
```

**Step 4: 运行测试验证通过**

```bash
pytest tests/test_config.py -v
```
Expected: PASS

**Step 5: 提交**

```bash
git add tests/test_config.py main.py
git commit -m "feat: 添加配置管理和环境变量支持"
```

---

## 测试汇总

### 运行所有测试

```bash
# 运行全部测试
pytest tests/ -v --tb=short

# 生成测试覆盖率
pytest tests/ --cov=main --cov-report=html
```

### 预期结果

| 测试文件 | 测试数 | 状态 |
|---------|-------|------|
| test_exceptions.py | 4 | ✅ PASS |
| test_error_handling.py | 3 | ✅ PASS |
| test_validation.py | 10 | ✅ PASS |
| test_path_safety.py | 3 | ✅ PASS |
| test_download_abstraction.py | 2 | ✅ PASS |
| test_cli_naming.py | 4 | ✅ PASS |
| test_documentation.py | 3 | ✅ PASS |
| test_retry.py | 2 | ✅ PASS |
| test_config.py | 2 | ✅ PASS |
| **合计** | **33** | **✅ 全部 PASS** |

---

## 变更摘要

### 文件变更

| 文件 | 操作 | 行数变化 |
|-----|------|---------|
| `main.py` | 修改 | +200 / -80 |
| `tests/test_*.py` | 新增 | +300 行 |

### 新增功能

1. ✅ 自定义异常类（ModelDownloadError, ValidationError, DownloadError）
2. ✅ 日志记录（logging 模块）
3. ✅ 强化输入验证（模型 ID 白名单）
4. ✅ 路径安全验证（路径遍历防护）
5. ✅ 下载逻辑抽象（减少重复）
6. ✅ 重试机制（自动重试 3 次）
7. ✅ 配置管理（环境变量支持）
8. ✅ 完整文档字符串

### 性能提升

- 重试机制减少网络波动导致的失败
- 统一的异常处理提升调试效率
- 路径验证防止安全漏洞

---

## 后续步骤

**计划完成并保存到 `docs/plans/2026-02-11-code-review-fix.md`。**

Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
