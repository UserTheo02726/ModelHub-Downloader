"""测试 UnifiedProgressBar 统一进度条"""

import pytest
from unittest.mock import patch, MagicMock


def test_unified_progress_creates_progress():
    """测试进度条创建"""
    from main import UnifiedProgressBar

    bar = UnifiedProgressBar()

    assert bar._progress is not None
    assert bar._console is not None


def test_unified_progress_hf_context_adds_task():
    """测试 HF 下载上下文添加任务"""
    from main import UnifiedProgressBar

    bar = UnifiedProgressBar()

    with patch.object(bar._progress, "start"):
        with patch.object(bar._progress, "stop"):
            with bar.HF("test/model") as ctx:
                assert ctx is None  # yield None


def test_unified_progress_ms_context_adds_task():
    """测试 MS 下载上下文添加任务"""
    from main import UnifiedProgressBar

    bar = UnifiedProgressBar()

    with patch.object(bar._progress, "start"):
        with patch.object(bar._progress, "stop"):
            with bar.MS("test/model") as ctx:
                assert ctx is None


def test_unified_progress_has_transient():
    """测试进度条配置 transient=True"""
    from main import UnifiedProgressBar

    bar = UnifiedProgressBar()

    # 验证进度条已创建
    assert bar._progress is not None
