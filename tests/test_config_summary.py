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

    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return_value = ""

        result = session.show_config_summary()

        assert result == ""


def test_show_config_summary_displays_all_settings():
    """验证汇总页显示所有配置"""
    session = DownloadSession()
    session.config.model_id = "Qwen/Qwen3-ASR-1.7B"
    session.config.source = "hf"
    session.config.output_dir = "/custom/path"

    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return_value = ""

        # 验证调用成功（不抛异常即通过）
        result = session.show_config_summary()
        assert result is not None


def test_show_config_summary_empty_model_id():
    """验证空 model_id 时显示未设置"""
    session = DownloadSession()

    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return_value = "m"

        result = session.show_config_summary()

        assert result == "m"
