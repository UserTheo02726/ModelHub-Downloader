"""测试步骤输入功能"""

import pytest
from unittest.mock import patch
from main import DownloadSession
import typer


def test_step_input_returns_value_on_confirm():
    """验证确认时返回输入值"""
    session = DownloadSession()

    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return_value = "Qwen/Qwen3-ASR-1.7B"

        result = session.step_input(
            title="Model ID", prompt="Model ID", default="Qwen/Qwen3-ASR-1.7B"
        )

        assert result == "Qwen/Qwen3-ASR-1.7B"


def test_step_input_returns_none_on_back():
    """验证返回时返回 None"""
    session = DownloadSession()

    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return_value = "b"

        result = session.step_input(title="Model ID", prompt="Model ID", default="test")

        assert result is None


def test_step_input_exits_on_cancel():
    """验证取消时退出程序"""
    session = DownloadSession()

    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return_value = "c"

        with pytest.raises(typer.Exit):
            session.step_input(title="Model ID", prompt="Model ID", default="test")


def test_step_input_with_validation():
    """验证带验证的输入"""
    session = DownloadSession()

    def validate_model_id(value: str) -> bool:
        return "/" in value

    with patch("rich.prompt.Prompt.ask") as mock_ask:
        # 第一次输入无效，第二次有效
        mock_ask.side_effect = ["invalid", "Qwen/Qwen3-ASR-1.7B"]

        result = session.step_input(
            title="Model ID",
            prompt="Model ID",
            default="Qwen/Qwen3-ASR-1.7B",
            validate=validate_model_id,
        )

        assert result == "Qwen/Qwen3-ASR-1.7B"
