"""测试认证错误和指南 UI 函数"""

import pytest
from unittest.mock import patch, MagicMock


def test_show_auth_error_returns_choice_1():
    """测试 show_auth_error 返回选择 '1' (查看指南)"""
    from main import show_auth_error

    with patch("main.Prompt.ask") as mock_prompt:
        mock_prompt.return_value = "1"

        result = show_auth_error("facebook/sam3", "hf", "Gated model error")

        assert result == "1"


def test_show_auth_error_returns_choice_2():
    """测试 show_auth_error 返回选择 '2' (返回主菜单)"""
    from main import show_auth_error

    with patch("main.Prompt.ask") as mock_prompt:
        mock_prompt.return_value = "2"

        result = show_auth_error("test/model", "ms", "Some error")

        assert result == "2"


def test_show_auth_error_returns_choice_3():
    """测试 show_auth_error 返回选择 '3' (重试下载)"""
    from main import show_auth_error

    with patch("main.Prompt.ask") as mock_prompt:
        mock_prompt.return_value = "3"

        result = show_auth_error("test/model", "hf", "401 Unauthorized")

        assert result == "3"


def test_show_auth_guide_renders_without_error():
    """测试 show_auth_guide 正常渲染"""
    from main import show_auth_guide

    # 验证函数可调用且不抛出异常
    try:
        show_auth_guide()
        assert True
    except Exception as e:
        pytest.fail(f"show_auth_guide raised: {e}")
