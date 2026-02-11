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

    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return_value = "1"

        choice = session.show_main_menu()

        assert choice == "1"
