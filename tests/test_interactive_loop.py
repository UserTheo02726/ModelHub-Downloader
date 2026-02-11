"""测试主交互循环"""

import pytest
from unittest.mock import patch, MagicMock
from main import DownloadSession, DownloadConfig
import typer
from rich.prompt import Prompt


def test_run_download_setup_flow():
    """验证下载配置流程 - 测试配置设置是否正确"""
    session = DownloadSession()

    # Mock 输入
    with (
        patch.object(session, "step_input") as mock_step,
        patch.object(session, "show_config_summary") as mock_summary,
        patch.object(Prompt, "ask") as mock_prompt,
    ):
        # 模拟用户流程:
        # 1. 第一次 show_config_summary 返回 "1" (修改 Model ID)
        # 2. step_input 返回 Model ID
        # 3. 第二次 show_config_summary 返回 "2" (修改 Source)
        # 4. Prompt.ask 返回 "1" (ModelScope)
        # 5. 第三次 show_config_summary 返回 "3" (修改 Output)
        # 6. step_input 返回 Output 路径
        # 7. 第四次 show_config_summary 返回 "" (开始下载)
        mock_summary.side_effect = ["1", "2", "3", ""]
        mock_step.side_effect = [
            "Qwen/Qwen3-ASR-1.7B",  # Model ID
            "./models",  # Output
        ]
        mock_prompt.return_value = "1"  # Source: ModelScope

        result = session.run_download_setup()

        # 验证配置正确设置
        assert session.config.model_id == "Qwen/Qwen3-ASR-1.7B"
        assert session.config.source == "ms"
        assert session.config.output_dir == "./models"
        # 返回 True 表示配置完成，准备下载
        assert result is True


def test_execute_download_success():
    """验证下载成功流程"""
    session = DownloadSession()
    session.config.model_id = "Qwen/Qwen3-ASR-1.7B"
    session.config.output_dir = "./models"

    with patch("main.ModelDownloader") as mock_downloader_class:
        mock_downloader = MagicMock()
        mock_downloader.download.return_value = True
        mock_downloader_class.return_value = mock_downloader

        with patch.object(session, "show_continue_menu") as mock_continue:
            mock_continue.return_value = "4"  # 退出

            session.execute_download()

            # 验证添加到历史
            assert "Qwen/Qwen3-ASR-1.7B" in session.download_history


def test_execute_download_failure():
    """验证下载失败流程"""
    session = DownloadSession()
    session.config.model_id = "Qwen/Qwen3-ASR-1.7B"
    session.config.output_dir = "./models"

    with patch("main.ModelDownloader") as mock_downloader_class:
        mock_downloader = MagicMock()
        mock_downloader.download.return_value = False
        mock_downloader_class.return_value = mock_downloader

        session.execute_download()

        # 不应添加到历史
        assert "Qwen/Qwen3-ASR-1.7B" not in session.download_history


def test_execute_download_auth_error_returns_false():
    """测试下载遇到认证错误时返回 False"""
    from main import DownloadSession

    session = DownloadSession()
    session.config.model_id = "facebook/sam3"
    session.config.source = "hf"
    session.config.output_dir = "./models"

    with patch.object(session, "auth_checker") as mock_auth:
        mock_auth.check_model_access.return_value = (True, "Gated model error")
        with patch("main.show_auth_error") as mock_error:
            mock_error.return_value = "2"  # 返回主菜单

            result = session.execute_download()

            assert result is False
            mock_error.assert_called_once_with(
                "facebook/sam3", "hf", "Gated model error"
            )


def test_execute_download_no_auth_check():
    """测试下载公开模型时不检查认证"""
    from main import DownloadSession

    session = DownloadSession()
    session.config.model_id = "Qwen/Qwen3-ASR-1.7B"
    session.config.source = "hf"
    session.config.output_dir = "./models"

    with patch("main.ModelDownloader") as mock_downloader_class:
        mock_downloader = MagicMock()
        mock_downloader.download.return_value = True
        mock_downloader_class.return_value = mock_downloader

        with patch.object(session, "auth_checker") as mock_auth:
            mock_auth.check_model_access.return_value = (False, "")

            session.execute_download()

            # 应该检查认证但不需要认证
            mock_auth.check_model_access.assert_called_once_with(
                "Qwen/Qwen3-ASR-1.7B", "hf"
            )
