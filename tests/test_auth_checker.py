"""测试 AuthChecker 认证检查功能"""

import pytest
from unittest.mock import patch, MagicMock


def test_auth_checker_public_model_returns_false():
    """测试公开模型不需要认证"""
    from main import AuthChecker

    checker = AuthChecker()

    # Mock HfApi.model_info 返回成功
    with patch("main.HfApi") as mock_api:
        mock_api.return_value.model_info.return_value = MagicMock()

        need_auth, error_msg = checker.check_model_access("Qwen/Qwen3-ASR-1.7B", "hf")

        assert need_auth is False
        assert error_msg == ""
        mock_api.return_value.model_info.assert_called_once_with(
            "Qwen/Qwen3-ASR-1.7B", files_metadata=False, timeout=10
        )


def test_auth_checker_gated_model_returns_true():
    """测试 Gated Model 需要认证"""
    from main import AuthChecker
    from huggingface_hub.errors import GatedRepoError
    from unittest.mock import Mock

    checker = AuthChecker()

    # 模拟 HTTP 响应
    mock_response = Mock()
    mock_response.status_code = 401

    with patch("main.HfApi") as mock_api:
        mock_api.return_value.model_info.side_effect = GatedRepoError(
            "Gated model: facebook/sam3. Please authenticate first.",
            response=mock_response,
        )

        need_auth, error_msg = checker.check_model_access("facebook/sam3", "hf")

        assert need_auth is True
        assert "Gated model" in error_msg


def test_auth_checker_401_returns_true():
    """测试 401 错误需要认证"""
    from main import AuthChecker
    from httpx import HTTPStatusError
    from unittest.mock import Mock

    checker = AuthChecker()

    # 创建模拟的 HTTPStatusError
    mock_response = Mock()
    mock_response.status_code = 401

    with patch("main.HfApi") as mock_api:
        mock_api.return_value.model_info.side_effect = HTTPStatusError(
            "401 Unauthorized", request=Mock(), response=mock_response
        )

        need_auth, error_msg = checker.check_model_access("test/model", "hf")

        assert need_auth is True
        assert "401" in error_msg


def test_auth_checker_ms_model_returns_false():
    """测试 ModelScope 模型不需要认证检查"""
    from main import AuthChecker

    checker = AuthChecker()

    need_auth, error_msg = checker.check_model_access("Qwen/Qwen3-ASR-1.7B", "ms")

    assert need_auth is False
    assert error_msg == ""


def test_auth_checker_auto_model_returns_false():
    """测试 Auto 模式不需要认证检查"""
    from main import AuthChecker

    checker = AuthChecker()

    need_auth, error_msg = checker.check_model_access("Qwen/Qwen3-ASR-1.7B", "auto")

    assert need_auth is False
    assert error_msg == ""
