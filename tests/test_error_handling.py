"""测试异常处理和日志记录"""

import pytest
from unittest.mock import Mock, patch
from main import ModelDownloader, DownloadError, ValidationError


def test_download_hf_logs_exception():
    """验证 download_hf 记录异常日志"""
    d = ModelDownloader()
    with patch("main.HfApi") as mock_api:
        mock_api.return_value.model_info.side_effect = Exception("API Error")
        with patch("main.snapshot_download") as mock_download:
            mock_download.side_effect = Exception("Network error")
            # 现在抛出异常
            with pytest.raises(DownloadError):
                d.download_hf("Qwen/Qwen3-ASR-1.7B")


def test_download_validates_model_id():
    """验证 download 方法验证 model_id，无效时抛出异常"""
    d = ModelDownloader()
    # 无效 ID (无 /) - 现在抛出异常
    with pytest.raises(ValidationError):
        d.download("invalid_id")

    # 无效 ID (多个 /) - 现在抛出异常
    with pytest.raises(ValidationError):
        d.download("a/b/c")


def test_download_raises_on_critical_error():
    """验证严重错误应该抛出异常而不是静默"""
    d = ModelDownloader()
    with patch("main.snapshot_download") as mock_download:
        mock_download.side_effect = Exception("Network error")
        # 应该抛出 DownloadError 而不是静默返回 False
        with pytest.raises(DownloadError):
            d.download_hf("Qwen/Qwen3-ASR-1.7B")
