"""测试文件路径安全验证"""

import pytest
from unittest.mock import patch
from main import ModelDownloader, path_validator, ValidationError
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

    # 恶意路径遍历 - 应该抛出异常
    with pytest.raises(ValidationError):
        d.get_path("../../../etc/passwd")

    with pytest.raises(ValidationError):
        d.get_path("valid/../../etc/passwd")


def test_download_validates_output_path():
    """验证 download 命令验证输出路径"""
    d = ModelDownloader(output_dir="../../sensitive")

    # mock 掉实际下载，只测试路径验证
    with patch("main.snapshot_download") as mock_download:
        with pytest.raises(ValidationError):
            d.download_hf("Qwen/Qwen3-ASR-1.7B")
