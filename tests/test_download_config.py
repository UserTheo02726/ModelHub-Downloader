"""测试 DownloadConfig 数据类"""

import pytest
from main import DownloadConfig, SOURCE_MS, DEFAULT_OUTPUT


def test_default_values():
    """验证默认配置值"""
    config = DownloadConfig()
    assert config.model_id == ""
    assert config.source == SOURCE_MS
    assert config.output_dir == DEFAULT_OUTPUT


def test_is_complete_false_when_empty():
    """验证 is_complete 在配置为空时返回 False"""
    config = DownloadConfig()
    assert config.is_complete() == False


def test_is_complete_true_when_model_id_set():
    """验证 is_complete 在设置 model_id 后返回 True"""
    config = DownloadConfig()
    config.model_id = "Qwen/Qwen3-ASR-1.7B"
    assert config.is_complete() == True


def test_reset_restores_defaults():
    """验证 reset 方法恢复默认值"""
    config = DownloadConfig()
    config.model_id = "custom/model"
    config.source = "hf"
    config.output_dir = "/custom/path"

    config.reset()

    assert config.model_id == ""
    assert config.source == SOURCE_MS
    assert config.output_dir == DEFAULT_OUTPUT
