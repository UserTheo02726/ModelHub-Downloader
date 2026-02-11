"""测试强化模型 ID 输入验证"""

import pytest
from main import ModelDownloader, model_id_validator, ValidationError


def test_model_id_validator_function():
    """验证 model_id_validator 函数存在且工作"""
    # 有效 ID
    assert model_id_validator("Qwen/Qwen3-ASR-1.7B") == True
    assert model_id_validator("facebook/opt-1.3b") == True
    assert model_id_validator("THUDM/chatglm3-6b") == True

    # 无效 ID - 缺少组织名
    assert model_id_validator("model-only") == False

    # 无效 ID - 多个 /
    assert model_id_validator("a/b/c") == False

    # 无效 ID - 包含特殊字符
    assert model_id_validator("org/model?name") == False
    assert model_id_validator("org/model#tag") == False

    # 无效 ID - 过长 (模型名超过 200)
    long_id = "a/" + "b" * 201
    assert model_id_validator(long_id) == False


def test_validate_method_uses_validator():
    """验证 ModelDownloader.validate 使用 validator"""
    d = ModelDownloader()
    # 有效
    assert d.validate("Qwen/Qwen3-ASR-1.7B") == True

    # 无效
    assert d.validate("invalid") == False


def test_validate_rejects_empty():
    """验证 validate 拒绝空值"""
    d = ModelDownloader()
    assert d.validate("") == False
