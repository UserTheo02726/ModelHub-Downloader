"""测试自定义异常类和路径验证器"""

import pytest
from main import (
    ModelDownloadError,
    ValidationError,
    DownloadError,
    path_validator,
)


def test_custom_exceptions_exist():
    """验证自定义异常类存在"""
    assert issubclass(ModelDownloadError, Exception)
    assert issubclass(ValidationError, ModelDownloadError)
    assert issubclass(DownloadError, ModelDownloadError)


def test_path_validator_function():
    """验证路径验证器存在且工作"""
    # 有效路径
    assert path_validator("/valid/path") == True
    assert path_validator("./relative") == True

    # 无效路径 - 路径遍历攻击
    assert path_validator("../../../etc/passwd") == False
    assert path_validator("/valid/../../etc") == False


def test_path_validator_with_special_chars():
    """验证路径验证器阻止特殊字符"""
    assert path_validator("/path with spaces") == False
    assert path_validator("/path\twith\ttabs") == False


def test_validation_error_details():
    """验证 ValidationError 包含详细信息"""
    error = ValidationError("model_id", "invalid", "Must be in org/model format")
    assert error.field == "model_id"
    assert error.value == "invalid"
    assert error.reason == "Must be in org/model format"
