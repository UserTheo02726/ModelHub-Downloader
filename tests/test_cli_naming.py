"""测试 CLI 命令命名规范"""

from main import app


def test_download_command_exists():
    """验证 download 命令存在"""
    commands = [cmd.name for cmd in app.registered_commands]
    assert "download" in commands


def test_clean_command_exists():
    """验证 clean 命令存在"""
    commands = [cmd.name for cmd in app.registered_commands]
    assert "clean" in commands


def test_list_command_exists():
    """验证 list 命令存在"""
    commands = [cmd.name for cmd in app.registered_commands]
    assert "list" in commands


def test_interactive_command_exists():
    """验证 interactive 命令存在"""
    commands = [cmd.name for cmd in app.registered_commands]
    assert "interactive" in commands
