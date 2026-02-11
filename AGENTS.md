# PROJECT KNOWLEDGE BASE

**Generated:** 2026-02-11
**Commit:** a3bd8c9
**Branch:** main

## OVERVIEW

公开模型快速下载 CLI 工具，支持 HuggingFace 和 ModelScope 双平台下载。使用 Typer + Rich 构建，无认证架构。

## STRUCTURE

```
./
├── main.py              # CLI入口 (Typer)
├── core/
│   └── downloader.py    # 核心下载逻辑 (ModelDownloader类)
├── utils/
│   └── progress.py      # Rich进度条工具
├── requirements.txt     # 依赖: typer, rich, huggingface_hub, modelscope
├── README.md            # 用户文档
└── LICENSE              # MIT
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| CLI入口 | `main.py` | Typer app，4个子命令 |
| 下载核心 | `core/downloader.py` | ModelDownloader类，双源支持 |
| 进度条 | `utils/progress.py` | Rich进度条封装 |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `app` | Typer | main.py | CLI应用 |
| `download()` | Command | main.py | 下载子命令 |
| `interactive()` | Command | main.py | 交互模式 |
| `list_sources()` | Command | main.py | 列出可用源 |
| `clean_cache()` | Command | main.py | 清理缓存 |
| `ModelDownloader` | Class | core/downloader.py | 下载器 |
| `download_hf()` | Method | core/downloader.py | HF下载 |
| `download_ms()` | Method | core/downloader.py | MS下载 |
| `validate_model_id()` | Method | core/downloader.py | ID验证 |

## CONVENTIONS

- **默认源**: ModelScope (`--source ms`)
- **CLI风格**: 使用Typer，命令分组
- **无认证**: 直接使用huggingface_hub/modelscope公开API

## ANTI-PATTERNS

- **禁止**: 修改`.ref/`目录（参考文件）
- **禁止**: 提交`__pycache__/`, `.ruff_cache/`

## COMMANDS

```bash
# 安装依赖
pip install -r requirements.txt

# CLI使用
python main.py --help
python main.py download Qwen/Qwen3-ASR-1.7B --source ms
python main.py interactive
python main.py clean --all
```

## NOTES

- ModelScope为中国用户推荐源（默认）
- 无需API密钥即可下载公开模型
- 缓存尊重`HF_HOME`和`MODELSCOPE_CACHE`环境变量
