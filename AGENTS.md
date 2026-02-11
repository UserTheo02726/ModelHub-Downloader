# PROJECT KNOWLEDGE BASE

**Generated:** 2026-02-11
**Project:** ModelHub-Downloader (单文件 CLI 工具)

## OVERVIEW

公开模型快速下载工具，支持 HuggingFace 和 ModelScope 双平台下载。

## STRUCTURE

```
./
├── ModelHub-Downloader.py    # 唯一入口文件 (620行)
├── requirements.txt           # 依赖: huggingface_hub, modelscope, tqdm, rich
├── README.md                  # 用户文档
└── LICENSE                    # MIT
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| 主程序 | `ModelHub-Downloader.py` | 单一入口 |
| CLI 参数 | `main()` (L570) | argparse 配置 |
| 下载逻辑 | `ModelDownloader` 类 | 核心业务逻辑 |
| 验证/缓存 | `_verify_download()`, `_auto_clean_*_cache()` | 完整性校验 |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `ModelDownloader` | Class | L64 | 下载器主类 |
| `_download_from_hf()` | Method | L327 | HuggingFace 下载 |
| `_download_from_ms()` | Method | L393 | ModelScope 下载 |
| `_verify_download()` | Method | L216 | 完整性校验 |
| `_auto_clean_hf_cache()` | Method | L239 | HF 缓存清理 |
| `_auto_clean_ms_cache()` | Method | L262 | MS 缓存清理 |
| `main()` | Function | L570 | CLI 入口 |
| `run()` | Method | L495 | 主循环 |

## ANTI-PATTERNS

- **README 与文件名不一致**: README 引用 `model_downloader.py`，实际为 `ModelHub-Downloader.py`
- **空目录**: `src/`, `test_download/` 存在但无代码
- **缺少版本声明**: 无 `__version__` 变量

## UNIQUE STYLES

- **注释标记**: 使用 `★` 标记核心代码 (L362)
- **分隔线**: 使用 `'═' * 50` 视觉分隔 (L511)
- **键盘中断处理**: 明确处理 `KeyboardInterrupt` 和 `EOFError`

## COMMANDS

```bash
# 开发运行
python ModelHub-Downloader.py

# 交互模式
python ModelHub-Downloader.py -m Qwen/Qwen3-ASR-1.7B -s auto -o ./models

# 清理缓存
python ModelHub-Downloader.py --clean-cache
```

## NOTES

- 单文件架构，无需分包
- 依赖检查在模块加载时执行 (L24-50)
- 缓存尊重 `HF_HOME` 和 `MODELSCOPE_CACHE` 环境变量
