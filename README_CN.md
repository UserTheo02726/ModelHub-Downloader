# ModelHub-Downloader

## 简介

公开模型快速下载工具，支持 HuggingFace 和 ModelScope 平台。
专为解决大模型下载繁琐、速度慢、目录结构混乱等问题设计。
## 功能特性

- ✅ **双平台支持**: 无缝支持 HuggingFace 和 ModelScope
- ✅ **智能缓存管理**: 下载验证后自动清理临时缓存，节省磁盘空间
- ✅ **断点续传**: 支持中断后继续下载，无需重新开始
- ✅ **交互式/命令行**: 支持友好的 CLI 交互界面和自动化脚本调用
- ✅ **完整性校验**: 下载完成后自动验证文件数量和大小

## 环境要求

- Python 3.8+
- Windows / Linux / macOS

## 安装指南

### 1. 准备环境

推荐使用虚拟环境隔离依赖：

```bash
# 创建虚拟环境
python -m venv .venv

# 激活环境

# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 2. 安装依赖

```bash
# (依赖包含: `huggingface_hub`, `modelscope`, `tqdm`, `rich`)
pip install -r requirements.txt
```

## 使用方法

### 🚀 交互式模式 (推荐)

直接运行脚本，按照提示操作即可：
```bash
python model_downloader.py
```

  

### 🛠️ 命令行模式 (自动化)

适合脚本调用或熟练用户：
```bash
# 基本用法
python model_downloader.py -m <模型ID> -s <源> -o <输出目录>

# 示例: 从 HuggingFace 下载 Qwen
python model_downloader.py -m Qwen/Qwen3-ASR-1.7B -s hf -o D:\ComfyUI\models

# 示例: 自动检测源 (优先 HF，失败转 ModelScope)
python model_downloader.py -m Qwen/Qwen3-ASR-1.7B -s auto -o ./models
```

**参数说明:**
- `-m, --model-id`: 模型 ID (格式: `namespace/model_name`)
- `-s, --source`: 下载源 (`hf` | `ms` | `auto`)
- `-o, --output-dir`: 模型保存的根目录
- `-y, --yes`: 非交互模式 (自动确认所有提示)
- `--clean-cache`: 仅清理所有缓存并退出

## 目录结构

工具会自动在输出目录下创建以**模型名**命名的文件夹（与 ModelScope 行为一致），保持原始仓库结构：
```
{output_dir}/
└── {model_name}/      <-- 自动创建
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── ... (保持仓库原始结构)
```

## 缓存管理

本工具内置了智能缓存管理策略：
1. **自动清理**: 每次下载并验证成功后，会自动尝试清理该模型的临时缓存（HuggingFace/ModelScope 默认会保留一份缓存副本，占用双倍空间）。
2. **手动清理**: 运行 `python model_downloader.py --clean-cache` 可一键清理所有下载缓存。

## 常见问题

### Q: 下载速度慢怎么办？

- 尝试切换下载源（如 HuggingFace 慢可尝试 ModelScope）。
- 配置系统代理（工具会自动尊重系统代理设置）。

### Q: 报错 "ImportError"?

- 请检查是否已激活虚拟环境，并执行了 `pip install -r requirements.txt`。

## 许可证

MIT