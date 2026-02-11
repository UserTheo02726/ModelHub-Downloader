# ModelHub Downloader

公开模型下载工具，支持 HuggingFace 和 ModelScope。

## 安装

```bash
pip install -r requirements.txt
```

## 使用

```bash
# 交互式模式 (推荐)
python main.py

# 命令行下载
python main.py download Qwen/Qwen3-ASR-1.7B

# 清理缓存
python main.py clean --all
```

## 常见问题

**下载慢？** 尝试 ModelScope 源（国内更快）

**报错 ImportError？** 检查虚拟环境是否激活

MIT License
