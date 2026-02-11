# ModelHub Downloader

公开模型下载工具，支持 HuggingFace 和 ModelScope。

## 准备工作

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 使用

```bash
# 交互式模式（推荐）
python main.py

# 命令行下载
python main.py download Qwen/Qwen3-ASR-1.7B --source ms

# 清理缓存
python main.py clean --all
```

## 常见问题

**下载慢？** 试一下 ModelScope 源，国内访问快一些

**报错 ImportError？** 检查虚拟环境有没有激活

MIT License
