# Vibe Coding 图像特征检测与匹配

本目录是 A3 新题目的完整提交文件夹。

## 文件说明

- `app.py`：Streamlit 桌面/Web 交互应用入口。
- `vibe_feature_core.py`：核心算法源码，包含 Canny/NMS、Harris、SIFT、特征匹配、RANSAC、图像对齐和全景拼接。
- `make_report.py`：生成实验截图和 PDF 报告。
- `assets/`：内置示例图像。
- `screenshots/`：由核心代码生成的实验结果截图。
- `outputs/`：PDF 报告和中间输出。

## 运行方式

```bash
streamlit run app.py
```

Streamlit Cloud 部署时请确保仓库根目录或 app 所在目录包含：

- `requirements.txt`：使用 `opencv-python-headless`，不要使用 `opencv-python`
- `runtime.txt`：固定为 `python-3.11`
- `packages.txt`：给 OpenCV 补充 Linux 系统库

如果 GitHub 仓库根目录就是本 `a3` 文件夹，Streamlit Cloud 的 Main file path 应填写：

```text
app.py
```

不要再填写 `a3/app.py`。

如果使用本机已有虚拟环境，可运行：

```bash
/Users/qiudianer/Desktop/pycharm/.venv311/bin/streamlit run app.py
```

## 使用的 Agent/LLM

Codex GPT-5。
