# Vibe Coding 图像滤波

本目录是 `a2` 作业提交材料，包含一个可部署的 Streamlit 交互应用、核心源码、演示截图和 PDF 实验报告。

## 文件说明

- `streamlit_app.py`：Streamlit 交互应用入口。
- `src/image_filtering.py`：Box、Gaussian、Median、Sobel、局部梯度方向、FFT 频域滤波与谱图变换对比的核心源码。
- `assets/default_image.jpg`：应用默认图片，上传图片后可替换。
- `outputs/screenshots/`：用于报告的结果截图。
- `reports/final_report.pdf`：PDF 实验报告。
- `reports/final_report.html`：同内容 HTML 版本。
- `prompt.txt`：本次 Vibe Coding 使用的 prompt 文本。
- `requirements.txt`：部署依赖。

## 本地运行

```bash
/Users/qiudianer/周四作业/.venv/bin/streamlit run /Users/qiudianer/周四作业/a2/streamlit_app.py
```

## Streamlit Cloud 部署

如果把 `a2` 文件夹内容作为 GitHub 仓库根目录上传，Main file path 填：

```text
streamlit_app.py
```

如果上传整个作业目录，Main file path 填：

```text
a2/streamlit_app.py
```

使用 Agent/LLM：Codex GPT-5。
