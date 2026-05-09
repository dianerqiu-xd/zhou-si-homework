# 图像颜色空间与图像插值实验

本目录为“Vibe Coding 图像颜色空间、图像插值”作业提交材料。

## 文件说明

- `prompt.txt`：Vibe Coding 使用的纯文本 prompt。
- `streamlit_app.py`：Streamlit 云端部署入口文件。
- `assets/default_image.jpg`：网站打开时默认显示的图片，上传图片后会被替换。
- `requirements.txt`：Streamlit Community Cloud 安装依赖。
- `src/image_color_interpolation.py`：核心算法 Python 源码。
- `outputs/color_spaces/`：RGB、HSV、YCrCb、Lab 各通道输出图。
- `outputs/interpolation/`：最近邻、双线性插值的放大、缩小、旋转拉伸结果。
- `outputs/screenshots/`：用于提交的测试截图拼图。
- `reports/作业小结.md`：作业小结。

## 本地生成离线结果

在 `/Users/qiudianer/周四作业` 目录下运行：

```bash
.venv/bin/python a1/src/image_color_interpolation.py \
  --input a5.jpg \
  --output a1/outputs
```

## 本地运行 Streamlit

在 `/Users/qiudianer/周四作业` 目录下运行：

```bash
.venv/bin/streamlit run a1/streamlit_app.py
```

## 部署到 Streamlit Community Cloud

1. 将 `a1` 文件夹中的内容上传到 GitHub 仓库。
2. 打开 Streamlit Community Cloud，新建 App。
3. Repository 选择你的 GitHub 仓库。
4. Branch 选择对应分支，例如 `main`。
5. Main file path 填写：

```text
streamlit_app.py
```

6. Deploy 后即可得到一个公网 URL，不是本地 localhost 地址。

## 依赖

使用 Python、Streamlit、OpenCV、NumPy、Matplotlib、Pillow。依赖已写入 `requirements.txt`。
