from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.image_filtering import (  # noqa: E402
    box_filter,
    compare_spatial_filters,
    frequency_filter,
    gaussian_filter,
    gradient_region_demo,
    median_filter,
    normalize_to_u8,
    rgb_to_gray,
    sobel_edge_image,
    transform_spectra,
)


st.set_page_config(page_title="Vibe Coding 图像滤波实验", layout="wide")


def decode_upload(uploaded_file) -> np.ndarray:
    data = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("图片读取失败，请上传 jpg、jpeg 或 png 文件。")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_default_image() -> np.ndarray:
    for default_path in (ROOT / "assets" / "default_image.jpg", ROOT / "assets" / "default_image.png"):
        if default_path.exists():
            return np.asarray(Image.open(default_path).convert("RGB"))
    x = np.linspace(0, 255, 420, dtype=np.uint8)
    y = np.linspace(0, 255, 300, dtype=np.uint8)
    xx, yy = np.meshgrid(x, y)
    return np.stack([xx, yy, 255 - xx // 2], axis=2)


def resize_for_interaction(rgb: np.ndarray, max_side: int = 640) -> np.ndarray:
    image = Image.fromarray(rgb)
    image.thumbnail((max_side, max_side))
    return np.asarray(image)


def show_metric_table(images: dict[str, np.ndarray]) -> None:
    rows = []
    for name, image in images.items():
        gray = rgb_to_gray(image) if image.ndim == 3 else image.astype(np.float32)
        rows.append(
            {
                "方法": name,
                "均值": round(float(np.mean(gray)), 2),
                "标准差": round(float(np.std(gray)), 2),
                "平均梯度": round(float(np.mean(sobel_edge_image(np.stack([gray] * 3, axis=2).astype(np.uint8)))), 2),
            }
        )
    st.dataframe(rows, hide_index=True, use_container_width=True)


def plot_histogram(images: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for name, image in images.items():
        if name == "Sobel":
            continue
        gray = rgb_to_gray(image) if image.ndim == 3 else image
        ax.hist(gray.ravel(), bins=48, alpha=0.45, label=name)
    ax.set_xlabel("Gray level")
    ax.set_ylabel("Pixels")
    ax.legend()
    st.pyplot(fig, clear_figure=True)


def main() -> None:
    st.title("Vibe Coding 图像滤波实验")

    uploaded_file = st.sidebar.file_uploader("上传测试图片", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            rgb = decode_upload(uploaded_file)
        except ValueError as exc:
            st.error(str(exc))
            return
        image_source = "上传图片"
    else:
        rgb = load_default_image()
        image_source = "默认图片"

    rgb = resize_for_interaction(rgb)

    st.sidebar.markdown("### 空间滤波参数")
    kernel_size = st.sidebar.slider("窗口大小", min_value=3, max_value=11, value=5, step=2)
    sigma = st.sidebar.slider("Gaussian sigma", min_value=0.5, max_value=4.0, value=1.2, step=0.1)

    st.sidebar.markdown("### 梯度 ROI")
    h, w = rgb.shape[:2]
    roi_x = st.sidebar.slider("ROI 左上角 x", 0, max(0, w - 10), max(0, w // 4))
    roi_y = st.sidebar.slider("ROI 左上角 y", 0, max(0, h - 10), max(0, h // 4))
    roi_w = st.sidebar.slider("ROI 宽度", 10, max(10, w - roi_x), max(10, w // 3))
    roi_h = st.sidebar.slider("ROI 高度", 10, max(10, h - roi_y), max(10, h // 3))

    st.sidebar.markdown("### 频域滤波参数")
    freq_mode = st.sidebar.selectbox("频域滤波器", ["low-pass", "high-pass", "band-pass"])
    radius_ratio = st.sidebar.slider("截止半径比例", min_value=0.05, max_value=0.6, value=0.22, step=0.01)

    tab_input, tab_spatial, tab_gradient, tab_frequency, tab_transform, tab_submit = st.tabs(
        ["原图", "空间滤波", "梯度演示", "频域滤波", "谱图变化", "提交说明"]
    )

    with tab_input:
        st.image(rgb, caption=f"{image_source}: {w} x {h}", use_container_width=True)

    with tab_spatial:
        spatial = compare_spatial_filters(rgb, kernel_size, sigma)
        cols = st.columns(5)
        for col, (name, image) in zip(cols, spatial.items()):
            col.image(image, caption=name, use_container_width=True)
        st.subheader("数值比较")
        show_metric_table(spatial)
        plot_histogram(spatial)

    with tab_gradient:
        gd = gradient_region_demo(rgb, roi_x, roi_y, roi_w, roi_h)
        cols = st.columns(3)
        cols[0].image(gd["boxed"], caption="框选局部区域与平均梯度方向", use_container_width=True)
        cols[1].image(normalize_to_u8(gd["magnitude"]), caption="Sobel 梯度幅值", use_container_width=True)
        cols[2].image(normalize_to_u8(gd["angle"]), caption="梯度方向角可视化", use_container_width=True)
        st.metric("ROI 平均梯度方向", f"{gd['mean_angle']:.2f} deg")
        st.metric("ROI 平均梯度强度", f"{gd['mean_magnitude']:.2f}")

    with tab_frequency:
        freq = frequency_filter(rgb, freq_mode, radius_ratio)
        cols = st.columns(5)
        for col, key, caption in zip(
            cols,
            ["gray", "spectrum", "mask", "filtered_spectrum", "restored"],
            ["灰度图", "中心化频谱", "频域掩膜", "滤波后频谱", "反变换结果"],
        ):
            col.image(freq[key], caption=caption, use_container_width=True)

    with tab_transform:
        spectra = transform_spectra(rgb)
        row1 = st.columns(4)
        row2 = st.columns(4)
        captions = {
            "original": "原图",
            "translate": "平移",
            "rotate": "旋转",
            "scale": "缩放",
        }
        for col, (name, (image, _)) in zip(row1, spectra.items()):
            col.image(image, caption=captions[name], use_container_width=True)
        for col, (name, (_, spectrum)) in zip(row2, spectra.items()):
            col.image(spectrum, caption=f"{captions[name]}频谱", use_container_width=True)

    with tab_submit:
        st.markdown(
            """
            ### 部署与提交

            - Streamlit 入口文件：`streamlit_app.py`
            - 核心源码：`src/image_filtering.py`
            - 默认图片：`assets/default_image.jpg`
            - 报告与截图：`reports/`、`outputs/screenshots/`
            - 使用 Agent/LLM：Codex GPT-5

            若将 `a2` 文件夹内容作为 GitHub 仓库根目录上传，Streamlit Community Cloud 的 Main file path 填 `streamlit_app.py`。
            若上传整个作业目录，则 Main file path 填 `a2/streamlit_app.py`。
            """
        )


if __name__ == "__main__":
    main()
