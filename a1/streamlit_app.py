from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.image_color_interpolation import (  # noqa: E402
    bilinear_resize,
    nearest_resize,
    normalize_to_u8,
    rotate_and_stretch,
    rgb_to_hsv_manual,
)


st.set_page_config(page_title="图像颜色空间与插值实验", layout="wide")


def decode_upload(uploaded_file) -> np.ndarray:
    data = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("图片读取失败，请上传 jpg、jpeg 或 png 文件。")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_default_image() -> np.ndarray:
    default_candidates = [
        ROOT / "assets" / "default_image.jpg",
        ROOT / "assets" / "default_image.png",
        ROOT / "outputs" / "input_rgb.png",
    ]
    for default_path in default_candidates:
        if default_path.exists():
            return np.asarray(Image.open(default_path).convert("RGB"))

    x = np.linspace(0, 255, 360, dtype=np.uint8)
    y = np.linspace(0, 255, 240, dtype=np.uint8)
    grid_x, grid_y = np.meshgrid(x, y)
    return np.stack([grid_x, grid_y, 255 - grid_x // 2], axis=2).astype(np.uint8)


def channel_image(channel: np.ndarray) -> np.ndarray:
    return normalize_to_u8(channel)


def show_channels(title: str, image: np.ndarray, names: tuple[str, str, str]) -> None:
    st.subheader(title)
    cols = st.columns(3)
    for col, name, index in zip(cols, names, range(3)):
        col.image(channel_image(image[:, :, index]), caption=f"{title} - {name}", use_container_width=True)


def resized_demo_image(rgb: np.ndarray, max_side: int = 420) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return rgb
    return bilinear_resize(rgb, max(1, int(w * scale)), max(1, int(h * scale)))


def main() -> None:
    st.title("图像颜色空间与图像插值实验")

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

    st.sidebar.markdown("### 插值参数")
    scale = st.sidebar.slider("缩放比例", min_value=0.3, max_value=2.5, value=1.5, step=0.1)
    angle = st.sidebar.slider("旋转角度", min_value=-60, max_value=60, value=25, step=5)
    stretch_x = st.sidebar.slider("横向拉伸", min_value=0.5, max_value=2.0, value=1.2, step=0.1)
    stretch_y = st.sidebar.slider("纵向拉伸", min_value=0.5, max_value=2.0, value=0.8, step=0.1)

    tab_input, tab_color, tab_interp, tab_report = st.tabs(["原图", "颜色空间", "图像插值", "提交说明"])

    with tab_input:
        st.image(rgb, caption=f"{image_source}：{rgb.shape[1]} x {rgb.shape[0]}", use_container_width=True)

    with tab_color:
        hsv = rgb_to_hsv_manual(rgb)
        ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        show_channels("RGB", rgb, ("R", "G", "B"))
        show_channels("HSV 手写转换", hsv, ("H", "S", "V"))
        show_channels("YCrCb", ycrcb, ("Y", "Cr", "Cb"))
        show_channels("Lab", lab, ("L", "a", "b"))

    with tab_interp:
        demo = resized_demo_image(rgb)
        h, w = demo.shape[:2]
        target_w = max(1, int(w * scale))
        target_h = max(1, int(h * scale))

        nearest_scaled = nearest_resize(demo, target_w, target_h)
        bilinear_scaled = bilinear_resize(demo, target_w, target_h)
        nearest_transform = rotate_and_stretch(demo, angle, stretch_x, stretch_y, "nearest")
        bilinear_transform = rotate_and_stretch(demo, angle, stretch_x, stretch_y, "bilinear")

        st.subheader("缩放插值")
        cols = st.columns(2)
        cols[0].image(nearest_scaled, caption="最近邻插值缩放", use_container_width=True)
        cols[1].image(bilinear_scaled, caption="双线性插值缩放", use_container_width=True)

        st.subheader("旋转拉伸")
        cols = st.columns(2)
        cols[0].image(nearest_transform, caption="最近邻插值旋转拉伸", use_container_width=True)
        cols[1].image(bilinear_transform, caption="双线性插值旋转拉伸", use_container_width=True)

    with tab_report:
        st.markdown(
            """
            ### 云端部署说明

            本项目可以上传到 GitHub 后部署到 Streamlit Community Cloud。

            - 入口文件：`streamlit_app.py`
            - 依赖文件：`requirements.txt`
            - 核心源码：`src/image_color_interpolation.py`
            - 提交材料：`prompt.txt`、`reports/作业小结.md`、`outputs/screenshots/`

            部署后得到的 URL 是公网地址，不是本地 `localhost` 地址。
            """
        )


if __name__ == "__main__":
    main()
