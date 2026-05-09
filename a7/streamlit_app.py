from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import core

st.set_page_config(page_title="A7 自监督学习 Vibe Coding", layout="wide")


def load_default_image() -> np.ndarray:
    for name in ["default_image.jpg", "default_image.jpeg", "default_image.png"]:
        path = ROOT / "assets" / name
        if path.exists():
            return np.asarray(Image.open(path).convert("RGB"))
    return np.zeros((240, 360, 3), dtype=np.uint8) + 235


def show_array(arr, caption: str):
    st.image(np.asarray(arr), caption=caption, use_container_width=True)


st.title("A7 自监督学习 Vibe Coding")
st.caption("学生：裘典儿 2025213456 · Agent/LLM：Codex GPT-5")

uploaded = st.sidebar.file_uploader("上传图片替换默认素材", type=["jpg", "jpeg", "png"])
if uploaded:
    image = np.asarray(Image.open(uploaded).convert("RGB"))
    image_source = f"上传图片：{uploaded.name}"
else:
    image = load_default_image()
    image_source = "默认图片：assets/default_image.jpg"

st.sidebar.image(image, caption=image_source, use_container_width=True)
base_input = np.asarray(Image.fromarray(image).resize((128, 128)))

tab_names = ['旋转预测', 'MAE遮挡重建', 'SimCLR', '部署']
tabs = st.tabs(tab_names)


with tabs[0]:
    st.subheader("旋转预测自监督任务")
    base = base_input
    angle = st.selectbox("旋转角度", [0, 90, 180, 270], index=1)
    res = core.rotate_task(base, angle)
    cols = st.columns(2)
    cols[0].image(base, caption="输入图像", use_container_width=True)
    cols[1].image(res["rotated"], caption=f"旋转 {angle}° 后的自监督输入", use_container_width=True)
    st.line_chart({"loss": res["loss"], "accuracy": res["acc"]})

with tabs[1]:
    st.subheader("MAE 遮挡重建")
    ratio = st.slider("遮挡比例", 0.1, 0.75, 0.45, 0.05)
    res = core.mae_mask(base_input, ratio)
    cols = st.columns(4)
    cols[0].image(base_input, caption=image_source, use_container_width=True)
    cols[1].image(res["masked"], caption="遮挡后", use_container_width=True)
    cols[2].image(res["recon"], caption="重建结果", use_container_width=True)
    cols[3].image(res["heat"], caption="误差热力图", use_container_width=True, clamp=True)
    st.line_chart({"mae_loss": res["loss"]})

with tabs[2]:
    st.subheader("SimCLR 简化对比学习")
    strong = st.toggle("使用更强数据增强", value=False)
    res = core.simclr_embeddings(strong_aug=strong)
    fig, ax = plt.subplots()
    ax.scatter(res["z"][:,0], res["z"][:,1], c=res["labels"], cmap="Set2")
    ax.set_title("2D representation")
    st.pyplot(fig)
    st.line_chart({"contrastive_loss": res["loss"]})


with tabs[-1]:
    st.subheader("部署说明")
    st.write("上传本文件夹到 GitHub 后，如果把该文件夹作为仓库根目录，Streamlit Cloud 的 Main file path 填 `streamlit_app.py`。")
    st.write("如果上传整个周四作业目录，则 Main file path 填 `a7/streamlit_app.py`。")
    st.write("本应用保留上传控件，可用自己的图片替换默认素材。")
