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

st.set_page_config(page_title="A8 生成模型与潜空间 Vibe Coding", layout="wide")


def load_default_image() -> np.ndarray:
    for name in ["default_image.png", "default_image.jpg", "default_image.jpeg"]:
        path = ROOT / "assets" / name
        if path.exists():
            return np.asarray(Image.open(path).convert("RGB"))
    return np.zeros((240, 360, 3), dtype=np.uint8) + 235


def show_array(arr, caption: str):
    st.image(to_display_image(arr), caption=caption, use_container_width=True)


def to_display_image(arr) -> np.ndarray:
    img = np.asarray(arr)
    if np.issubdtype(img.dtype, np.floating):
        min_v = float(np.nanmin(img))
        max_v = float(np.nanmax(img))
        if min_v < 0.0 or max_v > 1.0:
            img = (img - min_v) / (max_v - min_v + 1e-8)
        img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    return img


st.title("A8 生成模型与潜空间 Vibe Coding")
st.caption("学生：裘典儿 2025213456 · Agent/LLM：Codex GPT-5")

image = load_default_image()
image_source = "MNIST 示例图：assets/default_image.png"

st.sidebar.image(image, caption=image_source, use_container_width=True)

tab_names = ['AE/VAE', '潜空间', 'GAN/扩散参数', '部署']
tabs = st.tabs(tab_names)


with tabs[0]:
    st.subheader("Autoencoder 与 VAE 重构对比")
    st.image(image, caption=f"数据集示例预览：{image_source}", use_container_width=True)
    label = st.slider("样本编号", 0, 9, 3)
    beta = st.slider("VAE KL 权重 beta", 0.1, 2.0, 0.8, 0.1)
    res = core.ae_vae_compare(label, beta)
    cols = st.columns(4)
    cols[0].image(res["input"], caption="输入", use_container_width=True)
    cols[1].image(res["ae"], caption="AE 重构", use_container_width=True)
    cols[2].image(res["vae"], caption="VAE 重构", use_container_width=True)
    cols[3].image(to_display_image(res["err"]), caption="误差热力图", use_container_width=True)
    st.line_chart({"AE loss": res["ae_loss"], "VAE loss": res["vae_loss"]})

with tabs[1]:
    st.subheader("二维潜空间与插值")
    zx = st.slider("潜变量 z1", -2.0, 2.0, 0.6, 0.1)
    zy = st.slider("潜变量 z2", -2.0, 2.0, 0.7, 0.1)
    pts = core.latent_points()
    fig, ax = plt.subplots()
    ax.scatter(pts["z"][:,0], pts["z"][:,1], c=pts["labels"], cmap="tab10")
    ax.scatter([zx], [zy], c="black", s=120, marker="x")
    st.pyplot(fig)
    st.image(core.generate_from_latent(zx, zy), caption="点击/选择潜空间位置生成图像", use_container_width=False)
    cols = st.columns(7)
    for c, img in zip(cols, core.interpolate((-1.2, -0.5), (1.0, 0.8), 7)):
        c.image(img, use_container_width=True)

with tabs[2]:
    st.subheader("GAN 与文本提示参数实验")
    seed = st.slider("seed", 0, 99, 4)
    grid = core.gan_grid(seed)
    cols = st.columns(4)
    for i, img in enumerate(grid["images"][:8]):
        cols[i % 4].image(img, caption=f"D score {grid['scores'][i]:.2f}", use_container_width=True)
    prompt = st.text_input("prompt", "a bright city")
    negative = st.text_input("negative prompt", "blur")
    steps = st.slider("采样步数", 5, 50, 20)
    guidance = st.slider("guidance scale", 1.0, 12.0, 6.0)
    st.image(core.prompt_image(prompt, negative, seed, steps, guidance), caption="轻量文本到图像参数示例", use_container_width=False)


with tabs[-1]:
    st.subheader("部署说明")
    st.write("上传本文件夹到 GitHub 后，如果把该文件夹作为仓库根目录，Streamlit Cloud 的 Main file path 填 `streamlit_app.py`。")
    st.write("如果上传整个周四作业目录，则 Main file path 填 `a8/streamlit_app.py`。")
    st.write("本应用使用内置 MNIST 风格样本和参数控件完成重构、潜空间、GAN 与 prompt 实验展示。")
