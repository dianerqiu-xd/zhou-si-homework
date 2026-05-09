from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from vibe_feature_core import (
    canny_visualization,
    feature_comparison,
    match_pipeline,
    stitch_many,
    stitch_pair,
)


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
HOMEWORK_IMAGES = ROOT.parent / "图片"
MAX_DISPLAY_PIXELS = 12_000_000
MAX_DISPLAY_SIDE = 2200


st.set_page_config(
    page_title="Vibe Coding 图像特征检测与匹配",
    page_icon="A3",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; }
    h1, h2, h3 { letter-spacing: 0; }
    div[data-testid="stMetric"] { background: #171b22; border: 1px solid #303744; border-radius: 8px; padding: 10px 12px; }
    div[data-testid="stMetric"] label, div[data-testid="stMetric"] [data-testid="stMetricLabel"], div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #f5f7fb !important; }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] { color: #9aa4b2 !important; }
    .caption { color: #667085; font-size: 0.92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def read_upload(uploaded_file) -> np.ndarray:
    image = Image.open(uploaded_file).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def synthetic_asset(name: str) -> np.ndarray:
    seed = sum(ord(ch) for ch in name)
    rng = np.random.default_rng(seed)
    canvas = np.full((420, 640, 3), (238, 232, 221), dtype=np.uint8)
    cv2.rectangle(canvas, (40, 70), (600, 360), (205, 225, 238), -1)
    cv2.circle(canvas, (180, 180), 64, (82, 137, 191), -1)
    cv2.circle(canvas, (450, 250), 72, (184, 103, 91), -1)
    cv2.line(canvas, (80, 330), (560, 110), (54, 86, 112), 6)
    for _ in range(42):
        center = tuple(rng.integers([60, 60], [580, 360]).tolist())
        color = tuple(int(v) for v in rng.integers(40, 210, size=3))
        cv2.circle(canvas, center, int(rng.integers(4, 13)), color, -1)
    shift_map = {"1": -70, "left": -70, "2": 0, "mid": 0, "3": 70, "right": 70}
    shift = next((value for key, value in shift_map.items() if key in name), 0)
    matrix = np.float32([[1, 0, shift], [0, 1, 0]])
    return cv2.warpAffine(canvas, matrix, (640, 420), borderMode=cv2.BORDER_REFLECT)


def read_asset(name: str, prefer_homework: bool = False) -> np.ndarray:
    paths = [HOMEWORK_IMAGES / name, ASSETS / name] if prefer_homework else [ASSETS / name, HOMEWORK_IMAGES / name]
    for path in paths:
        image = cv2.imread(str(path))
        if image is not None:
            return image
    return synthetic_asset(name)


def fit_for_display(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    pixels = h * w
    if pixels <= MAX_DISPLAY_PIXELS and max(h, w) <= MAX_DISPLAY_SIDE:
        return image
    scale = min(MAX_DISPLAY_SIDE / max(h, w), (MAX_DISPLAY_PIXELS / pixels) ** 0.5)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def default_panorama_samples() -> list[np.ndarray]:
    return [read_asset("boat1.jpg"), read_asset("boat2.jpg"), read_asset("boat3.jpg")]


def show_bgr(image: np.ndarray | None, caption: str = "", width: int | None = None) -> None:
    if image is None:
        st.warning("当前步骤未生成图像。")
        return
    image = fit_for_display(image)
    image_kwargs = {"caption": caption}
    if width is not None:
        image_kwargs["width"] = width
    if image.ndim == 2:
        st.image(image, clamp=True, **image_kwargs)
    else:
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), **image_kwargs)


def image_picker(label: str, sample_name: str, key: str, homework_name: str | None = None) -> np.ndarray | None:
    options = []
    if homework_name and (HOMEWORK_IMAGES / homework_name).exists():
        options.append("使用作业图片")
    options.extend(["使用内置示例", "上传图片"])
    source = st.radio(
        label,
        options,
        horizontal=True,
        key=f"{key}_source_v3",
    )
    if source == "上传图片":
        uploaded = st.file_uploader("选择图片", type=["jpg", "jpeg", "png", "bmp"], key=f"{key}_upload")
        return read_upload(uploaded) if uploaded else None
    if source == "使用作业图片" and homework_name:
        return read_asset(homework_name, prefer_homework=True)
    return read_asset(sample_name)


st.title("Vibe Coding 图像特征检测与匹配")
st.caption("Canny 边缘、Harris/SIFT 特征、RANSAC 匹配、图像对齐与多图全景拼接的一体化交互实验应用")

tab_edges, tab_features, tab_matching, tab_panorama, tab_about = st.tabs(
    ["Canny 边缘", "特征点检测", "两图匹配", "全景拼接", "说明"]
)

with tab_edges:
    left, right = st.columns([1, 1])
    with left:
        edge_image = image_picker("输入图像", "feature_sample.jpg", "edge", "1.jpg")
        low = st.slider("低阈值", 0, 200, 50, key="edge_low")
        high = st.slider("高阈值", 50, 350, 150, key="edge_high")
        sigma = st.slider("高斯 sigma", 0.6, 3.0, 1.2, 0.1, key="edge_sigma")
    if edge_image is not None:
        results = canny_visualization(edge_image, low, high, sigma)
        with right:
            show_bgr(results["original"], "原始图像")
        st.subheader("非最大值抑制前后对比")
        c1, c2, c3 = st.columns(3)
        with c1:
            show_bgr(results["before_nms"], "NMS 前：梯度幅值响应，边缘较厚")
        with c2:
            show_bgr(results["after_nms"], "NMS 后：只保留局部最大响应，边缘变细")
        with c3:
            show_bgr(results["canny_edges"], "Canny 双阈值与连接后的最终边缘")
        show_bgr(results["overlay"], "边缘叠加在原图上的效果")

with tab_features:
    col1, col2 = st.columns([1, 1])
    with col1:
        feat_image = image_picker("输入图像", "feature_sample.jpg", "feature", "2.jpg")
        harris_threshold = st.slider("Harris 响应阈值比例", 0.001, 0.05, 0.01, 0.001)
        sift_nfeatures = st.slider("SIFT 最大特征点数", 100, 1500, 600, 50)
    if feat_image is not None:
        comparison = feature_comparison(feat_image, harris_threshold, sift_nfeatures)
        harris = comparison["harris"]
        sift = comparison["sift"]
        with col2:
            show_bgr(feat_image, "原始图像")
        m1, m2 = st.columns(2)
        m1.metric("Harris 角点数", len(harris["points"]))
        m2.metric(f"{sift['method']} 特征点数", len(sift["keypoints"]))
        c1, c2, c3 = st.columns(3)
        with c1:
            show_bgr(harris["view"], "Harris：绿色中心点 + 蓝色圆形邻域")
        with c2:
            show_bgr(sift["view"], f"{sift['method']}：圆半径表示尺度，方向线表示主方向")
        with c3:
            show_bgr(harris["heatmap"], "Harris 响应热力图")

with tab_matching:
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        match_img1 = image_picker("图像 1", "match_1.jpg", "match1", "3.jpg")
    with img_col2:
        match_img2 = image_picker("图像 2", "match_2.jpg", "match2", "4.jpg")

    p1, p2, p3 = st.columns(3)
    detector = p1.selectbox("特征检测器", ["SIFT", "ORB"], key="match_detector")
    ratio = p2.slider("Lowe ratio", 0.50, 0.95, 0.75, 0.05)
    ransac = p3.slider("RANSAC 重投影阈值", 1.0, 10.0, 4.0, 0.5)

    if match_img1 is not None and match_img2 is not None:
        pipeline = match_pipeline(match_img1, match_img2, detector, ratio, ransac)
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("图像1特征点", len(pipeline.keypoints_1))
        s2.metric("图像2特征点", len(pipeline.keypoints_2))
        s3.metric("初始匹配", len(pipeline.initial_matches))
        s4.metric("RANSAC 内点", len(pipeline.inlier_matches))

        st.subheader("1. 特征点检测与描述")
        c1, c2 = st.columns(2)
        with c1:
            show_bgr(pipeline.feature_view_1, "图像 1 特征点")
        with c2:
            show_bgr(pipeline.feature_view_2, "图像 2 特征点")

        st.subheader("2. 初始匹配与 3. RANSAC 筛选")
        c1, c2 = st.columns(2)
        with c1:
            show_bgr(pipeline.initial_view, "初始匹配")
        with c2:
            show_bgr(pipeline.ransac_view, "RANSAC 内点匹配")

        st.subheader("4. 单应性变换与匹配对齐")
        c1, c2 = st.columns(2)
        with c1:
            show_bgr(pipeline.transform_view, "图像 1 透视变换结果")
        with c2:
            show_bgr(pipeline.aligned_view, "与图像 2 半透明对齐")

with tab_panorama:
    mode = st.segmented_control("拼接模式", ["两幅图像", "多幅图像"], default="多幅图像")
    blend = st.selectbox("Blending 方法", ["feather", "average", "multiband", "none"], index=0)

    if mode == "两幅图像":
        c1, c2 = st.columns(2)
        with c1:
            pano_1 = image_picker("左/基准图", "boat1.jpg", "pano1")
        with c2:
            pano_2 = image_picker("右/待拼接图", "boat2.jpg", "pano2")
        if pano_1 is not None and pano_2 is not None:
            panorama, matches, homography, pipeline = stitch_pair(pano_1, pano_2, blend)
            st.metric("RANSAC 内点", len(pipeline.inlier_matches))
            show_bgr(matches, "用于拼接的内点匹配")
            if panorama is None:
                st.error("拼接失败：请使用具有足够重叠区域的图片。")
            else:
                show_bgr(panorama, f"全景图：{blend}")
    else:
        samples = default_panorama_samples()
        uploaded_many = st.file_uploader(
            "上传多幅同一场景图片，留空则使用三张不同的连续全景示例图",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True,
        )
        images = [read_upload(f) for f in uploaded_many] if uploaded_many else samples
        preview_cols = st.columns(min(len(images), 4))
        for i, img in enumerate(images[:4]):
            with preview_cols[i % len(preview_cols)]:
                show_bgr(img, f"输入图 {i + 1}")
        if len(images) >= 2:
            panorama, steps = stitch_many(images, blending=blend)
            st.subheader("拼接过程")
            for step in steps:
                c1, c2 = st.columns(2)
                with c1:
                    st.caption(f"步骤 {step['step']}：初始匹配 {step['initial_matches']}，内点 {step['inliers']}")
                    show_bgr(step["matches_view"], "RANSAC 匹配")
                with c2:
                    show_bgr(step["panorama"], "中间全景图")
            st.subheader("最终全景图")
            if panorama is None:
                st.error("拼接失败：相邻图像之间需要有明显重叠区域。")
            else:
                show_bgr(panorama, f"最终全景图：{blend}")
        else:
            st.warning("至少需要两幅图片。")

with tab_about:
    st.markdown(
        """
        本应用实现了题目要求的五个核心部分：Canny 边缘检测及 NMS 前后对比、Harris 与 SIFT 特征点可视化、两幅图像的检测-描述-匹配-RANSAC-变换-对齐流程、多幅图像全景拼接，以及不同 blending 方法对比。

        使用的 Agent/LLM：Codex GPT-5。

        本地运行命令：

        ```bash
        streamlit run app.py
        ```
        """
    )
