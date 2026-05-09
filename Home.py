from __future__ import annotations

import streamlit as st


st.set_page_config(
    page_title="周四作业 A1-A8 汇总展示",
    page_icon="A1",
    layout="wide",
)


APPS = [
    ("A1", "图像颜色空间与插值", "颜色空间通道、最近邻/双线性插值、旋转拉伸交互实验。", "pages/01_A1_Color_Interpolation.py"),
    ("A2", "图像滤波", "空间滤波、梯度 ROI、频域滤波和谱图变换可视化。", "pages/02_A2_Image_Filtering.py"),
    ("A3", "图像特征检测与匹配", "Canny、Harris/SIFT、RANSAC 匹配、图像对齐与全景拼接。", "pages/03_A3_Feature_Matching.py"),
    ("A4", "机器学习基础可视化", "Least Squares、KNN、线性分类器、优化器与损失函数演示。", "pages/04_A4_ML_Visual_Lab.py"),
    ("A5", "图像识别与神经网络", "HOG+BOW+SVM、CNN/LeNet、ResNet 深度对比与云端 fallback。", "pages/05_A5_Neural_Networks.py"),
    ("A6", "语义分割与目标检测", "FCN、R-CNN 系列、Mask R-CNN 与方法性能对比。", "pages/06_A6_Detection_Segmentation.py"),
    ("A7", "自监督学习", "旋转预测、MAE 遮挡重建、SimCLR 对比学习交互展示。", "pages/07_A7_Self_Supervised.py"),
    ("A8", "生成模型与潜空间", "AE/VAE、二维潜空间插值、GAN 与 prompt 参数实验。", "pages/08_A8_Generative_Models.py"),
]


st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    .app-card {
        border: 1px solid #d8dee9;
        border-radius: 8px;
        padding: 18px 18px 14px;
        min-height: 176px;
        background: #ffffff;
    }
    .app-code {
        color: #0f766e;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0;
        margin-bottom: 0.35rem;
    }
    .app-title {
        color: #111827;
        font-size: 1.08rem;
        font-weight: 700;
        line-height: 1.35;
        margin-bottom: 0.55rem;
    }
    .app-desc {
        color: #4b5563;
        font-size: 0.94rem;
        line-height: 1.55;
        min-height: 4.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("周四作业 A1-A8 汇总展示")
st.caption("一个公网 URL 访问八个 Streamlit 交互 app。左侧侧边栏也可以直接切换页面。")

cols = st.columns(2)
for index, (code, title, desc, page) in enumerate(APPS):
    with cols[index % 2]:
        st.markdown(
            f"""
            <div class="app-card">
                <div class="app-code">{code}</div>
                <div class="app-title">{title}</div>
                <div class="app-desc">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.page_link(page, label=f"打开 {code}", icon=":material/open_in_new:")
        st.write("")

st.divider()
st.subheader("部署入口")
st.write("上传整个 `周四作业` 目录到 GitHub 后，在 Streamlit Community Cloud 新建 app。")
st.code("Main file path: Home.py", language="text")
st.write("部署完成后得到的 Streamlit URL 就是可直接打开的公网链接，不需要本地运行。")
