from __future__ import annotations

import os
from pathlib import Path
from textwrap import wrap

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties

from vibe_feature_core import (
    canny_visualization,
    feature_comparison,
    match_pipeline,
    stitch_many,
)


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
SCREENSHOTS = ROOT / "screenshots"
OUTPUTS = ROOT / "outputs"
FONT_PATH = "/System/Library/Fonts/STHeiti Medium.ttc"


def font(size: int = 11) -> FontProperties:
    if Path(FONT_PATH).exists():
        return FontProperties(fname=FONT_PATH, size=size)
    return FontProperties(size=size)


def read_image(name: str):
    image = cv2.imread(str(ASSETS / name))
    if image is None:
        raise FileNotFoundError(name)
    return image


def to_rgb(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_grid(path: Path, images: list, titles: list[str], cols: int = 2, figsize=(12, 8)):
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax, image, title in zip(axes, images, titles):
        ax.imshow(to_rgb(image))
        ax.set_title(title, fontproperties=font(12))
        ax.axis("off")
    for ax in axes[len(images):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_text_page(pdf: PdfPages, title: str, paragraphs: list[str], subtitle: str | None = None):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    y = 0.93
    fig.text(0.08, y, title, fontproperties=font(22), color="#101828")
    y -= 0.035
    if subtitle:
        fig.text(0.08, y, subtitle, fontproperties=font(11), color="#475467")
        y -= 0.055
    for paragraph in paragraphs:
        if paragraph.startswith("# "):
            y -= 0.012
            fig.text(0.08, y, paragraph[2:], fontproperties=font(14), color="#175CD3")
            y -= 0.032
            continue
        for line in wrap(paragraph, width=48):
            fig.text(0.08, y, line, fontproperties=font(10.8), color="#1D2939")
            y -= 0.024
        y -= 0.02
        if y < 0.08:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            fig = plt.figure(figsize=(8.27, 11.69))
            y = 0.93
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def write_image_page(pdf: PdfPages, title: str, image_path: Path, caption: str):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    fig.text(0.05, 0.94, title, fontproperties=font(18), color="#101828")
    ax = fig.add_axes([0.04, 0.12, 0.92, 0.76])
    ax.imshow(to_rgb(image))
    ax.axis("off")
    fig.text(0.05, 0.055, caption, fontproperties=font(10), color="#475467")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def generate_screenshots() -> dict[str, Path]:
    SCREENSHOTS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    feature_img = read_image("feature_sample.jpg")
    scene_1 = read_image("match_1.jpg")
    scene_2 = read_image("match_2.jpg")
    pano_images = [read_image("pano_left.jpg"), read_image("pano_mid.jpg"), read_image("pano_right.jpg")]

    canny = canny_visualization(feature_img, 50, 150, 1.2)
    edge_path = SCREENSHOTS / "01_canny_nms_comparison.png"
    save_grid(
        edge_path,
        [canny["original"], canny["before_nms"], canny["after_nms"], canny["canny_edges"], canny["overlay"]],
        ["原图", "NMS 前：梯度幅值", "NMS 后：细化响应", "Canny 最终边缘", "边缘叠加"],
        cols=3,
        figsize=(13, 8),
    )

    features = feature_comparison(feature_img, 0.01, 600)
    feature_path = SCREENSHOTS / "02_harris_sift_features.png"
    save_grid(
        feature_path,
        [feature_img, features["harris"]["view"], features["sift"]["view"], features["harris"]["heatmap"]],
        [
            "原图",
            f"Harris 角点：{len(features['harris']['points'])} 个",
            f"{features['sift']['method']} 特征点：{len(features['sift']['keypoints'])} 个",
            "Harris 响应热力图",
        ],
        cols=2,
        figsize=(12, 9),
    )

    pipeline = match_pipeline(scene_1, scene_2, "SIFT", 0.75, 4.0)
    matching_path = SCREENSHOTS / "03_matching_pipeline.png"
    save_grid(
        matching_path,
        [
            pipeline.feature_view_1,
            pipeline.feature_view_2,
            pipeline.initial_view,
            pipeline.ransac_view,
            pipeline.aligned_view,
        ],
        [
            "图像 1 特征点检测与描述",
            "图像 2 特征点检测与描述",
            f"初始匹配：{len(pipeline.initial_matches)} 对",
            f"RANSAC 内点：{len(pipeline.inlier_matches)} 对",
            "透视变换后的半透明对齐",
        ],
        cols=2,
        figsize=(13, 11),
    )

    blend_images = []
    blend_titles = []
    for method in ["none", "average", "feather", "multiband"]:
        panorama, steps = stitch_many(pano_images, blending=method)
        if panorama is not None:
            blend_images.append(panorama)
            last_inliers = steps[-1]["inliers"] if steps else 0
            blend_titles.append(f"{method} blending，末步内点 {last_inliers}")
            cv2.imwrite(str(OUTPUTS / f"panorama_{method}.png"), panorama)
    blending_path = SCREENSHOTS / "04_panorama_blending_comparison.png"
    save_grid(blending_path, blend_images, blend_titles, cols=2, figsize=(13, 9))

    panorama, steps = stitch_many(pano_images, blending="feather")
    step_images = [step["matches_view"] for step in steps] + [step["panorama"] for step in steps if step["panorama"] is not None]
    step_titles = [f"步骤 {step['step']} RANSAC 匹配" for step in steps] + [f"步骤 {step['step']} 中间全景" for step in steps if step["panorama"] is not None]
    process_path = SCREENSHOTS / "05_panorama_process.png"
    save_grid(process_path, step_images, step_titles, cols=2, figsize=(13, 10))
    if panorama is not None:
        cv2.imwrite(str(OUTPUTS / "panorama_final_feather.png"), panorama)

    return {
        "edge": edge_path,
        "feature": feature_path,
        "matching": matching_path,
        "blending": blending_path,
        "process": process_path,
    }


def generate_pdf(screenshots: dict[str, Path]) -> Path:
    pdf_path = OUTPUTS / "Vibe Coding 图像特征检测与匹配实验报告.pdf"
    prompt_text = (
        "参考文件夹里面的 skill，完成 Vibe Coding 图像特征检测与匹配：实现 Canny 边缘检测及非最大值抑制前后可视化；"
        "实现 Harris、SIFT 特征点检测并在图像中用以特征点为中心的圆形区域可视化；实现两幅图像的特征点检测、描述、初始匹配、"
        "RANSAC、变换和匹配对齐流程；实现多幅图像全景拼接并对比不同 blending；最终做成有交互操作的桌面或 Web 应用，"
        "提交 PDF 实验报告、核心源码 py 文件，并列出所使用的 Agent 或 LLM。"
    )
    with PdfPages(pdf_path) as pdf:
        write_text_page(
            pdf,
            "Vibe Coding 图像特征检测与匹配实验报告",
            [
                "# 一、Prompt 文本",
                prompt_text,
                "# 二、实现概述",
                "本实验采用 Python + OpenCV + Streamlit 实现交互应用。核心算法集中在 vibe_feature_core.py，界面入口为 app.py，报告与截图由 make_report.py 自动生成。",
                "Canny 部分展示灰度、梯度幅值、非最大值抑制后的细化结果和最终双阈值边缘。特征点部分对比 Harris 与 SIFT，Harris 以中心点和圆形邻域展示角点，SIFT 使用尺度圆和主方向展示关键点。",
                "匹配部分按检测、描述、Lowe ratio 初始匹配、RANSAC 内点筛选、单应性变换和半透明对齐顺序可视化。拼接部分支持多幅图像，提供 none、average、feather、multiband 四种 blending 对比。",
                "# 三、使用的 Agent/LLM",
                "Codex GPT-5。",
            ],
            subtitle="A3 提交文件夹：/Users/qiudianer/Desktop/周四作业/a3",
        )
        write_image_page(
            pdf,
            "截图 1：Canny 边缘检测与 NMS 前后对比",
            screenshots["edge"],
            "NMS 前的梯度响应较厚，NMS 后仅保留梯度方向上的局部最大值，最终 Canny 边缘更加细且连续。",
        )
        write_image_page(
            pdf,
            "截图 2：Harris 与 SIFT 特征点检测",
            screenshots["feature"],
            "Harris 主要定位角点区域，SIFT 同时编码尺度和方向，因此圆形区域的大小与方向线更能表达局部结构差异。",
        )
        write_image_page(
            pdf,
            "截图 3：两幅图像的匹配流程",
            screenshots["matching"],
            "从特征点检测和描述开始，经初始匹配、RANSAC 内点筛选，最终估计单应性矩阵并完成图像对齐。",
        )
        write_image_page(
            pdf,
            "截图 4：多幅图像全景拼接与 blending 对比",
            screenshots["blending"],
            "none 会出现明显覆盖边界，average 能减弱接缝，feather 和 multiband 在重叠区域过渡更自然。",
        )
        write_image_page(
            pdf,
            "截图 5：全景拼接过程可视化",
            screenshots["process"],
            "多图拼接按顺序逐步完成，每一步都输出 RANSAC 匹配和中间全景图，便于检查失败点和重叠区域质量。",
        )
        write_text_page(
            pdf,
            "小结",
            [
                "本次实验把图像特征检测、匹配和拼接整理成一个可交互应用。通过 Canny 的 NMS 前后对比，可以直观看到边缘由粗响应变为细边缘；通过 Harris 和 SIFT 对比，可以看到角点检测与尺度不变特征的表达差异。",
                "在两图匹配中，Lowe ratio 能先剔除大量描述子歧义匹配，RANSAC 进一步去掉几何关系不一致的外点。全景拼接的质量主要依赖相邻图像的重叠区域、特征点数量和内点比例；blending 方法会显著影响接缝自然度。",
                "交互应用提供上传图片、参数调节和内置示例，方便在不同图像上重复实验。核心源码文件为 vibe_feature_core.py，Web 应用入口为 app.py。",
            ],
        )
    return pdf_path


def main():
    screenshots = generate_screenshots()
    pdf_path = generate_pdf(screenshots)
    print(f"PDF saved to: {pdf_path}")
    for key, value in screenshots.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
