from __future__ import annotations

import textwrap
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "a4_mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import app


ROOT = Path(__file__).parent
SCREENSHOT_DIR = ROOT / "screenshots"
OUTPUT_DIR = ROOT / "outputs"
PDF_PATH = OUTPUT_DIR / "Vibe_Coding_A4_Report.pdf"


def save_fig(fig, name: str):
    path = SCREENSHOT_DIR / name
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def make_screenshots():
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    x, y, pred, theta, mse = app.least_squares_fit(70, 1.8, -0.7, 1.2, 12)
    p1 = save_fig(app.plot_linear_regression(x, y, pred, theta, mse), "01_least_squares.png")

    train_x, train_y, test_x, test_y, _ = app.get_image_data(60, 7)
    defaults = app.default_image_paths()
    query_img = app.image_to_query_array(defaults[0]) if defaults else test_x[8]
    _, nn, _, _ = app.knn_predict(train_x, train_y, query_img, 5)
    p2 = save_fig(app.plot_knn_embedding(train_x, train_y, query_img, nn, 5), "02_knn_k5.png")

    w, _, _, _, losses, acc, source = app.train_linear_classifier(60, 13, 260, 0.18, 1e-3)
    p3 = save_fig(app.plot_weight_templates(w), "03_linear_templates.png")

    p4 = save_fig(app.plot_optimizer_compare(0.12, 0.82, 28), "04_optimizer_compare.png")

    p5 = SCREENSHOT_DIR / "05_loss_table.png"
    make_loss_table(p5)

    return {
        "least_squares": p1,
        "knn": p2,
        "templates": p3,
        "optimizer": p4,
        "loss": p5,
        "acc": acc,
        "source": source,
        "losses": losses,
    }


def font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/STHeiti Medium.ttc" if bold else "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def wrap_by_chars(text: str, width: int):
    lines = []
    for raw in text.splitlines():
        if not raw:
            lines.append("")
            continue
        lines.extend(textwrap.wrap(raw, width=width, break_long_words=False, replace_whitespace=False))
    return lines


def draw_text_box(draw, xy, text, width_chars, fill="#111827", size=28, bold=False, line_gap=8):
    x, y = xy
    f = font(size, bold=bold)
    for line in wrap_by_chars(text, width_chars):
        draw.text((x, y), line, fill=fill, font=f)
        y += size + line_gap
    return y


def paste_fit(page, image_path: Path, box):
    img = Image.open(image_path).convert("RGB")
    x, y, w, h = box
    img.thumbnail((w, h), Image.Resampling.LANCZOS)
    page.paste(img, (x + (w - img.width) // 2, y + (h - img.height) // 2))


def new_page():
    return Image.new("RGB", (1240, 1754), "white")


def add_footer(draw, number: int):
    draw.line((90, 1668, 1150, 1668), fill="#CBD5E1", width=2)
    draw.text((90, 1685), f"Vibe Coding A4 Report · Page {number}", fill="#64748B", font=font(22))


def make_loss_table(path: Path):
    scores = np.array([2.4, -0.7, 3.1, 0.8, -1.2])
    margins, svm, probs, ce = app.loss_demo(scores, correct_label=2)
    page = Image.new("RGB", (1240, 760), "#FFFFFF")
    d = ImageDraw.Draw(page)
    d.rectangle((0, 0, 1240, 760), fill="#F8FAFC")
    d.text((60, 48), "Loss 计算过程演示", fill="#111827", font=font(42, bold=True))
    d.text((60, 108), f"SVM hinge loss = {svm:.4f}    Softmax CE loss = {ce:.4f}", fill="#334155", font=font(28))
    headers = ["class", "score", "margin", "softmax prob"]
    xs = [80, 330, 570, 840]
    y = 190
    d.rectangle((60, y - 20, 1180, y + 54), fill="#E2E8F0")
    for x, h in zip(xs, headers):
        d.text((x, y), h, fill="#111827", font=font(26, bold=True))
    y += 82
    for i in range(5):
        fill = "#FFFFFF" if i != 2 else "#E0F2FE"
        d.rectangle((60, y - 18, 1180, y + 54), fill=fill)
        values = [str(i), f"{scores[i]:.2f}", f"{margins[i]:.3f}", f"{probs[i]:.3f}"]
        for x, v in zip(xs, values):
            d.text((x, y), v, fill="#111827", font=font(26))
        y += 78
    d.text((70, 632), "SVM: sum(max(0, s_j - s_y + 1))", fill="#334155", font=font(24))
    d.text((70, 674), "Softmax CE: -log(exp(s_y) / sum(exp(s_j)))", fill="#334155", font=font(24))
    page.save(path)


def build_pdf(assets):
    pages = []

    page = new_page()
    d = ImageDraw.Draw(page)
    d.text((90, 90), "Vibe Coding 实验报告", fill="#111827", font=font(54, bold=True))
    y = 176
    y = draw_text_box(
        d,
        (90, y),
        "题目：实现 Least Squares Linear Regression 示例；实现 KNN/线性分类器（图像数据）；展示 CIFAR 线性分类器模板、不同 K 的 KNN 对比、SGD/动量更新对比、不同 loss 的计算过程，并做成可交互桌面或 Web 应用。",
        38,
        size=30,
        fill="#334155",
    )
    y += 30
    y = draw_text_box(
        d,
        (90, y),
        "Prompt 文本：参考文件夹里面的 skill, 做这个新题目，保存到新建的 a4 文件夹里面。交付 PDF 实验报告、核心源码 py 文件、可选外链 URL，并列出所使用的 Agent 或 LLM。",
        40,
        size=28,
        fill="#334155",
    )
    y += 30
    y = draw_text_box(
        d,
        (90, y),
        "使用的 Agent/LLM：Codex（GPT-5 系列）。开发环境：Python 3.11、Streamlit、NumPy、Matplotlib、Pillow。",
        40,
        size=28,
        fill="#334155",
    )
    y += 42
    d.rectangle((90, y, 1150, y + 250), fill="#F8FAFC", outline="#CBD5E1", width=2)
    draw_text_box(
        d,
        (120, y + 32),
        "应用交互模块：\n1. 最小二乘：调节样本数、噪声、真实斜率/截距，也可上传/选择默认图片并对亮度剖面做闭式解回归。\n2. KNN：上传/选择默认图片和 K 值，查看最近邻、投票结果、PCA 可视化。\n3. 线性分类器：训练 softmax 线性分类器，把权重还原成模板图，并对上传/默认图片输出分类分数。\n4. 优化与 Loss：比较普通梯度下降和动量路径，用手动 scores 或上传/默认图片的分类器 scores 展示 SVM hinge loss 与 softmax cross-entropy。",
        42,
        size=25,
        fill="#0F172A",
    )
    add_footer(d, 1)
    pages.append(page)

    page = new_page()
    d = ImageDraw.Draw(page)
    d.text((90, 80), "截图 1：Least Squares Linear Regression", fill="#111827", font=font(42, bold=True))
    paste_fit(page, assets["least_squares"], (90, 150, 1060, 650))
    draw_text_box(
        d,
        (90, 850),
        "核心公式 theta = (X.T X)^(-1) X.T y。用户改变噪声和样本量后，散点与拟合直线会重新计算，指标区显示估计斜率、截距和 MSE。",
        44,
        size=28,
        fill="#334155",
    )
    add_footer(d, 2)
    pages.append(page)

    page = new_page()
    d = ImageDraw.Draw(page)
    d.text((90, 80), "截图 2：KNN 图像分类与不同 K 对比", fill="#111827", font=font(42, bold=True))
    paste_fit(page, assets["knn"], (90, 150, 1060, 650))
    draw_text_box(
        d,
        (90, 850),
        "KNN 使用 32x32x3 图像向量的欧氏距离。交互页支持上传图片，也默认读取 a4/assets/default_images 中的图片作为 query，并展示最近邻缩略图、不同 K 的投票结果和 PCA 二维邻域。",
        43,
        size=28,
        fill="#334155",
    )
    add_footer(d, 3)
    pages.append(page)

    page = new_page()
    d = ImageDraw.Draw(page)
    d.text((90, 80), "截图 3：线性分类器学到的模板图像", fill="#111827", font=font(42, bold=True))
    paste_fit(page, assets["templates"], (90, 150, 1060, 610))
    draw_text_box(
        d,
        (90, 800),
        f"训练数据源：{assets['source']}。Softmax 线性分类器把每个类别的一列权重 reshape 成 32x32 RGB 图像，亮色区域表示该类别更依赖的颜色/空间模式。本次脚本生成的测试准确率约为 {assets['acc'] * 100:.1f}%。",
        43,
        size=28,
        fill="#334155",
    )
    add_footer(d, 4)
    pages.append(page)

    page = new_page()
    d = ImageDraw.Draw(page)
    d.text((90, 80), "截图 4：SGD/动量与 Loss 计算", fill="#111827", font=font(42, bold=True))
    paste_fit(page, assets["optimizer"], (90, 145, 1060, 580))
    paste_fit(page, assets["loss"], (100, 775, 1040, 650))
    draw_text_box(
        d,
        (90, 1470),
        "小结：动量通过累积历史梯度让路径在狭长等高线中更稳定地前进；hinge loss 关注错误类别超过正确类别的 margin，softmax CE 则把分数转成概率后惩罚正确类概率过低。",
        43,
        size=26,
        fill="#334155",
    )
    add_footer(d, 5)
    pages.append(page)

    pages[0].save(PDF_PATH, save_all=True, append_images=pages[1:], resolution=150.0)
    return PDF_PATH


def main():
    assets = make_screenshots()
    pdf = build_pdf(assets)
    print(pdf)


if __name__ == "__main__":
    main()
