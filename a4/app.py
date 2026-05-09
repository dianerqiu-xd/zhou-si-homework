from __future__ import annotations

import math
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "a4_mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def configure_matplotlib_fonts():
    for font_path in [
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
    ]:
        if Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)
            plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=font_path).get_name()]
            plt.rcParams["axes.unicode_minus"] = False
            return


configure_matplotlib_fonts()


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

CLASS_CN = {
    "airplane": "飞机",
    "automobile": "汽车",
    "bird": "鸟",
    "cat": "猫",
    "deer": "鹿",
    "dog": "狗",
    "frog": "青蛙",
    "horse": "马",
    "ship": "船",
    "truck": "卡车",
}

PALETTE = [
    "#2F6BFF",
    "#D84A38",
    "#E8A317",
    "#7B4B94",
    "#4E9A51",
    "#8C5E3C",
    "#19A974",
    "#B65F32",
    "#008E9B",
    "#5F6B7A",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def rng_for(seed: int, *keys: int) -> np.random.Generator:
    value = seed
    for key in keys:
        value = (value * 1103515245 + key * 12345 + 97) % (2**32)
    return np.random.default_rng(value)


def default_image_paths():
    app_dir = Path(__file__).parent
    candidates = [
        app_dir / "assets" / "default_images",
        app_dir.parent / "图片",
    ]
    paths = []
    for folder in candidates:
        if folder.exists():
            paths.extend(
                sorted(
                    [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
                    key=lambda p: p.name,
                )
            )
    seen = set()
    unique_paths = []
    for path in paths:
        if path.name not in seen:
            seen.add(path.name)
            unique_paths.append(path)
    return unique_paths


def image_to_query_array(source) -> np.ndarray:
    img = Image.open(source).convert("RGB")
    img = ImageOps.fit(img, (32, 32), method=Image.Resampling.BICUBIC)
    return np.asarray(img).astype(np.float32) / 255.0


def image_query_picker(key_prefix: str, label: str = "上传图片"):
    uploaded = st.file_uploader(label, type=["jpg", "jpeg", "png", "bmp", "webp"], key=f"{key_prefix}_upload")
    defaults = default_image_paths()
    if uploaded is not None:
        return image_to_query_array(uploaded), "上传图片：已缩放到 32x32 RGB", "uploaded"
    if defaults:
        selected = st.selectbox(
            "默认图片",
            defaults,
            format_func=lambda p: p.name,
            key=f"{key_prefix}_default_image",
        )
        return image_to_query_array(selected), f"默认图片：{selected.name}，已缩放到 32x32 RGB", "default"
    return None, "没有找到默认图片，请上传一张图片。", "missing"


def draw_cifar_like(label: int, seed: int = 0) -> np.ndarray:
    """Generate a tiny 32x32 RGB image with CIFAR-10-like class semantics."""
    r = rng_for(seed, label)
    bg = np.zeros((32, 32, 3), dtype=np.uint8)
    base_colors = [
        (129, 182, 234),
        (72, 75, 82),
        (129, 191, 223),
        (183, 152, 125),
        (129, 171, 100),
        (181, 151, 118),
        (94, 176, 107),
        (181, 142, 93),
        (69, 142, 178),
        (96, 98, 101),
    ]
    bg[:] = np.array(base_colors[label], dtype=np.uint8)
    noise = r.normal(0, 12, bg.shape).astype(np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(bg, "RGB")
    draw = ImageDraw.Draw(img)
    dx = int(r.integers(-3, 4))
    dy = int(r.integers(-3, 4))

    if label == 0:  # airplane
        draw.rectangle((4 + dx, 20 + dy, 28 + dx, 23 + dy), fill=(228, 234, 238))
        draw.polygon([(14 + dx, 11 + dy), (20 + dx, 21 + dy), (9 + dx, 21 + dy)], fill=(245, 245, 245))
        draw.polygon([(23 + dx, 17 + dy), (29 + dx, 14 + dy), (27 + dx, 20 + dy)], fill=(210, 218, 226))
    elif label == 1:  # automobile
        draw.rectangle((2, 22, 31, 31), fill=(47, 47, 52))
        draw.rounded_rectangle((6 + dx, 13 + dy, 26 + dx, 23 + dy), radius=3, fill=(214, 60, 48))
        draw.rectangle((10 + dx, 10 + dy, 21 + dx, 16 + dy), fill=(238, 99, 77))
        draw.ellipse((8 + dx, 21 + dy, 14 + dx, 27 + dy), fill=(24, 24, 24))
        draw.ellipse((20 + dx, 21 + dy, 26 + dx, 27 + dy), fill=(24, 24, 24))
    elif label == 2:  # bird
        draw.ellipse((11 + dx, 9 + dy, 22 + dx, 20 + dy), fill=(236, 176, 44))
        draw.polygon([(15 + dx, 17 + dy), (5 + dx, 23 + dy), (17 + dx, 22 + dy)], fill=(199, 133, 38))
        draw.polygon([(20 + dx, 14 + dy), (29 + dx, 12 + dy), (22 + dx, 17 + dy)], fill=(250, 204, 87))
        draw.ellipse((19 + dx, 11 + dy, 21 + dx, 13 + dy), fill=(20, 20, 20))
    elif label == 3:  # cat
        draw.ellipse((8 + dx, 12 + dy, 25 + dx, 27 + dy), fill=(143, 91, 74))
        draw.polygon([(10 + dx, 13 + dy), (13 + dx, 6 + dy), (16 + dx, 14 + dy)], fill=(123, 74, 61))
        draw.polygon([(20 + dx, 14 + dy), (23 + dx, 6 + dy), (25 + dx, 15 + dy)], fill=(123, 74, 61))
        draw.arc((3 + dx, 14 + dy, 12 + dx, 30 + dy), 90, 260, fill=(95, 60, 51), width=2)
    elif label == 4:  # deer
        draw.rectangle((11 + dx, 16 + dy, 24 + dx, 24 + dy), fill=(130, 86, 48))
        draw.ellipse((20 + dx, 10 + dy, 29 + dx, 18 + dy), fill=(151, 100, 55))
        for x in (13, 21):
            draw.line((x + dx, 23 + dy, x - 1 + dx, 30 + dy), fill=(70, 49, 34), width=2)
        draw.line((25 + dx, 11 + dy, 25 + dx, 4 + dy), fill=(73, 51, 34), width=2)
        draw.line((25 + dx, 7 + dy, 21 + dx, 4 + dy), fill=(73, 51, 34), width=1)
        draw.line((25 + dx, 7 + dy, 29 + dx, 4 + dy), fill=(73, 51, 34), width=1)
    elif label == 5:  # dog
        draw.ellipse((7 + dx, 13 + dy, 24 + dx, 27 + dy), fill=(150, 102, 65))
        draw.ellipse((19 + dx, 9 + dy, 29 + dx, 19 + dy), fill=(131, 84, 54))
        draw.ellipse((18 + dx, 10 + dy, 22 + dx, 20 + dy), fill=(81, 54, 39))
        draw.line((8 + dx, 17 + dy, 3 + dx, 12 + dy), fill=(121, 77, 49), width=2)
    elif label == 6:  # frog
        draw.ellipse((7 + dx, 11 + dy, 25 + dx, 27 + dy), fill=(44, 168, 84))
        draw.ellipse((8 + dx, 8 + dy, 14 + dx, 14 + dy), fill=(84, 203, 106))
        draw.ellipse((19 + dx, 8 + dy, 25 + dx, 14 + dy), fill=(84, 203, 106))
        draw.ellipse((10 + dx, 10 + dy, 12 + dx, 12 + dy), fill=(18, 32, 22))
        draw.ellipse((21 + dx, 10 + dy, 23 + dx, 12 + dy), fill=(18, 32, 22))
    elif label == 7:  # horse
        draw.rectangle((8 + dx, 15 + dy, 24 + dx, 24 + dy), fill=(121, 70, 39))
        draw.polygon([(21 + dx, 13 + dy), (28 + dx, 9 + dy), (27 + dx, 19 + dy), (22 + dx, 18 + dy)], fill=(101, 56, 34))
        for x in (10, 22):
            draw.line((x + dx, 23 + dy, x - 1 + dx, 30 + dy), fill=(63, 43, 30), width=2)
        draw.line((8 + dx, 16 + dy, 3 + dx, 11 + dy), fill=(69, 44, 27), width=2)
    elif label == 8:  # ship
        draw.rectangle((0, 21, 32, 32), fill=(35, 94, 145))
        draw.polygon([(6 + dx, 20 + dy), (27 + dx, 20 + dy), (23 + dx, 27 + dy), (10 + dx, 27 + dy)], fill=(102, 61, 45))
        draw.line((16 + dx, 8 + dy, 16 + dx, 21 + dy), fill=(230, 230, 220), width=2)
        draw.polygon([(17 + dx, 9 + dy), (26 + dx, 18 + dy), (17 + dx, 18 + dy)], fill=(239, 239, 226))
    else:  # truck
        draw.rectangle((3, 22, 31, 31), fill=(45, 45, 47))
        draw.rectangle((5 + dx, 12 + dy, 20 + dx, 22 + dy), fill=(91, 107, 120))
        draw.rectangle((20 + dx, 15 + dy, 29 + dx, 22 + dy), fill=(114, 125, 134))
        draw.ellipse((8 + dx, 21 + dy, 15 + dx, 28 + dy), fill=(24, 24, 24))
        draw.ellipse((22 + dx, 21 + dy, 29 + dx, 28 + dy), fill=(24, 24, 24))

    if r.random() < 0.55:
        img = ImageEnhance.Color(img).enhance(float(r.uniform(0.85, 1.25)))
    img = ImageEnhance.Brightness(img).enhance(float(r.uniform(0.9, 1.12)))
    return np.asarray(img).astype(np.float32) / 255.0


def generate_synthetic_cifar(samples_per_class: int = 60, seed: int = 4):
    xs, ys = [], []
    for label in range(10):
        for i in range(samples_per_class):
            xs.append(draw_cifar_like(label, seed + i * 37))
            ys.append(label)
    x = np.stack(xs)
    y = np.array(ys, dtype=np.int64)
    order = rng_for(seed, 9).permutation(len(y))
    return x[order], y[order]


def load_cifar_batches(data_dir: Path):
    batch_dir = data_dir / "cifar-10-batches-py"
    if not batch_dir.exists():
        return None
    xs, ys = [], []
    for name in ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]:
        path = batch_dir / name
        if not path.exists():
            continue
        with path.open("rb") as f:
            batch = pickle.load(f, encoding="latin1")
        data = batch["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = np.array(batch["labels"], dtype=np.int64)
        xs.append(data.astype(np.float32) / 255.0)
        ys.append(labels)
    if not xs:
        return None
    return np.concatenate(xs), np.concatenate(ys)


@st.cache_data(show_spinner=False)
def get_image_data(samples_per_class: int, seed: int):
    local = load_cifar_batches(Path(__file__).parent / "data")
    source = "本地 CIFAR-10 batch"
    if local is None:
        x, y = generate_synthetic_cifar(samples_per_class=samples_per_class, seed=seed)
        source = "离线 CIFAR-10 风格数据"
    else:
        x, y = local
        keep = []
        r = rng_for(seed, samples_per_class)
        for label in range(10):
            idx = np.where(y == label)[0]
            take = r.choice(idx, size=min(samples_per_class, len(idx)), replace=False)
            keep.extend(take.tolist())
        keep = np.array(keep)
        x, y = x[keep], y[keep]
    split = int(len(y) * 0.78)
    return x[:split], y[:split], x[split:], y[split:], source


def least_squares_fit(n: int, slope: float, intercept: float, noise: float, seed: int):
    r = rng_for(seed, n)
    x = np.linspace(-4, 4, n)
    y = slope * x + intercept + r.normal(0, noise, n)
    design = np.c_[np.ones_like(x), x]
    theta = np.linalg.pinv(design.T @ design) @ design.T @ y
    pred = design @ theta
    mse = float(np.mean((pred - y) ** 2))
    return x, y, pred, theta, mse


def image_brightness_regression(query_img: np.ndarray):
    brightness = query_img.mean(axis=2)
    y = brightness.mean(axis=0)
    x = np.arange(len(y), dtype=float)
    design = np.c_[np.ones_like(x), x]
    theta = np.linalg.pinv(design.T @ design) @ design.T @ y
    pred = design @ theta
    mse = float(np.mean((pred - y) ** 2))
    return x, y, pred, theta, mse


def plot_linear_regression(x, y, pred, theta, mse):
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    ax.scatter(x, y, s=30, color="#2F6BFF", alpha=0.78, label="samples")
    ax.plot(x, pred, color="#D84A38", linewidth=2.8, label="least squares fit")
    ax.set_title("Least Squares Linear Regression")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper left")
    ax.text(
        0.02,
        0.04,
        f"y = {theta[1]:.3f}x + {theta[0]:.3f}\nMSE = {mse:.4f}",
        transform=ax.transAxes,
        bbox=dict(facecolor="white", edgecolor="#CBD5E1", alpha=0.92),
    )
    fig.tight_layout()
    return fig


def plot_image_brightness_regression(query_img, x, y, pred, theta, mse):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), dpi=150, gridspec_kw={"width_ratios": [0.85, 1.35]})
    axes[0].imshow(query_img)
    axes[0].set_title("Query image")
    axes[0].axis("off")
    axes[1].scatter(x, y, s=24, color="#2F6BFF", alpha=0.78, label="column brightness")
    axes[1].plot(x, pred, color="#D84A38", linewidth=2.4, label="least squares fit")
    axes[1].set_title("Image Brightness Linear Regression")
    axes[1].set_xlabel("column")
    axes[1].set_ylabel("mean brightness")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.22)
    axes[1].legend(fontsize=8)
    axes[1].text(
        0.03,
        0.05,
        f"y = {theta[1]:.4f}x + {theta[0]:.4f}\nMSE = {mse:.5f}",
        transform=axes[1].transAxes,
        bbox=dict(facecolor="white", edgecolor="#CBD5E1", alpha=0.92),
        fontsize=8,
    )
    fig.tight_layout()
    return fig


def flatten_images(x):
    return x.reshape(len(x), -1)


def knn_predict(train_x, train_y, query_x, k: int):
    train_f = flatten_images(train_x)
    q = query_x.reshape(1, -1)
    distances = np.sum((train_f - q) ** 2, axis=1)
    order = np.argsort(distances)
    nn = order[:k]
    votes = np.bincount(train_y[nn], minlength=10)
    pred = int(np.argmax(votes))
    return pred, nn, distances[nn], votes


def pca_2d(x):
    f = flatten_images(x)
    f = f - f.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(f, full_matrices=False)
    return f @ vt[:2].T


def plot_knn_embedding(train_x, train_y, test_img, neighbor_idx, k: int):
    subset = min(420, len(train_y))
    features = pca_2d(np.concatenate([train_x[:subset], test_img[None]], axis=0))
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=150)
    for label in range(10):
        idx = np.where(train_y[:subset] == label)[0]
        ax.scatter(
            features[idx, 0],
            features[idx, 1],
            s=18,
            color=PALETTE[label],
            alpha=0.55,
            label=CLASS_NAMES[label],
        )
    visible_nn = [i for i in neighbor_idx[:k] if i < subset]
    if visible_nn:
        ax.scatter(
            features[visible_nn, 0],
            features[visible_nn, 1],
            s=130,
            facecolors="none",
            edgecolors="#111827",
            linewidths=1.8,
            label=f"K={k} neighbors",
        )
    ax.scatter(
        features[-1, 0],
        features[-1, 1],
        marker="*",
        s=260,
        color="#D84A38",
        edgecolor="white",
        linewidth=1,
        label="query",
    )
    ax.set_title("KNN neighbors on a 2D PCA view")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.18)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=7)
    fig.tight_layout()
    return fig


def standardize_train(x):
    f = flatten_images(x)
    mean = f.mean(axis=0, keepdims=True)
    std = f.std(axis=0, keepdims=True) + 1e-5
    return (f - mean) / std, mean, std


def softmax_loss_and_grad(w, b, x, y, reg=1e-3):
    n = len(y)
    scores = x @ w + b
    scores -= scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    loss = -np.log(probs[np.arange(n), y] + 1e-12).mean() + 0.5 * reg * np.sum(w * w)
    probs[np.arange(n), y] -= 1
    probs /= n
    dw = x.T @ probs + reg * w
    db = probs.sum(axis=0)
    return float(loss), dw, db


def softmax_from_scores(scores):
    shifted = scores - np.max(scores)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores)


@st.cache_data(show_spinner=False)
def train_linear_classifier(samples_per_class: int, seed: int, steps: int, lr: float, reg: float):
    train_x, train_y, test_x, test_y, source = get_image_data(samples_per_class, seed)
    x_train, mean, std = standardize_train(train_x)
    x_test = (flatten_images(test_x) - mean) / std
    r = rng_for(seed, steps)
    w = r.normal(0, 0.005, (x_train.shape[1], 10))
    b = np.zeros(10)
    losses = []
    batch = min(128, len(train_y))
    for step in range(steps):
        idx = r.choice(len(train_y), size=batch, replace=False)
        loss, dw, db = softmax_loss_and_grad(w, b, x_train[idx], train_y[idx], reg=reg)
        w -= lr * dw
        b -= lr * db
        if step % max(1, steps // 40) == 0 or step == steps - 1:
            losses.append(loss)
    pred = np.argmax(x_test @ w + b, axis=1)
    acc = float(np.mean(pred == test_y))
    return w, b, mean, std, losses, acc, source


def classify_query_image(w, b, mean, std, query_img):
    x = (query_img.reshape(1, -1) - mean) / std
    scores = (x @ w + b).ravel()
    probs = softmax_from_scores(scores)
    pred = int(np.argmax(probs))
    return scores, probs, pred


def weight_templates(w):
    images = []
    for c in range(10):
        raw = w[:, c].reshape(32, 32, 3)
        lo, hi = np.percentile(raw, [2, 98])
        img = np.clip((raw - lo) / (hi - lo + 1e-8), 0, 1)
        images.append(img)
    return images


def plot_weight_templates(w):
    templates = weight_templates(w)
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.2), dpi=150)
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(templates[i])
        ax.set_title(f"{CLASS_NAMES[i]}\n{CLASS_CN[CLASS_NAMES[i]]}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Linear Classifier Learned Templates", fontsize=14)
    fig.tight_layout()
    return fig


def optimizer_paths(lr: float, momentum: float, steps: int, start=(-4.5, 4.2)):
    def grad(point):
        x, y = point
        return np.array([4.0 * x, 0.7 * y])

    vanilla = [np.array(start, dtype=float)]
    heavy = [np.array(start, dtype=float)]
    velocity = np.zeros(2)
    for _ in range(steps):
        vanilla.append(vanilla[-1] - lr * grad(vanilla[-1]))
        velocity = momentum * velocity - lr * grad(heavy[-1])
        heavy.append(heavy[-1] + velocity)
    return np.array(vanilla), np.array(heavy)


def plot_optimizer_compare(lr: float, momentum: float, steps: int):
    vanilla, heavy = optimizer_paths(lr, momentum, steps)
    xx, yy = np.meshgrid(np.linspace(-5, 5, 180), np.linspace(-5, 5, 180))
    zz = 2.0 * xx**2 + 0.35 * yy**2
    fig, ax = plt.subplots(figsize=(7.1, 4.8), dpi=150)
    ax.contour(xx, yy, zz, levels=22, cmap="Greys", alpha=0.55)
    ax.plot(vanilla[:, 0], vanilla[:, 1], "o-", color="#2F6BFF", markersize=3, label="SGD / GD")
    ax.plot(heavy[:, 0], heavy[:, 1], "o-", color="#D84A38", markersize=3, label="Momentum")
    ax.scatter([0], [0], color="#111827", marker="x", s=80, label="minimum")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_title("Gradient Descent: SGD vs Momentum")
    ax.set_xlabel("w1")
    ax.set_ylabel("w2")
    ax.grid(True, alpha=0.16)
    ax.legend()
    fig.tight_layout()
    return fig


def loss_demo(scores, correct_label, delta=1.0):
    scores = np.asarray(scores, dtype=float)
    correct_score = scores[correct_label]
    margins = np.maximum(0.0, scores - correct_score + delta)
    margins[correct_label] = 0.0
    svm = float(np.sum(margins))
    shifted = scores - scores.max()
    probs = np.exp(shifted) / np.exp(shifted).sum()
    ce = float(-np.log(probs[correct_label] + 1e-12))
    return margins, svm, probs, ce


def fig_to_image(fig):
    fig.canvas.draw()
    data = np.asarray(fig.canvas.buffer_rgba())
    return Image.fromarray(data[..., :3])


def show_header():
    st.set_page_config(page_title="Vibe Coding: ML Visual Lab", layout="wide")
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.6rem; }
        div[data-testid="stMetric"] {
            background: rgba(148, 163, 184, 0.12);
            border: 1px solid rgba(148, 163, 184, 0.35);
            padding: 12px;
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Vibe Coding: Machine Learning Visual Lab")
    st.caption("Least Squares Linear Regression, KNN, Linear Classifier, Gradient Descent and Loss Functions")


def page_least_squares():
    st.subheader("1. Least Squares Linear Regression 示例")
    c1, c2 = st.columns([0.32, 0.68])
    with c1:
        n = st.slider("样本数量", 20, 180, 70, 5)
        slope = st.slider("真实斜率", -4.0, 4.0, 1.8, 0.1)
        intercept = st.slider("真实截距", -5.0, 5.0, -0.7, 0.1)
        noise = st.slider("噪声强度", 0.0, 5.0, 1.2, 0.1)
        seed = st.slider("随机种子", 0, 99, 12)
    x, y, pred, theta, mse = least_squares_fit(n, slope, intercept, noise, seed)
    with c2:
        st.pyplot(plot_linear_regression(x, y, pred, theta, mse), clear_figure=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("估计斜率", f"{theta[1]:.4f}")
    m2.metric("估计截距", f"{theta[0]:.4f}")
    m3.metric("MSE", f"{mse:.4f}")
    st.code("theta = (X.T @ X)^(-1) @ X.T @ y", language="python")

    st.divider()
    st.write("上传/默认图片的亮度线性回归")
    img_left, img_right = st.columns([0.28, 0.72])
    with img_left:
        query_img, caption, state = image_query_picker("ls", "上传图片用于亮度回归")
        if query_img is not None:
            st.image(query_img, caption=caption, width=150)
        else:
            st.warning(caption)
    if query_img is not None:
        ix, iy, ipred, itheta, imse = image_brightness_regression(query_img)
        with img_right:
            st.pyplot(plot_image_brightness_regression(query_img, ix, iy, ipred, itheta, imse), clear_figure=True)
        im1, im2, im3 = st.columns(3)
        im1.metric("图片亮度斜率", f"{itheta[1]:.5f}")
        im2.metric("图片亮度截距", f"{itheta[0]:.5f}")
        im3.metric("图片亮度 MSE", f"{imse:.5f}")


def page_knn():
    st.subheader("2. KNN 图像分类与不同 K 对比")
    train_x, train_y, test_x, test_y, source = get_image_data(60, 7)
    left, right = st.columns([0.28, 0.72])
    with left:
        st.info(f"数据源：{source}")
        query_img, caption, state = image_query_picker("knn", "上传一张图片作为 KNN query")
        query_label = None
        if query_img is None:
            query_id = st.slider("测试图像编号", 0, len(test_y) - 1, 8)
            query_img = test_x[query_id]
            query_label = int(test_y[query_id])
            caption = f"真实类别：{CLASS_NAMES[query_label]} / {CLASS_CN[CLASS_NAMES[query_label]]}"
        k = st.select_slider("K 值", options=[1, 3, 5, 7, 9, 15], value=5)
        if state == "default":
            st.caption("默认图片来自 a4/assets/default_images；本地开发时也会回退读取 ../图片。")
        if state == "missing":
            st.caption("没有默认图片或上传图片时，才使用数据集测试样本。")
        st.image(query_img, caption=caption, width=150)
    pred, nn, distances, votes = knn_predict(train_x, train_y, query_img, k)
    with right:
        st.pyplot(plot_knn_embedding(train_x, train_y, query_img, nn, k), clear_figure=True)
    st.success(f"K={k} 预测：{CLASS_NAMES[pred]} / {CLASS_CN[CLASS_NAMES[pred]]}")
    if query_label is not None:
        st.caption(f"数据集标签：{CLASS_NAMES[query_label]} / {CLASS_CN[CLASS_NAMES[query_label]]}")
    st.write("最近邻图像")
    cols = st.columns(min(k, 9))
    for col, idx, dist in zip(cols, nn[: min(k, 9)], distances[: min(k, 9)]):
        with col:
            st.image(train_x[idx], width="stretch")
            st.caption(f"{CLASS_NAMES[train_y[idx]]}\nd={dist:.2f}")
    st.write("不同 K 的投票结果")
    rows = []
    for kk in [1, 3, 5, 7, 9, 15]:
        pp, _, _, vv = knn_predict(train_x, train_y, query_img, kk)
        rows.append({"K": kk, "预测类别": CLASS_NAMES[pp], "中文": CLASS_CN[CLASS_NAMES[pp]], "最高票数": int(vv.max())})
    st.table(rows)


def page_templates():
    st.subheader("3. 基于 CIFAR 图像的线性分类器模板可视化")
    c1, c2 = st.columns([0.3, 0.7])
    with c1:
        samples = st.slider("每类训练样本", 30, 120, 60, 10)
        steps = st.slider("SGD 训练步数", 80, 600, 260, 20)
        lr = st.slider("学习率", 0.02, 0.8, 0.18, 0.02)
        reg = st.select_slider("L2 正则", options=[0.0, 1e-4, 1e-3, 1e-2], value=1e-3)
    with st.spinner("训练线性 softmax 分类器..."):
        w, b, mean, std, losses, acc, source = train_linear_classifier(samples, 13, steps, lr, reg)
    with c1:
        st.metric("测试准确率", f"{acc * 100:.1f}%")
        st.caption(f"数据源：{source}")
        st.line_chart(losses)
        st.write("上传/默认图片分类")
        query_img, caption, state = image_query_picker("template", "上传图片给线性分类器")
        if query_img is not None:
            scores, probs, pred = classify_query_image(w, b, mean, std, query_img)
            st.image(query_img, caption=caption, width=150)
            st.success(f"预测：{CLASS_NAMES[pred]} / {CLASS_CN[CLASS_NAMES[pred]]}")
            top = np.argsort(probs)[::-1][:5]
            st.table(
                [
                    {
                        "类别": CLASS_NAMES[i],
                        "中文": CLASS_CN[CLASS_NAMES[i]],
                        "score": f"{scores[i]:.3f}",
                        "prob": f"{probs[i]:.3f}",
                    }
                    for i in top
                ]
            )
        else:
            st.warning(caption)
    with c2:
        st.pyplot(plot_weight_templates(w), clear_figure=True)


def page_optimizer_loss():
    st.subheader("4. SGD/动量更新与不同 Loss 计算过程")
    a, b = st.columns([0.55, 0.45])
    with a:
        lr = st.slider("优化学习率", 0.01, 0.45, 0.12, 0.01)
        momentum = st.slider("动量系数", 0.0, 0.98, 0.82, 0.02)
        steps = st.slider("更新步数", 5, 80, 28, 1)
        st.pyplot(plot_optimizer_compare(lr, momentum, steps), clear_figure=True)
    with b:
        st.write("多分类 SVM loss 与 Softmax cross-entropy")
        score_source = st.radio("分数来源", ["上传/默认图片的线性分类器 scores", "手动输入 scores"], horizontal=False)
        if score_source == "上传/默认图片的线性分类器 scores":
            query_img, caption, state = image_query_picker("loss", "上传图片生成 loss scores")
            if query_img is not None:
                w, bb, mean, std, _, _, _ = train_linear_classifier(60, 13, 260, 0.18, 1e-3)
                scores, probs_preview, pred = classify_query_image(w, bb, mean, std, query_img)
                correct_label = st.selectbox(
                    "假设正确类别",
                    list(range(10)),
                    format_func=lambda x: f"{x} - {CLASS_NAMES[x]} / {CLASS_CN[CLASS_NAMES[x]]}",
                    index=pred,
                )
                st.image(query_img, caption=f"{caption}；线性分类器预测：{CLASS_NAMES[pred]}", width=150)
            else:
                st.warning(caption)
                scores = np.array([2.4, -0.7, 3.1, 0.8, -1.2])
                correct_label = 2
        else:
            correct_label = st.selectbox("正确类别", list(range(5)), format_func=lambda x: f"class {x}", index=2)
            manual_scores = []
            defaults = [2.4, -0.7, 3.1, 0.8, -1.2]
            for i in range(5):
                manual_scores.append(st.slider(f"score[{i}]", -5.0, 5.0, defaults[i], 0.1))
            scores = np.array(manual_scores)
        margins, svm, probs, ce = loss_demo(scores, correct_label)
        st.metric("SVM hinge loss", f"{svm:.4f}")
        st.metric("Softmax CE loss", f"{ce:.4f}")
        st.table(
            [
                {
                    "class": CLASS_NAMES[i] if len(scores) == 10 else i,
                    "score": f"{scores[i]:.2f}",
                    "margin": f"{margins[i]:.3f}",
                    "softmax_prob": f"{probs[i]:.3f}",
                }
                for i in range(5)
            ]
        )
        st.code(
            "SVM: sum(max(0, s_j - s_y + 1))\nSoftmax CE: -log(exp(s_y) / sum(exp(s_j)))",
            language="text",
        )


def main():
    show_header()
    page = st.sidebar.radio(
        "选择演示模块",
        ["Least Squares", "KNN Image Classifier", "Linear Classifier Templates", "Optimizer and Loss"],
    )
    if page == "Least Squares":
        page_least_squares()
    elif page == "KNN Image Classifier":
        page_knn()
    elif page == "Linear Classifier Templates":
        page_templates()
    else:
        page_optimizer_loss()


if __name__ == "__main__":
    main()
