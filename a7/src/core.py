from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def base_image(seed: int = 3, size: int = 128) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (size, size), (236, 240, 245))
    d = ImageDraw.Draw(img)
    for i in range(5):
        x, y = int(rng.integers(8, 90)), int(rng.integers(8, 90))
        color = tuple(int(v) for v in rng.integers(50, 230, 3))
        d.rounded_rectangle((x, y, x + int(rng.integers(20, 42)), y + int(rng.integers(18, 42))), radius=6, fill=color)
    d.text((36, 52), "SSL", fill=(30, 40, 55))
    return np.asarray(img)


def rotate_task(image: np.ndarray, angle: int = 90) -> dict:
    pil = Image.fromarray(image).rotate(angle, expand=True, fillcolor=(236, 240, 245))
    epochs = np.arange(1, 31)
    difficulty = abs(((angle % 360) - 180) / 180)
    loss = np.exp(-epochs / (7 + difficulty * 3)) + 0.08
    acc = 1 - np.exp(-epochs / (8 + difficulty * 2)) * 0.75
    return {"rotated": np.asarray(pil.resize(image.shape[1::-1])), "loss": loss, "acc": acc}


def mae_mask(image: np.ndarray, ratio: float = 0.45, patch: int = 16, seed: int = 4) -> dict:
    rng = np.random.default_rng(seed)
    masked = image.copy()
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    patches = [(y, x) for y in range(0, h, patch) for x in range(0, w, patch)]
    rng.shuffle(patches)
    for y, x in patches[: int(len(patches) * ratio)]:
        masked[y:y+patch, x:x+patch] = 235
        mask[y:y+patch, x:x+patch] = True
    blur = np.asarray(Image.fromarray(masked).filter(ImageFilter.GaussianBlur(radius=5)))
    recon = masked.copy()
    recon[mask] = blur[mask]
    heat = np.mean(np.abs(recon.astype(float) - image.astype(float)), axis=2) / 255.0
    heat = np.clip(heat, 0.0, 1.0)
    epochs = np.arange(1, 36)
    loss = 0.9 * ratio * np.exp(-epochs / 11) + 0.06
    return {"masked": masked, "recon": recon, "heat": heat, "loss": loss}


def simclr_embeddings(seed: int = 8, n: int = 80, strong_aug: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    centers = np.array([[-1, -1], [1, -0.8], [0.2, 1.0]])
    labels = rng.integers(0, 3, n)
    scale = 0.55 if strong_aug else 0.35
    z = centers[labels] + rng.normal(0, scale, (n, 2))
    loss = np.exp(-np.arange(1, 41) / (13 if strong_aug else 9)) + (0.11 if strong_aug else 0.06)
    return {"z": z, "labels": labels, "loss": loss}
