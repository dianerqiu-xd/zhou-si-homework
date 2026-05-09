from __future__ import annotations

import hashlib
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def digit_like(label: int = 3, size: int = 64) -> np.ndarray:
    img = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(img)
    if label % 5 == 0:
        d.ellipse((14, 10, 50, 54), outline=240, width=7)
    elif label % 5 == 1:
        d.line((32, 10, 32, 54), fill=240, width=8)
    elif label % 5 == 2:
        d.arc((12, 8, 52, 38), 200, 20, fill=240, width=7); d.line((48, 32, 16, 55), fill=240, width=7); d.line((16, 55, 52, 55), fill=240, width=6)
    elif label % 5 == 3:
        d.arc((12, 8, 50, 34), 210, 35, fill=240, width=7); d.arc((12, 30, 50, 58), 320, 150, fill=240, width=7)
    else:
        d.line((18, 10, 18, 36), fill=240, width=7); d.line((18, 36, 50, 36), fill=240, width=7); d.line((48, 10, 48, 56), fill=240, width=7)
    return np.asarray(img.filter(ImageFilter.GaussianBlur(0.6)))


def ae_vae_compare(label: int = 3, beta: float = 0.8) -> dict:
    x = digit_like(label)
    ae = np.asarray(Image.fromarray(x).filter(ImageFilter.GaussianBlur(1.0)))
    vae = np.asarray(Image.fromarray(x).filter(ImageFilter.GaussianBlur(1.8)))
    noise = np.random.default_rng(label).normal(0, 8 * beta, x.shape)
    vae = np.clip(vae.astype(float) + noise, 0, 255).astype(np.uint8)
    err = np.abs(ae.astype(float) - x.astype(float))
    err = (err / (err.max() + 1e-8) * 255).astype(np.uint8)
    epochs = np.arange(1, 35)
    ae_loss = 0.75 * np.exp(-epochs / 8) + 0.045
    vae_loss = 0.82 * np.exp(-epochs / 10) + 0.08 + beta * 0.01
    return {"input": x, "ae": ae, "vae": vae, "err": err, "ae_loss": ae_loss, "vae_loss": vae_loss}


def latent_points(n: int = 120, seed: int = 2) -> dict:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 10, n)
    angles = labels / 10 * 2 * np.pi
    z = np.c_[np.cos(angles), np.sin(angles)] + rng.normal(0, 0.18, (n, 2))
    return {"z": z, "labels": labels}


def generate_from_latent(zx: float, zy: float, size: int = 64) -> np.ndarray:
    label = int((math.atan2(zy, zx) + math.pi) / (2 * math.pi) * 10) % 10
    img = digit_like(label, size)
    shift = int(np.clip(zx * 5, -6, 6))
    img = np.roll(img, shift, axis=1)
    return img


def interpolate(z1: tuple[float, float], z2: tuple[float, float], steps: int = 7) -> list[np.ndarray]:
    return [generate_from_latent(z1[0] * (1 - t) + z2[0] * t, z1[1] * (1 - t) + z2[1] * t) for t in np.linspace(0, 1, steps)]


def gan_grid(seed: int = 4, count: int = 12) -> dict:
    rng = np.random.default_rng(seed)
    images, scores = [], []
    for i in range(count):
        z = rng.normal(size=2)
        img = generate_from_latent(float(z[0]), float(z[1]))
        images.append(img)
        scores.append(float(1 / (1 + np.exp(-np.linalg.norm(z)))))
    return {"images": images, "scores": scores}


def prompt_image(prompt: str, negative: str = "", seed: int = 1, steps: int = 20, guidance: float = 6.0) -> np.ndarray:
    key = int(hashlib.sha256((prompt + negative + str(seed)).encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(key)
    img = Image.new("RGB", (128, 128), tuple(int(v) for v in rng.integers(40, 210, 3)))
    d = ImageDraw.Draw(img, "RGBA")
    words = prompt.lower()
    color = (245, 220, 80, 190) if "sun" in words or "light" in words else (95, 160, 235, 185)
    if "cat" in words:
        d.ellipse((38, 42, 90, 92), fill=color); d.polygon([(45, 45), (55, 25), (62, 48)], fill=color); d.polygon([(76, 48), (86, 25), (88, 48)], fill=color)
    elif "city" in words:
        for x in range(12, 118, 18):
            h = int(rng.integers(35, 90)); d.rectangle((x, 118 - h, x + 13, 118), fill=color)
    else:
        for _ in range(5):
            x, y = int(rng.integers(8, 90)), int(rng.integers(8, 90))
            d.rounded_rectangle((x, y, x + 28, y + 24), radius=5, fill=color)
    blur = max(0.2, 3.0 - steps / 14)
    img = img.filter(ImageFilter.GaussianBlur(blur))
    return np.asarray(img)
