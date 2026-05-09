from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def normalize_to_u8(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if mx - mn < 1e-8:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip((arr - mn) * 255.0 / (mx - mn), 0, 255).astype(np.uint8)


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float32)
    return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]


def ensure_odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def convolve2d_gray(gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kernel = np.asarray(kernel, dtype=np.float32)
    pad_y, pad_x = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded = np.pad(gray.astype(np.float32), ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")
    out = np.zeros_like(gray, dtype=np.float32)
    flipped = kernel[::-1, ::-1]
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            patch = padded[y : y + kernel.shape[0], x : x + kernel.shape[1]]
            out[y, x] = float(np.sum(patch * flipped))
    return out


def apply_gray_kernel_to_rgb(rgb: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    channels = [convolve2d_gray(rgb[:, :, idx], kernel) for idx in range(3)]
    return np.clip(np.stack(channels, axis=2), 0, 255).astype(np.uint8)


def box_filter(rgb: np.ndarray, size: int = 5) -> np.ndarray:
    size = ensure_odd(size)
    kernel = np.ones((size, size), dtype=np.float32) / float(size * size)
    return apply_gray_kernel_to_rgb(rgb, kernel)


def gaussian_kernel(size: int = 5, sigma: float = 1.2) -> np.ndarray:
    size = ensure_odd(size)
    radius = size // 2
    axis = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    kernel = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)


def gaussian_filter(rgb: np.ndarray, size: int = 5, sigma: float = 1.2) -> np.ndarray:
    return apply_gray_kernel_to_rgb(rgb, gaussian_kernel(size, sigma))


def median_filter(rgb: np.ndarray, size: int = 5) -> np.ndarray:
    size = ensure_odd(size)
    radius = size // 2
    padded = np.pad(rgb, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")
    out = np.zeros_like(rgb)
    for y in range(rgb.shape[0]):
        for x in range(rgb.shape[1]):
            patch = padded[y : y + size, x : x + size, :]
            out[y, x, :] = np.median(patch, axis=(0, 1))
    return out.astype(np.uint8)


def sobel_gradients(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    gx = convolve2d_gray(gray, kx)
    gy = convolve2d_gray(gray, ky)
    magnitude = np.hypot(gx, gy)
    angle = np.degrees(np.arctan2(gy, gx))
    return gx, gy, magnitude, angle


def sobel_edge_image(rgb: np.ndarray) -> np.ndarray:
    _, _, magnitude, _ = sobel_gradients(rgb_to_gray(rgb))
    return normalize_to_u8(magnitude)


def compare_spatial_filters(rgb: np.ndarray, size: int = 5, sigma: float = 1.2) -> dict[str, np.ndarray]:
    return {
        "Original": rgb,
        "Box": box_filter(rgb, size),
        "Gaussian": gaussian_filter(rgb, size, sigma),
        "Median": median_filter(rgb, size),
        "Sobel": sobel_edge_image(rgb),
    }


def gradient_region_demo(
    rgb: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
) -> dict[str, np.ndarray | float | tuple[int, int, int, int]]:
    gray = rgb_to_gray(rgb)
    gx, gy, magnitude, angle = sobel_gradients(gray)
    h, w = gray.shape
    x = int(np.clip(x, 0, max(0, w - 1)))
    y = int(np.clip(y, 0, max(0, h - 1)))
    width = int(np.clip(width, 1, w - x))
    height = int(np.clip(height, 1, h - y))
    roi_mag = magnitude[y : y + height, x : x + width]
    roi_gx = gx[y : y + height, x : x + width]
    roi_gy = gy[y : y + height, x : x + width]
    mean_gx = float(np.mean(roi_gx))
    mean_gy = float(np.mean(roi_gy))
    mean_mag = float(np.mean(roi_mag))
    mean_angle = float(np.degrees(np.arctan2(mean_gy, mean_gx)))

    boxed = rgb.copy()
    cv2.rectangle(boxed, (x, y), (x + width, y + height), (255, 40, 40), 3)
    center = (x + width // 2, y + height // 2)
    arrow_len = max(24, min(width, height) // 2)
    theta = np.radians(mean_angle)
    end = (
        int(center[0] + arrow_len * np.cos(theta)),
        int(center[1] + arrow_len * np.sin(theta)),
    )
    cv2.arrowedLine(boxed, center, end, (30, 220, 80), 4, tipLength=0.25)
    return {
        "boxed": boxed,
        "gx": gx,
        "gy": gy,
        "magnitude": magnitude,
        "angle": angle,
        "mean_gx": mean_gx,
        "mean_gy": mean_gy,
        "mean_magnitude": mean_mag,
        "mean_angle": mean_angle,
        "roi": (x, y, width, height),
    }


def fft_spectrum(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fshift = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    spectrum = normalize_to_u8(np.log1p(np.abs(fshift)))
    return fshift, spectrum


def circular_frequency_mask(shape: tuple[int, int], radius_ratio: float, mode: str) -> np.ndarray:
    h, w = shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radius = max(1.0, float(radius_ratio) * min(h, w) / 2.0)
    low = dist <= radius
    if mode == "low-pass":
        return low.astype(np.float32)
    if mode == "high-pass":
        return (~low).astype(np.float32)
    if mode == "band-pass":
        inner = radius * 0.45
        outer = radius
        return ((dist >= inner) & (dist <= outer)).astype(np.float32)
    raise ValueError(f"Unsupported mode: {mode}")


def frequency_filter(rgb: np.ndarray, mode: str = "low-pass", radius_ratio: float = 0.22) -> dict[str, np.ndarray]:
    gray = rgb_to_gray(rgb)
    fshift, spectrum = fft_spectrum(gray)
    mask = circular_frequency_mask(gray.shape, radius_ratio, mode)
    filtered_shift = fshift * mask
    restored = np.real(np.fft.ifft2(np.fft.ifftshift(filtered_shift)))
    return {
        "gray": normalize_to_u8(gray),
        "spectrum": spectrum,
        "mask": (mask * 255).astype(np.uint8),
        "filtered_spectrum": normalize_to_u8(np.log1p(np.abs(filtered_shift))),
        "restored": normalize_to_u8(restored),
    }


def transform_image(rgb: np.ndarray, transform: str) -> np.ndarray:
    h, w = rgb.shape[:2]
    if transform == "original":
        return rgb.copy()
    if transform == "translate":
        mat = np.float32([[1, 0, w * 0.14], [0, 1, h * 0.10]])
        return cv2.warpAffine(rgb, mat, (w, h), borderMode=cv2.BORDER_REFLECT)
    if transform == "rotate":
        mat = cv2.getRotationMatrix2D((w / 2, h / 2), 32, 1.0)
        return cv2.warpAffine(rgb, mat, (w, h), borderMode=cv2.BORDER_REFLECT)
    if transform == "scale":
        scaled = cv2.resize(rgb, None, fx=0.72, fy=0.72, interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros_like(rgb)
        sh, sw = scaled.shape[:2]
        y0 = (h - sh) // 2
        x0 = (w - sw) // 2
        canvas[y0 : y0 + sh, x0 : x0 + sw] = scaled
        return canvas
    raise ValueError(f"Unsupported transform: {transform}")


def transform_spectra(rgb: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in ("original", "translate", "rotate", "scale"):
        transformed = transform_image(rgb, name)
        _, spectrum = fft_spectrum(rgb_to_gray(transformed))
        result[name] = (transformed, spectrum)
    return result


def load_rgb(path: str | Path, max_side: int = 720) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image.thumbnail((max_side, max_side))
    return np.asarray(image)


def save_rgb(path: str | Path, image: np.ndarray) -> None:
    arr = image
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=2)
    Image.fromarray(arr.astype(np.uint8)).save(path)


def build_demo_figures(input_path: str | Path, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rgb = load_rgb(input_path, max_side=520)

    spatial = compare_spatial_filters(rgb, 5, 1.2)
    fig, axes = plt.subplots(1, len(spatial), figsize=(15, 3.5))
    for ax, (name, img) in zip(axes, spatial.items()):
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(name)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "screenshot_01_spatial_filters.png", dpi=180)
    plt.close(fig)

    gd = gradient_region_demo(rgb, rgb.shape[1] // 4, rgb.shape[0] // 4, rgb.shape[1] // 3, rgb.shape[0] // 3)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(gd["boxed"])
    axes[0].set_title(f"ROI angle {gd['mean_angle']:.1f} deg")
    axes[1].imshow(normalize_to_u8(gd["magnitude"]), cmap="gray")
    axes[1].set_title("Gradient magnitude")
    axes[2].imshow(normalize_to_u8(gd["angle"]), cmap="twilight")
    axes[2].set_title("Gradient direction")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "screenshot_02_gradient_region.png", dpi=180)
    plt.close(fig)

    freq = frequency_filter(rgb, "low-pass", 0.22)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
    titles = ["Gray", "Spectrum", "Mask", "Filtered spectrum", "Inverse result"]
    imgs = [freq["gray"], freq["spectrum"], freq["mask"], freq["filtered_spectrum"], freq["restored"]]
    for ax, title, img in zip(axes, titles, imgs):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "screenshot_03_frequency_filter.png", dpi=180)
    plt.close(fig)

    spectra = transform_spectra(rgb)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for col, (name, (img, spec)) in enumerate(spectra.items()):
        axes[0, col].imshow(img)
        axes[0, col].set_title(name)
        axes[1, col].imshow(spec, cmap="gray")
        axes[1, col].set_title(f"{name} spectrum")
        axes[0, col].axis("off")
        axes[1, col].axis("off")
    fig.tight_layout()
    fig.savefig(out / "screenshot_04_transform_spectra.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo outputs for Vibe Coding image filtering.")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output screenshot directory")
    args = parser.parse_args()
    build_demo_figures(args.input, args.output)


if __name__ == "__main__":
    main()
