"""Color space channel extraction and image interpolation demo.

This script is written for the "Vibe Coding: color space and interpolation"
assignment.  It uses OpenCV for image I/O while keeping the core RGB->HSV,
nearest-neighbor interpolation, and bilinear interpolation algorithms explicit
in Python/Numpy so the implementation is easy to inspect.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_rgb_image(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_rgb_image(path: Path, rgb: np.ndarray) -> None:
    ensure_dir(path.parent)
    rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))


def normalize_to_u8(channel: np.ndarray) -> np.ndarray:
    channel = channel.astype(np.float32)
    min_value = float(channel.min())
    max_value = float(channel.max())
    if max_value - min_value < 1e-6:
        return np.zeros(channel.shape, dtype=np.uint8)
    normalized = (channel - min_value) / (max_value - min_value)
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def save_gray_channel(path: Path, channel: np.ndarray) -> None:
    ensure_dir(path.parent)
    cv2.imwrite(str(path), normalize_to_u8(channel))


def save_rgb_color_channel(path: Path, rgb: np.ndarray, channel_index: int) -> None:
    isolated = np.zeros_like(rgb)
    isolated[:, :, channel_index] = rgb[:, :, channel_index]
    write_rgb_image(path, isolated)


def rgb_to_hsv_manual(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 image to HSV uint8 image with H in OpenCV's 0..179 range."""
    rgb_float = rgb.astype(np.float32) / 255.0
    r = rgb_float[:, :, 0]
    g = rgb_float[:, :, 1]
    b = rgb_float[:, :, 2]

    max_c = np.max(rgb_float, axis=2)
    min_c = np.min(rgb_float, axis=2)
    delta = max_c - min_c

    hue = np.zeros_like(max_c)
    mask = delta > 1e-6

    r_is_max = (max_c == r) & mask
    g_is_max = (max_c == g) & mask
    b_is_max = (max_c == b) & mask

    hue[r_is_max] = ((g[r_is_max] - b[r_is_max]) / delta[r_is_max]) % 6.0
    hue[g_is_max] = ((b[g_is_max] - r[g_is_max]) / delta[g_is_max]) + 2.0
    hue[b_is_max] = ((r[b_is_max] - g[b_is_max]) / delta[b_is_max]) + 4.0
    hue_degrees = hue * 60.0

    saturation = np.zeros_like(max_c)
    non_black = max_c > 1e-6
    saturation[non_black] = delta[non_black] / max_c[non_black]
    value = max_c

    hsv = np.zeros_like(rgb, dtype=np.uint8)
    hsv[:, :, 0] = np.clip(hue_degrees / 2.0, 0, 179).astype(np.uint8)
    hsv[:, :, 1] = np.clip(saturation * 255.0, 0, 255).astype(np.uint8)
    hsv[:, :, 2] = np.clip(value * 255.0, 0, 255).astype(np.uint8)
    return hsv


def save_color_space_channels(rgb: np.ndarray, output_dir: Path) -> dict[str, Path]:
    channel_dir = ensure_dir(output_dir / "color_spaces")

    spaces: dict[str, tuple[np.ndarray, tuple[str, ...]]] = {
        "RGB": (rgb, ("R", "G", "B")),
        "HSV_manual": (rgb_to_hsv_manual(rgb), ("H", "S", "V")),
        "YCrCb_cv2": (cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb), ("Y", "Cr", "Cb")),
        "Lab_cv2": (cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB), ("L", "a", "b")),
    }

    saved: dict[str, Path] = {}
    for space_name, (space_img, channel_names) in spaces.items():
        for index, channel_name in enumerate(channel_names):
            gray_path = channel_dir / f"{space_name}_{channel_name}_gray.png"
            save_gray_channel(gray_path, space_img[:, :, index])
            saved[f"{space_name}_{channel_name}_gray"] = gray_path

            if space_name == "RGB":
                color_path = channel_dir / f"{space_name}_{channel_name}_color.png"
                save_rgb_color_channel(color_path, rgb, index)
                saved[f"{space_name}_{channel_name}_color"] = color_path

    hsv_visual = cv2.cvtColor(spaces["HSV_manual"][0], cv2.COLOR_HSV2RGB)
    hsv_visual_path = channel_dir / "HSV_manual_visualized_as_RGB.png"
    write_rgb_image(hsv_visual_path, hsv_visual)
    saved["HSV_manual_visualized_as_RGB"] = hsv_visual_path
    return saved


def nearest_resize(image: np.ndarray, new_width: int, new_height: int) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    dst = np.zeros((new_height, new_width, image.shape[2]), dtype=image.dtype)
    x_scale = src_w / new_width
    y_scale = src_h / new_height

    for y in range(new_height):
        src_y = min(int(round((y + 0.5) * y_scale - 0.5)), src_h - 1)
        src_y = max(src_y, 0)
        for x in range(new_width):
            src_x = min(int(round((x + 0.5) * x_scale - 0.5)), src_w - 1)
            src_x = max(src_x, 0)
            dst[y, x] = image[src_y, src_x]
    return dst


def bilinear_resize(image: np.ndarray, new_width: int, new_height: int) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    dst = np.zeros((new_height, new_width, image.shape[2]), dtype=np.float32)
    x_scale = src_w / new_width
    y_scale = src_h / new_height

    for y in range(new_height):
        src_y = (y + 0.5) * y_scale - 0.5
        y0 = int(math.floor(src_y))
        y1 = y0 + 1
        wy = src_y - y0
        y0 = min(max(y0, 0), src_h - 1)
        y1 = min(max(y1, 0), src_h - 1)

        for x in range(new_width):
            src_x = (x + 0.5) * x_scale - 0.5
            x0 = int(math.floor(src_x))
            x1 = x0 + 1
            wx = src_x - x0
            x0 = min(max(x0, 0), src_w - 1)
            x1 = min(max(x1, 0), src_w - 1)

            top = (1.0 - wx) * image[y0, x0] + wx * image[y0, x1]
            bottom = (1.0 - wx) * image[y1, x0] + wx * image[y1, x1]
            dst[y, x] = (1.0 - wy) * top + wy * bottom

    return np.clip(dst, 0, 255).astype(image.dtype)


def sample_nearest(image: np.ndarray, x: float, y: float, background: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    xi = int(round(x))
    yi = int(round(y))
    if xi < 0 or xi >= w or yi < 0 or yi >= h:
        return background
    return image[yi, xi]


def sample_bilinear(image: np.ndarray, x: float, y: float, background: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if x < 0 or x > w - 1 or y < 0 or y > h - 1:
        return background

    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    wx = x - x0
    wy = y - y0

    top = (1.0 - wx) * image[y0, x0].astype(np.float32) + wx * image[y0, x1].astype(np.float32)
    bottom = (1.0 - wx) * image[y1, x0].astype(np.float32) + wx * image[y1, x1].astype(np.float32)
    value = (1.0 - wy) * top + wy * bottom
    return np.clip(value, 0, 255).astype(image.dtype)


def rotate_and_stretch(
    image: np.ndarray,
    angle_degrees: float,
    stretch_x: float,
    stretch_y: float,
    method: str,
) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    out_w = max(1, int(round(src_w * stretch_x)))
    out_h = max(1, int(round(src_h * stretch_y)))
    dst = np.zeros((out_h, out_w, image.shape[2]), dtype=image.dtype)

    src_cx = (src_w - 1) / 2.0
    src_cy = (src_h - 1) / 2.0
    dst_cx = (out_w - 1) / 2.0
    dst_cy = (out_h - 1) / 2.0
    radians = math.radians(angle_degrees)
    cos_a = math.cos(radians)
    sin_a = math.sin(radians)
    background = np.array([245, 245, 245], dtype=image.dtype)

    sampler = sample_nearest if method == "nearest" else sample_bilinear
    for y in range(out_h):
        for x in range(out_w):
            dx = (x - dst_cx) / stretch_x
            dy = (y - dst_cy) / stretch_y
            src_x = cos_a * dx + sin_a * dy + src_cx
            src_y = -sin_a * dx + cos_a * dy + src_cy
            dst[y, x] = sampler(image, src_x, src_y, background)
    return dst


def center_crop_or_pad(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    h, w = image.shape[:2]
    result = np.full((target_height, target_width, image.shape[2]), 245, dtype=image.dtype)

    crop_w = min(w, target_width)
    crop_h = min(h, target_height)
    src_x0 = max((w - crop_w) // 2, 0)
    src_y0 = max((h - crop_h) // 2, 0)
    dst_x0 = max((target_width - crop_w) // 2, 0)
    dst_y0 = max((target_height - crop_h) // 2, 0)

    result[dst_y0 : dst_y0 + crop_h, dst_x0 : dst_x0 + crop_w] = image[
        src_y0 : src_y0 + crop_h, src_x0 : src_x0 + crop_w
    ]
    return result


def save_interpolation_outputs(rgb: np.ndarray, output_dir: Path) -> dict[str, Path]:
    interp_dir = ensure_dir(output_dir / "interpolation")
    h, w = rgb.shape[:2]
    demo = rgb
    max_side = max(h, w)
    if max_side > 360:
        scale = 360 / max_side
        demo = bilinear_resize(rgb, int(round(w * scale)), int(round(h * scale)))
        h, w = demo.shape[:2]

    cases = {
        "nearest_zoom_160": nearest_resize(demo, int(w * 1.6), int(h * 1.6)),
        "bilinear_zoom_160": bilinear_resize(demo, int(w * 1.6), int(h * 1.6)),
        "nearest_shrink_055": nearest_resize(demo, max(1, int(w * 0.55)), max(1, int(h * 0.55))),
        "bilinear_shrink_055": bilinear_resize(demo, max(1, int(w * 0.55)), max(1, int(h * 0.55))),
        "nearest_rotate_stretch": rotate_and_stretch(demo, 25, 1.18, 0.82, "nearest"),
        "bilinear_rotate_stretch": rotate_and_stretch(demo, 25, 1.18, 0.82, "bilinear"),
    }

    saved: dict[str, Path] = {}
    for name, image in cases.items():
        path = interp_dir / f"{name}.png"
        write_rgb_image(path, image)
        saved[name] = path
    return saved


def make_collage(
    images: list[np.ndarray],
    titles: list[str],
    output_path: Path,
    columns: int = 3,
    figure_size: tuple[float, float] = (12, 8),
) -> None:
    ensure_dir(output_path.parent)
    rows = math.ceil(len(images) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=figure_size)
    axes_array = np.array(axes).reshape(-1)
    for axis in axes_array:
        axis.axis("off")

    for axis, image, title in zip(axes_array, images, titles):
        if image.ndim == 2:
            axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        else:
            axis.imshow(image)
        axis.set_title(title, fontsize=11)
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def load_rgb_or_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def make_screenshots(rgb: np.ndarray, output_dir: Path, saved_paths: dict[str, Path]) -> dict[str, Path]:
    screenshot_dir = ensure_dir(output_dir / "screenshots")

    color_keys = [
        "RGB_R_gray",
        "RGB_G_gray",
        "RGB_B_gray",
        "HSV_manual_H_gray",
        "HSV_manual_S_gray",
        "HSV_manual_V_gray",
        "YCrCb_cv2_Y_gray",
        "YCrCb_cv2_Cr_gray",
        "YCrCb_cv2_Cb_gray",
    ]
    make_collage(
        [load_rgb_or_gray(saved_paths[key]) for key in color_keys],
        ["RGB R", "RGB G", "RGB B", "HSV H", "HSV S", "HSV V", "YCrCb Y", "YCrCb Cr", "YCrCb Cb"],
        screenshot_dir / "color_space_channels.png",
        columns=3,
        figure_size=(11, 10),
    )

    interp_keys = [
        "nearest_zoom_160",
        "bilinear_zoom_160",
        "nearest_shrink_055",
        "bilinear_shrink_055",
        "nearest_rotate_stretch",
        "bilinear_rotate_stretch",
    ]
    interp_images = [load_rgb_or_gray(saved_paths[key]) for key in interp_keys]
    base_h, base_w = rgb.shape[:2]
    view_w = min(480, max(220, base_w))
    view_h = min(340, max(180, base_h))
    interp_images = [center_crop_or_pad(img, view_w, view_h) for img in interp_images]
    make_collage(
        interp_images,
        [
            "Nearest zoom 1.6x",
            "Bilinear zoom 1.6x",
            "Nearest shrink 0.55x",
            "Bilinear shrink 0.55x",
            "Nearest rotate/stretch",
            "Bilinear rotate/stretch",
        ],
        screenshot_dir / "interpolation_results.png",
        columns=2,
        figure_size=(10, 10),
    )

    zoom_nearest = load_rgb_or_gray(saved_paths["nearest_zoom_160"])
    zoom_bilinear = load_rgb_or_gray(saved_paths["bilinear_zoom_160"])
    zh, zw = zoom_nearest.shape[:2]
    crop = (
        zoom_nearest[zh // 3 : zh // 3 + min(160, zh // 2), zw // 3 : zw // 3 + min(220, zw // 2)],
        zoom_bilinear[zh // 3 : zh // 3 + min(160, zh // 2), zw // 3 : zw // 3 + min(220, zw // 2)],
    )
    make_collage(
        [crop[0], crop[1]],
        ["Nearest detail", "Bilinear detail"],
        screenshot_dir / "interpolation_detail_compare.png",
        columns=2,
        figure_size=(8, 4),
    )

    return {
        "color_space_channels": screenshot_dir / "color_space_channels.png",
        "interpolation_results": screenshot_dir / "interpolation_results.png",
        "interpolation_detail_compare": screenshot_dir / "interpolation_detail_compare.png",
    }


def run_demo(input_path: Path, output_dir: Path) -> dict[str, Path]:
    rgb = read_rgb_image(input_path)
    ensure_dir(output_dir)
    write_rgb_image(output_dir / "input_rgb.png", rgb)

    saved = {}
    saved.update(save_color_space_channels(rgb, output_dir))
    saved.update(save_interpolation_outputs(rgb, output_dir))
    saved.update(make_screenshots(rgb, output_dir, saved))
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Color spaces and interpolation assignment demo.")
    parser.add_argument("--input", type=Path, required=True, help="Input image path.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved = run_demo(args.input, args.output)
    print("Generated files:")
    for key, path in sorted(saved.items()):
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()
