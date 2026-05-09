from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


MAX_WARP_PIXELS = 12_000_000
MAX_WARP_SIDE = 4200


@dataclass
class MatchPipelineResult:
    keypoints_1: list
    keypoints_2: list
    descriptors_1: np.ndarray | None
    descriptors_2: np.ndarray | None
    initial_matches: list
    inlier_matches: list
    homography: np.ndarray | None
    inlier_mask: np.ndarray | None
    feature_view_1: np.ndarray
    feature_view_2: np.ndarray
    initial_view: np.ndarray
    ransac_view: np.ndarray
    transform_view: np.ndarray
    aligned_view: np.ndarray


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def resize_for_work(image: np.ndarray, max_side: int = 1100) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale == 1.0:
        return image.copy()
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def side_by_side(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    target_h = max(h1, h2)
    canvas = np.zeros((target_h, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = _ensure_bgr(img1)
    canvas[:h2, w1:w1 + w2] = _ensure_bgr(img2)
    return canvas


def non_maximum_suppression(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    h, w = magnitude.shape
    angle = (np.rad2deg(direction) + 180) % 180
    suppressed = np.zeros((h, w), dtype=np.float32)

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            a = angle[y, x]
            q = 0.0
            r = 0.0

            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                q = magnitude[y, x + 1]
                r = magnitude[y, x - 1]
            elif 22.5 <= a < 67.5:
                q = magnitude[y + 1, x - 1]
                r = magnitude[y - 1, x + 1]
            elif 67.5 <= a < 112.5:
                q = magnitude[y + 1, x]
                r = magnitude[y - 1, x]
            else:
                q = magnitude[y - 1, x - 1]
                r = magnitude[y + 1, x + 1]

            if magnitude[y, x] >= q and magnitude[y, x] >= r:
                suppressed[y, x] = magnitude[y, x]

    return suppressed


def canny_visualization(
    image: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
    sigma: float = 1.2,
) -> dict[str, np.ndarray]:
    image = resize_for_work(_ensure_bgr(image))
    gray = _to_gray(image)
    blurred = cv2.GaussianBlur(gray, (5, 5), sigma)

    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.hypot(grad_x, grad_y).astype(np.float32)
    direction = np.arctan2(grad_y, grad_x)

    nms = non_maximum_suppression(magnitude, direction)
    edges = cv2.Canny(blurred, low_threshold, high_threshold, L2gradient=True)

    before_nms = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    after_nms = cv2.normalize(nms, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    after_nms_thresholded = np.where(after_nms >= low_threshold, after_nms, 0).astype(np.uint8)
    overlay = image.copy()
    overlay[edges > 0] = (0, 220, 80)
    overlay = cv2.addWeighted(image, 0.72, overlay, 0.28, 0)

    return {
        "original": image,
        "gray": gray,
        "blurred": blurred,
        "gradient_magnitude": before_nms,
        "before_nms": before_nms,
        "after_nms": after_nms_thresholded,
        "canny_edges": edges,
        "overlay": overlay,
    }


def harris_features(
    image: np.ndarray,
    threshold_ratio: float = 0.01,
    max_points: int = 350,
    radius: int = 7,
) -> dict[str, object]:
    image = resize_for_work(_ensure_bgr(image))
    gray = _to_gray(image)
    response = cv2.cornerHarris(np.float32(gray), blockSize=3, ksize=3, k=0.04)
    response = cv2.dilate(response, None)
    threshold = threshold_ratio * response.max()
    ys, xs = np.where(response > threshold)

    candidates = sorted(
        ((float(response[y, x]), int(x), int(y)) for y, x in zip(ys, xs)),
        reverse=True,
    )
    selected: list[tuple[int, int, float]] = []
    min_dist_sq = (radius * 2) ** 2
    for score, x, y in candidates:
        if all((x - px) ** 2 + (y - py) ** 2 >= min_dist_sq for px, py, _ in selected):
            selected.append((x, y, score))
        if len(selected) >= max_points:
            break

    view = image.copy()
    for x, y, score in selected:
        local_radius = int(np.clip(radius + score / (response.max() + 1e-6) * 8, 5, 16))
        cv2.circle(view, (x, y), local_radius, (255, 95, 0), 1, cv2.LINE_AA)
        cv2.circle(view, (x, y), 2, (0, 255, 120), -1, cv2.LINE_AA)

    positive = np.clip(response, 0, None)
    nonzero = positive[positive > 0]
    if nonzero.size:
        high = float(np.percentile(nonzero, 99.5))
        if high <= 0:
            high = float(nonzero.max())
        heat_gray = np.clip(positive / (high + 1e-6) * 255, 0, 255).astype(np.uint8)
    else:
        heat_gray = np.zeros_like(gray, dtype=np.uint8)
    heat_color = cv2.applyColorMap(heat_gray, cv2.COLORMAP_INFERNO)
    heat = cv2.addWeighted(image, 0.42, heat_color, 0.58, 0)
    for x, y, _ in selected[:120]:
        cv2.circle(heat, (x, y), 3, (0, 255, 255), -1, cv2.LINE_AA)
    return {"points": selected, "response": response, "view": view, "heatmap": heat}


def make_sift(nfeatures: int = 500):
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=nfeatures), "SIFT"
    return cv2.ORB_create(nfeatures=nfeatures), "ORB"


def sift_features(image: np.ndarray, nfeatures: int = 500) -> dict[str, object]:
    image = resize_for_work(_ensure_bgr(image))
    gray = _to_gray(image)
    detector, method = make_sift(nfeatures=nfeatures)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    view = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        color=(40, 210, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    return {"method": method, "keypoints": keypoints, "descriptors": descriptors, "view": view}


def feature_comparison(image: np.ndarray, harris_threshold: float = 0.01, sift_nfeatures: int = 500) -> dict[str, object]:
    return {
        "harris": harris_features(image, threshold_ratio=harris_threshold),
        "sift": sift_features(image, nfeatures=sift_nfeatures),
    }


def _detector_and_norm(detector_name: str = "SIFT", nfeatures: int = 700):
    name = detector_name.upper()
    if name == "SIFT" and hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=nfeatures), cv2.NORM_L2, "SIFT"
    return cv2.ORB_create(nfeatures=nfeatures), cv2.NORM_HAMMING, "ORB"


def _ratio_matches(descriptors_1: np.ndarray, descriptors_2: np.ndarray, norm: int, ratio: float) -> list:
    matcher = cv2.BFMatcher(norm, crossCheck=False)
    raw = matcher.knnMatch(descriptors_1, descriptors_2, k=2)
    good = []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return sorted(good, key=lambda m: m.distance)


def match_pipeline(
    img1: np.ndarray,
    img2: np.ndarray,
    detector_name: str = "SIFT",
    ratio: float = 0.75,
    ransac_threshold: float = 4.0,
    nfeatures: int = 900,
) -> MatchPipelineResult:
    img1 = resize_for_work(_ensure_bgr(img1))
    img2 = resize_for_work(_ensure_bgr(img2))
    gray1 = _to_gray(img1)
    gray2 = _to_gray(img2)

    detector, norm, _ = _detector_and_norm(detector_name, nfeatures)
    keypoints_1, descriptors_1 = detector.detectAndCompute(gray1, None)
    keypoints_2, descriptors_2 = detector.detectAndCompute(gray2, None)

    feature_view_1 = cv2.drawKeypoints(img1, keypoints_1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    feature_view_2 = cv2.drawKeypoints(img2, keypoints_2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    if descriptors_1 is None or descriptors_2 is None or len(keypoints_1) < 4 or len(keypoints_2) < 4:
        blank = side_by_side(img1, img2)
        return MatchPipelineResult(keypoints_1, keypoints_2, descriptors_1, descriptors_2, [], [], None, None, feature_view_1, feature_view_2, blank, blank, img1, blank)

    initial_matches = _ratio_matches(descriptors_1, descriptors_2, norm, ratio)
    initial_view = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, initial_matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    homography = None
    inlier_mask = None
    inlier_matches: list = []
    ransac_view = initial_view.copy()
    transform_view = img1.copy()
    aligned_view = side_by_side(img1, img2)

    if len(initial_matches) >= 4:
        src = np.float32([keypoints_1[m.queryIdx].pt for m in initial_matches]).reshape(-1, 1, 2)
        dst = np.float32([keypoints_2[m.trainIdx].pt for m in initial_matches]).reshape(-1, 1, 2)
        homography, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_threshold)
        if inlier_mask is not None:
            flat_mask = inlier_mask.ravel().astype(bool)
            inlier_matches = [m for m, keep in zip(initial_matches, flat_mask) if keep]
        ransac_view = cv2.drawMatches(
            img1,
            keypoints_1,
            img2,
            keypoints_2,
            inlier_matches[:100],
            None,
            matchColor=(0, 220, 80),
            singlePointColor=(80, 80, 255),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

    if homography is not None and len(inlier_matches) >= 8:
        warped = warp_and_align(img1, img2, homography)
        if warped is not None:
            transform_view, aligned_view = warped
        else:
            homography = None
    elif homography is not None:
        homography = None

    return MatchPipelineResult(
        keypoints_1,
        keypoints_2,
        descriptors_1,
        descriptors_2,
        initial_matches,
        inlier_matches,
        homography,
        inlier_mask,
        feature_view_1,
        feature_view_2,
        initial_view,
        ransac_view,
        transform_view,
        aligned_view,
    )


def safe_panorama_bounds(
    img1: np.ndarray,
    img2: np.ndarray,
    homography: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]] | None:
    try:
        translation, (width, height), offset = panorama_bounds(img1, img2, homography)
    except cv2.error:
        return None
    if width <= 0 or height <= 0:
        return None
    if width * height > MAX_WARP_PIXELS or max(width, height) > MAX_WARP_SIDE:
        return None
    if not np.all(np.isfinite(translation)):
        return None
    return translation, (width, height), offset


def panorama_bounds(img1: np.ndarray, img2: np.ndarray, homography: np.ndarray) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    warped_corners1 = cv2.perspectiveTransform(corners1, homography)
    all_corners = np.concatenate([warped_corners1, corners2], axis=0)
    x_min, y_min = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0).ravel()).astype(int)
    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
    return translation, (x_max - x_min, y_max - y_min), (-x_min, -y_min)


def warp_and_align(img1: np.ndarray, img2: np.ndarray, homography: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    bounds = safe_panorama_bounds(img1, img2, homography)
    if bounds is None:
        return None
    translation, (width, height), (x_offset, y_offset) = bounds
    warped_1 = cv2.warpPerspective(img1, translation @ homography, (width, height))
    warped_2 = np.zeros_like(warped_1)
    h2, w2 = img2.shape[:2]
    warped_2[y_offset:y_offset + h2, x_offset:x_offset + w2] = img2
    aligned = cv2.addWeighted(warped_1, 0.5, warped_2, 0.5, 0)
    return warped_1, aligned


def _canvas_images(img1: np.ndarray, img2: np.ndarray, homography: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    bounds = safe_panorama_bounds(img1, img2, homography)
    if bounds is None:
        return None
    translation, (width, height), (x_offset, y_offset) = bounds
    warped_1 = cv2.warpPerspective(img1, translation @ homography, (width, height))
    mask_1_src = np.ones(img1.shape[:2], dtype=np.uint8) * 255
    mask_1 = cv2.warpPerspective(mask_1_src, translation @ homography, (width, height))

    warped_2 = np.zeros_like(warped_1)
    mask_2 = np.zeros((height, width), dtype=np.uint8)
    h2, w2 = img2.shape[:2]
    warped_2[y_offset:y_offset + h2, x_offset:x_offset + w2] = img2
    mask_2[y_offset:y_offset + h2, x_offset:x_offset + w2] = 255
    return warped_1, warped_2, mask_1, mask_2


def crop_black_border(image: np.ndarray) -> np.ndarray:
    gray = _to_gray(image)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    return image[y:y + h, x:x + w]


def blend_images(warped_1: np.ndarray, warped_2: np.ndarray, mask_1: np.ndarray, mask_2: np.ndarray, method: str) -> np.ndarray:
    method = method.lower()
    if method == "none":
        result = warped_1.copy()
        result[mask_2 > 0] = warped_2[mask_2 > 0]
        return crop_black_border(result)

    mask_1_f = (mask_1 > 0).astype(np.float32)
    mask_2_f = (mask_2 > 0).astype(np.float32)

    if method == "average":
        weight_1 = mask_1_f
        weight_2 = mask_2_f
    else:
        dist_1 = cv2.distanceTransform((mask_1 > 0).astype(np.uint8), cv2.DIST_L2, 5)
        dist_2 = cv2.distanceTransform((mask_2 > 0).astype(np.uint8), cv2.DIST_L2, 5)
        if method == "multiband":
            dist_1 = cv2.GaussianBlur(dist_1, (0, 0), 9)
            dist_2 = cv2.GaussianBlur(dist_2, (0, 0), 9)
        weight_1 = dist_1
        weight_2 = dist_2

    total = weight_1 + weight_2
    total[total == 0] = 1
    alpha_1 = (weight_1 / total)[:, :, None]
    alpha_2 = (weight_2 / total)[:, :, None]
    blended = warped_1.astype(np.float32) * alpha_1 + warped_2.astype(np.float32) * alpha_2
    return crop_black_border(np.clip(blended, 0, 255).astype(np.uint8))


def stitch_pair(
    img1: np.ndarray,
    img2: np.ndarray,
    blending: str = "feather",
    detector_name: str = "SIFT",
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray | None, MatchPipelineResult]:
    result = match_pipeline(img1, img2, detector_name=detector_name, ratio=0.75, ransac_threshold=4.0, nfeatures=1200)
    if result.homography is None or len(result.inlier_matches) < 4:
        return None, result.ransac_view, None, result

    canvas = _canvas_images(resize_for_work(_ensure_bgr(img1)), resize_for_work(_ensure_bgr(img2)), result.homography)
    if canvas is None:
        return None, result.ransac_view, None, result
    warped_1, warped_2, mask_1, mask_2 = canvas
    panorama = blend_images(warped_1, warped_2, mask_1, mask_2, blending)
    return panorama, result.ransac_view, result.homography, result


def stitch_many(
    images: Iterable[np.ndarray],
    blending: str = "feather",
    detector_name: str = "SIFT",
) -> tuple[np.ndarray | None, list[dict[str, object]]]:
    image_list = [resize_for_work(_ensure_bgr(img)) for img in images]
    if len(image_list) < 2:
        return None, []

    panorama = image_list[0]
    steps: list[dict[str, object]] = []
    for idx, next_image in enumerate(image_list[1:], start=2):
        panorama, matches_view, homography, pipeline = stitch_pair(panorama, next_image, blending=blending, detector_name=detector_name)
        steps.append(
            {
                "step": idx - 1,
                "image_index": idx,
                "matches_view": matches_view,
                "homography": homography,
                "initial_matches": len(pipeline.initial_matches),
                "inliers": len(pipeline.inlier_matches),
                "panorama": panorama,
            }
        )
        if panorama is None:
            return None, steps
    return panorama, steps
