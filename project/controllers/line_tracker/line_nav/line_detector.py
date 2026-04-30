"""Black line + node detector for the down-facing camera.

This module deliberately separates two ideas:

1. line following
   - estimate the center of the currently visible black line inside a narrow ROI
   - used while the robot is in FOLLOW_LINE

2. node detection/classification
   - detect grid nodes such as T, cross, and corner shapes
   - estimate the *geometric node center* as the intersection of the horizontal
     and vertical line axes, not as the centroid of all black pixels
   - used while the robot is in CENTER_NODE

Supported node types:
- CROSS
- T_MISSING_FRONT / T_MISSING_BACK / T_MISSING_LEFT / T_MISSING_RIGHT
- CORNER_FRONT_LEFT / CORNER_FRONT_RIGHT / CORNER_BACK_LEFT / CORNER_BACK_RIGHT
- STRAIGHT_FRONT_BACK / STRAIGHT_LEFT_RIGHT  (reported but not used as a node)
- DEAD_END_*                              (reported but not used by default)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np


@dataclass
class LineDetectionResult:
    # Basic line-following result
    found: bool
    error: float
    center: Tuple[int, int] | None
    mask_area: int
    roi: Tuple[int, int, int, int]
    mask: np.ndarray
    angle_deg: float | None = None
    reason: str = ""

    # General node result.  A node means a decision point such as T/cross/corner.
    node_found: bool = False
    node_type: str = "NONE"
    node_center: Tuple[int, int] | None = None
    node_lateral_error: float = 0.0
    node_forward_error: float = 0.0
    node_area: int = 0
    node_span_x: float = 0.0
    node_span_y: float = 0.0
    node_reason: str = ""
    active_dirs: Tuple[str, ...] = ()  # robot-relative: front/back/left/right
    active_image_dirs: Tuple[str, ...] = ()  # image-relative: up/down/left/right

    # Backward-compatible aliases for the previous intersection-centering code.
    intersection_found: bool = False
    intersection_center: Tuple[int, int] | None = None
    intersection_lateral_error: float = 0.0
    intersection_forward_error: float = 0.0
    intersection_area: int = 0
    intersection_span_x: float = 0.0
    intersection_span_y: float = 0.0
    intersection_reason: str = ""


def _make_roi(width: int, height: int, section: Dict, prefix: str = "roi") -> Tuple[int, int, int, int]:
    x1 = int(width * float(section.get(f"{prefix}_x_min_ratio", 0.0)))
    x2 = int(width * float(section.get(f"{prefix}_x_max_ratio", 1.0)))
    y1 = int(height * float(section.get(f"{prefix}_y_min_ratio", 0.0)))
    y2 = int(height * float(section.get(f"{prefix}_y_max_ratio", 1.0)))

    x1 = max(0, min(width - 1, x1))
    x2 = max(x1 + 1, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def _normalized_lateral_error(
    center_x: float,
    center_y: float,
    width: int,
    height: int,
    forward_dir: str,
) -> float:
    """Return normalized lateral error in [-1, 1] approximately."""
    if forward_dir in {"up", "down"}:
        denom = max(width / 2.0, 1.0)
        return float((center_x - width / 2.0) / denom)
    if forward_dir in {"left", "right"}:
        denom = max(height / 2.0, 1.0)
        return float((center_y - height / 2.0) / denom)
    raise ValueError(f"Invalid forward_dir_in_image: {forward_dir}")


def _normalized_forward_error(
    center_x: float,
    center_y: float,
    width: int,
    height: int,
    forward_dir: str,
) -> float:
    """Return normalized forward/backward error.

    Positive means the node center is in front of the robot, so the robot should
    move forward. Negative means it overshot the center and should move backward.
    """
    if forward_dir == "up":
        return float((height / 2.0 - center_y) / max(height / 2.0, 1.0))
    if forward_dir == "down":
        return float((center_y - height / 2.0) / max(height / 2.0, 1.0))
    if forward_dir == "left":
        return float((width / 2.0 - center_x) / max(width / 2.0, 1.0))
    if forward_dir == "right":
        return float((center_x - width / 2.0) / max(width / 2.0, 1.0))
    raise ValueError(f"Invalid forward_dir_in_image: {forward_dir}")


def _estimate_angle_deg(points_xy: np.ndarray) -> float | None:
    """Estimate dominant line angle in image coordinates using cv2.fitLine."""
    if points_xy.shape[0] < 10:
        return None
    pts = points_xy.astype(np.float32).reshape(-1, 1, 2)
    vx, vy, _, _ = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    angle = np.degrees(np.arctan2(float(vy), float(vx)))
    return float(angle)


def _weighted_peak_index(projection: np.ndarray, threshold_ratio: float) -> int | None:
    """Return weighted average index around the dominant projection peak."""
    if projection.size == 0:
        return None
    max_value = float(np.max(projection))
    if max_value <= 0.0:
        return None

    threshold = max_value * float(threshold_ratio)
    idx = np.where(projection >= threshold)[0]
    if idx.size == 0:
        return int(np.argmax(projection))

    weights = projection[idx].astype(np.float64)
    if float(weights.sum()) <= 0.0:
        return int(np.argmax(projection))
    return int(round(float(np.average(idx, weights=weights))))


def _max_consecutive_true(flags: np.ndarray) -> int:
    best = 0
    cur = 0
    for value in flags.astype(bool):
        if value:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def _image_direction_vectors() -> Dict[str, Tuple[int, int]]:
    return {
        "up": (0, -1),
        "down": (0, 1),
        "left": (-1, 0),
        "right": (1, 0),
    }


def _relative_direction_vectors(forward_dir: str) -> Dict[str, Tuple[int, int]]:
    image_vecs = _image_direction_vectors()
    if forward_dir not in image_vecs:
        raise ValueError(f"Invalid forward_dir_in_image: {forward_dir}")

    fx, fy = image_vecs[forward_dir]
    # Image y-axis points downward.  Robot-left in image coordinates is obtained
    # by rotating the forward vector counter-clockwise in the robot frame, which
    # becomes (fy, -fx) in image coordinates.
    left = (fy, -fx)
    right = (-left[0], -left[1])
    back = (-fx, -fy)
    return {
        "front": (fx, fy),
        "back": back,
        "left": left,
        "right": right,
    }


def _map_image_dirs_to_relative(active_image_dirs: Iterable[str], forward_dir: str) -> Tuple[str, ...]:
    image_vecs = _image_direction_vectors()
    rel_vecs = _relative_direction_vectors(forward_dir)
    result: List[str] = []
    for image_dir in active_image_dirs:
        v = image_vecs[image_dir]
        for rel_name, rel_v in rel_vecs.items():
            if v == rel_v:
                result.append(rel_name)
                break
    order = ["front", "back", "left", "right"]
    return tuple([d for d in order if d in result])


def _classify_active_dirs(active_dirs: Tuple[str, ...], dead_end_as_node: bool = False) -> Tuple[str, bool]:
    """Return (node_type, node_found).

    `node_found` is True only for decision nodes by default.  Straight lines are
    classified for debugging but do not trigger CENTER_NODE.
    """
    s = set(active_dirs)
    n = len(s)

    if n == 4:
        return "CROSS", True

    if n == 3:
        missing = [d for d in ["front", "back", "left", "right"] if d not in s][0]
        return f"T_MISSING_{missing.upper()}", True

    if n == 2:
        if s == {"front", "back"}:
            return "STRAIGHT_FRONT_BACK", False
        if s == {"left", "right"}:
            return "STRAIGHT_LEFT_RIGHT", False
        # Adjacent two arms => corner.  This covers ㄱ/ㄴ/ㄷ-like right-angle
        # appearances depending on the current camera orientation.
        ordered = [d for d in ["front", "back", "left", "right"] if d in s]
        return "CORNER_" + "_".join(d.upper() for d in ordered), True

    if n == 1:
        only = next(iter(s))
        return f"DEAD_END_{only.upper()}", bool(dead_end_as_node)

    return "NONE", False


def _arm_presence(mask: np.ndarray, cx: int, cy: int, direction: str, node_cfg: Dict) -> bool:
    """Check whether a line arm exists from node center toward one image side."""
    h, w = mask.shape[:2]
    band_ratio = float(node_cfg.get("arm_band_ratio", 0.075))
    min_arm_length_ratio = float(node_cfg.get("min_arm_length_ratio", 0.10))
    min_row_col_occupancy = float(node_cfg.get("min_row_col_occupancy", 0.18))
    center_gap_px = int(node_cfg.get("center_gap_px", 6))

    # Do not make the band too thin.  The generated line is not one pixel wide,
    # and Webots antialiasing/texture sampling can soften edges.
    half_band_x = max(4, int(round(w * band_ratio * 0.5)))
    half_band_y = max(4, int(round(h * band_ratio * 0.5)))

    if direction == "up":
        x1, x2 = max(0, cx - half_band_x), min(w, cx + half_band_x + 1)
        y1, y2 = 0, max(0, cy - center_gap_px)
        region = mask[y1:y2, x1:x2]
        if region.size == 0:
            return False
        row_counts = cv2.countNonZero(region) if region.ndim == 2 else 0
        # Use per-row occupancy for run length.
        per = (region > 0).sum(axis=1)
        threshold = max(2, int(region.shape[1] * min_row_col_occupancy))
        run = _max_consecutive_true(per >= threshold)
        return run >= int(h * min_arm_length_ratio) and row_counts >= int(h * min_arm_length_ratio)

    if direction == "down":
        x1, x2 = max(0, cx - half_band_x), min(w, cx + half_band_x + 1)
        y1, y2 = min(h, cy + center_gap_px), h
        region = mask[y1:y2, x1:x2]
        if region.size == 0:
            return False
        count = cv2.countNonZero(region)
        per = (region > 0).sum(axis=1)
        threshold = max(2, int(region.shape[1] * min_row_col_occupancy))
        run = _max_consecutive_true(per >= threshold)
        return run >= int(h * min_arm_length_ratio) and count >= int(h * min_arm_length_ratio)

    if direction == "left":
        x1, x2 = 0, max(0, cx - center_gap_px)
        y1, y2 = max(0, cy - half_band_y), min(h, cy + half_band_y + 1)
        region = mask[y1:y2, x1:x2]
        if region.size == 0:
            return False
        count = cv2.countNonZero(region)
        per = (region > 0).sum(axis=0)
        threshold = max(2, int(region.shape[0] * min_row_col_occupancy))
        run = _max_consecutive_true(per >= threshold)
        return run >= int(w * min_arm_length_ratio) and count >= int(w * min_arm_length_ratio)

    if direction == "right":
        x1, x2 = min(w, cx + center_gap_px), w
        y1, y2 = max(0, cy - half_band_y), min(h, cy + half_band_y + 1)
        region = mask[y1:y2, x1:x2]
        if region.size == 0:
            return False
        count = cv2.countNonZero(region)
        per = (region > 0).sum(axis=0)
        threshold = max(2, int(region.shape[0] * min_row_col_occupancy))
        run = _max_consecutive_true(per >= threshold)
        return run >= int(w * min_arm_length_ratio) and count >= int(w * min_arm_length_ratio)

    raise ValueError(f"Invalid image direction: {direction}")


def _detect_node(processed_mask: np.ndarray, cfg: Dict) -> Dict:
    """Detect and classify grid nodes: T, cross, and right-angle corners.

    The center estimate is the crossing of the vertical-axis peak and the
    horizontal-axis peak.  This is intentionally different from the centroid of
    black pixels, because a corner/T shape has an asymmetric pixel mass.
    """
    h, w = processed_mask.shape[:2]
    node_cfg = cfg.get("node_detection", cfg.get("intersection", {}))
    if not bool(node_cfg.get("enabled", True)):
        return {"found": False, "reason": "disabled", "type": "NONE"}

    x1, y1, x2, y2 = _make_roi(w, h, node_cfg, prefix="roi")
    roi_mask = processed_mask[y1:y2, x1:x2]
    area = int(cv2.countNonZero(roi_mask))
    min_area = int(node_cfg.get("min_area", 3500))
    if area < min_area:
        return {"found": False, "reason": f"area<{min_area}", "type": "NONE", "area": area}

    ys, xs = np.nonzero(roi_mask)
    if xs.size == 0 or ys.size == 0:
        return {"found": False, "reason": "no_pixels", "type": "NONE", "area": area}

    span_x = float((xs.max() - xs.min() + 1) / max(roi_mask.shape[1], 1))
    span_y = float((ys.max() - ys.min() + 1) / max(roi_mask.shape[0], 1))

    col_sum = roi_mask.sum(axis=0).astype(np.float64) / 255.0
    row_sum = roi_mask.sum(axis=1).astype(np.float64) / 255.0

    # Axis presence: a true vertical arm creates a strong column peak; a true
    # horizontal arm creates a strong row peak.  This avoids interpreting a plain
    # horizontal line as a corner just because its bounding box is wide.
    vertical_peak_ratio = float(np.max(col_sum) / max(roi_mask.shape[0], 1))
    horizontal_peak_ratio = float(np.max(row_sum) / max(roi_mask.shape[1], 1))
    min_axis_peak_ratio = float(node_cfg.get("min_axis_peak_ratio", 0.16))

    if vertical_peak_ratio < min_axis_peak_ratio or horizontal_peak_ratio < min_axis_peak_ratio:
        node_type = "STRAIGHT_OR_WEAK_NODE"
        return {
            "found": False,
            "reason": (
                f"axis_peak_fail v={vertical_peak_ratio:.2f} "
                f"h={horizontal_peak_ratio:.2f}"
            ),
            "type": node_type,
            "area": area,
            "span_x": span_x,
            "span_y": span_y,
        }

    peak_ratio = float(node_cfg.get("projection_threshold_ratio", 0.45))
    cx_local = _weighted_peak_index(col_sum, peak_ratio)
    cy_local = _weighted_peak_index(row_sum, peak_ratio)
    if cx_local is None or cy_local is None:
        return {"found": False, "reason": "peak_fail", "type": "NONE", "area": area, "span_x": span_x, "span_y": span_y}

    cx = int(x1 + cx_local)
    cy = int(y1 + cy_local)

    active_image_dirs = tuple(
        d for d in ["up", "down", "left", "right"]
        if _arm_presence(processed_mask, cx, cy, d, node_cfg)
    )
    forward_dir = cfg["camera"].get("forward_dir_in_image", "up")
    active_dirs = _map_image_dirs_to_relative(active_image_dirs, forward_dir)
    node_type, is_decision_node = _classify_active_dirs(
        active_dirs,
        dead_end_as_node=bool(node_cfg.get("dead_end_as_node", False)),
    )

    # Optional: require at least one front/back arm so side-only horizontal lines
    # do not accidentally stop the robot during early line following.
    if bool(node_cfg.get("require_longitudinal_arm", True)):
        if "front" not in active_dirs and "back" not in active_dirs:
            is_decision_node = False
            if node_type == "STRAIGHT_LEFT_RIGHT":
                reason = "side_only_straight"
            else:
                reason = "no_longitudinal_arm"
        else:
            reason = "ok" if is_decision_node else node_type.lower()
    else:
        reason = "ok" if is_decision_node else node_type.lower()

    lateral_error = _normalized_lateral_error(cx, cy, w, h, forward_dir)
    forward_error = _normalized_forward_error(cx, cy, w, h, forward_dir)

    return {
        "found": bool(is_decision_node),
        "reason": reason,
        "type": node_type,
        "center": (cx, cy),
        "area": area,
        "span_x": span_x,
        "span_y": span_y,
        "active_dirs": active_dirs,
        "active_image_dirs": active_image_dirs,
        "lateral_error": max(-1.0, min(1.0, float(lateral_error))),
        "forward_error": max(-1.0, min(1.0, float(forward_error))),
        "vertical_peak_ratio": vertical_peak_ratio,
        "horizontal_peak_ratio": horizontal_peak_ratio,
    }


def detect_line(bgr: np.ndarray, cfg: Dict) -> LineDetectionResult:
    """Detect the black line and node information."""
    if bgr is None:
        raise ValueError("Input image is None")

    height, width = bgr.shape[:2]
    line_cfg = cfg["line_detection"]
    x1, y1, x2, y2 = _make_roi(width, height, line_cfg, prefix="roi")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    threshold = int(line_cfg["black_threshold"])

    # Black pixels become white in mask.
    processed_mask = cv2.inRange(gray, 0, threshold)

    ksize = int(line_cfg.get("morph_kernel_size", 5))
    if ksize > 1:
        if ksize % 2 == 0:
            ksize += 1
        kernel = np.ones((ksize, ksize), dtype=np.uint8)
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel)
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel)

    node = _detect_node(processed_mask, cfg)

    roi_mask = processed_mask[y1:y2, x1:x2]
    mask_area = int(cv2.countNonZero(roi_mask))
    min_area = int(line_cfg["min_mask_area"])

    base_kwargs = {
        "roi": (x1, y1, x2, y2),
        "mask": processed_mask,
        "node_found": bool(node.get("found", False)),
        "node_type": str(node.get("type", "NONE")),
        "node_center": node.get("center"),
        "node_lateral_error": float(node.get("lateral_error", 0.0)),
        "node_forward_error": float(node.get("forward_error", 0.0)),
        "node_area": int(node.get("area", 0)),
        "node_span_x": float(node.get("span_x", 0.0)),
        "node_span_y": float(node.get("span_y", 0.0)),
        "node_reason": str(node.get("reason", "")),
        "active_dirs": tuple(node.get("active_dirs", ())),
        "active_image_dirs": tuple(node.get("active_image_dirs", ())),
        # Aliases for older debug/runtime code names.
        "intersection_found": bool(node.get("found", False)),
        "intersection_center": node.get("center"),
        "intersection_lateral_error": float(node.get("lateral_error", 0.0)),
        "intersection_forward_error": float(node.get("forward_error", 0.0)),
        "intersection_area": int(node.get("area", 0)),
        "intersection_span_x": float(node.get("span_x", 0.0)),
        "intersection_span_y": float(node.get("span_y", 0.0)),
        "intersection_reason": f"{node.get('type', 'NONE')}:{node.get('reason', '')}",
    }

    if mask_area < min_area:
        return LineDetectionResult(
            found=False,
            error=0.0,
            center=None,
            mask_area=mask_area,
            angle_deg=None,
            reason=f"mask_area<{min_area}",
            **base_kwargs,
        )

    moments = cv2.moments(roi_mask, binaryImage=True)
    if abs(moments["m00"]) < 1e-9:
        return LineDetectionResult(
            found=False,
            error=0.0,
            center=None,
            mask_area=mask_area,
            angle_deg=None,
            reason="zero_moment",
            **base_kwargs,
        )

    cx_roi = moments["m10"] / moments["m00"]
    cy_roi = moments["m01"] / moments["m00"]
    cx = int(round(x1 + cx_roi))
    cy = int(round(y1 + cy_roi))

    forward_dir = cfg["camera"]["forward_dir_in_image"]
    error = _normalized_lateral_error(cx, cy, width, height, forward_dir)
    error = max(-1.0, min(1.0, error))

    ys, xs = np.nonzero(roi_mask)
    if xs.size > 0:
        points = np.stack([xs + x1, ys + y1], axis=1)
        angle = _estimate_angle_deg(points)
    else:
        angle = None

    return LineDetectionResult(
        found=True,
        error=float(error),
        center=(cx, cy),
        mask_area=mask_area,
        angle_deg=angle,
        reason="ok",
        **base_kwargs,
    )
