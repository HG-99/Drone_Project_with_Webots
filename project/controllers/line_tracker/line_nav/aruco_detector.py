"""ArUco detection and marker-memory utilities for the line-tracker mission.

Mission meaning:
- detect floor ArUco markers with the down-facing camera
- remember each marker ID only once, in first-detection order
- bind each marker to the nearest/canonical topology node so it can be revisited
- expose the reverse-order target list for the marker-revisit phase
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from .config_loader import resolve_project_path


@dataclass
class ArucoObservation:
    marker_id: int
    corners: np.ndarray  # shape: (4, 2)
    center: Tuple[float, float]
    area_px: float


@dataclass
class ArucoDetectionResult:
    found: bool
    observations: List[ArucoObservation]
    reason: str = ""


@dataclass
class MarkerRecord:
    marker_id: int
    node_key: str
    x: float
    y: float
    yaw: float
    first_seen_frame: int
    last_seen_frame: int
    seen_count: int = 1


def _get_aruco_module():
    aruco = getattr(cv2, "aruco", None)
    if aruco is None:
        return None
    return aruco


def _get_dictionary(aruco, dictionary_name: str):
    dictionary_id = getattr(aruco, dictionary_name, None)
    if dictionary_id is None:
        raise ValueError(f"Unsupported ArUco dictionary: {dictionary_name}")
    return aruco.getPredefinedDictionary(dictionary_id)


def _make_detector(aruco, dictionary, aruco_cfg: Dict):
    # Compatible with both OpenCV 4.7+ and older opencv-contrib-python APIs.
    if hasattr(aruco, "DetectorParameters"):
        params = aruco.DetectorParameters()
    else:  # pragma: no cover - old API fallback
        params = aruco.DetectorParameters_create()

    # Optional tuning knobs. They are only set when the current OpenCV build has
    # the attribute, so this remains robust across Webots/OpenCV environments.
    for cfg_key, attr in [
        ("adaptive_thresh_win_size_min", "adaptiveThreshWinSizeMin"),
        ("adaptive_thresh_win_size_max", "adaptiveThreshWinSizeMax"),
        ("adaptive_thresh_win_size_step", "adaptiveThreshWinSizeStep"),
        ("min_marker_perimeter_rate", "minMarkerPerimeterRate"),
        ("max_marker_perimeter_rate", "maxMarkerPerimeterRate"),
        ("polygonal_approx_accuracy_rate", "polygonalApproxAccuracyRate"),
    ]:
        if cfg_key in aruco_cfg and hasattr(params, attr):
            setattr(params, attr, aruco_cfg[cfg_key])

    if hasattr(aruco, "ArucoDetector"):
        return aruco.ArucoDetector(dictionary, params), params
    return None, params


def _polygon_area(corners: np.ndarray) -> float:
    pts = corners.astype(np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def detect_aruco(bgr: np.ndarray, cfg: Dict) -> ArucoDetectionResult:
    """Detect ArUco markers in a BGR image.

    Returns an empty result when aruco.enabled is false or cv2.aruco is missing.
    """
    aruco_cfg = cfg.get("aruco", {}) or {}
    if not bool(aruco_cfg.get("enabled", False)):
        return ArucoDetectionResult(False, [], "disabled")

    aruco = _get_aruco_module()
    if aruco is None:
        return ArucoDetectionResult(False, [], "cv2.aruco_unavailable")

    dictionary_name = str(aruco_cfg.get("dictionary", "DICT_4X4_50"))
    dictionary = _get_dictionary(aruco, dictionary_name)
    detector, params = _make_detector(aruco, dictionary, aruco_cfg)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if detector is not None:
        corners_list, ids, _rejected = detector.detectMarkers(gray)
    else:  # pragma: no cover - old API fallback
        corners_list, ids, _rejected = aruco.detectMarkers(gray, dictionary, parameters=params)

    if ids is None or len(ids) == 0:
        return ArucoDetectionResult(False, [], "no_marker")

    min_area = float(aruco_cfg.get("min_area_px", 400.0))
    observations: List[ArucoObservation] = []
    for corners, marker_id_arr in zip(corners_list, ids.flatten()):
        pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
        area = _polygon_area(pts)
        if area < min_area:
            continue
        center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
        observations.append(
            ArucoObservation(
                marker_id=int(marker_id_arr),
                corners=pts,
                center=center,
                area_px=area,
            )
        )

    observations.sort(key=lambda o: o.marker_id)
    return ArucoDetectionResult(bool(observations), observations, "ok" if observations else "area_filtered")


def draw_aruco_observations(image: np.ndarray, result: ArucoDetectionResult) -> None:
    """Draw detected marker outlines and IDs on a debug image in-place."""
    if not result or not result.observations:
        return
    for obs in result.observations:
        pts = obs.corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [pts], True, (0, 180, 255), 2, lineType=cv2.LINE_AA)
        cx, cy = int(round(obs.center[0])), int(round(obs.center[1]))
        cv2.circle(image, (cx, cy), 4, (0, 180, 255), -1)
        cv2.putText(image, f"Aruco {obs.marker_id}", (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 2)


class ArucoMarkerMemory:
    """Stores first-seen ArUco markers and converts them into revisit targets."""

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.aruco_cfg = cfg.get("aruco", {}) or {}
        self.enabled = bool(self.aruco_cfg.get("enabled", False))
        self.expected_count = self._resolve_expected_count()
        self.records: Dict[int, MarkerRecord] = {}
        self.first_seen_order: List[int] = []
        self._last_printed_complete = False

    def _resolve_expected_count(self) -> int:
        value = self.aruco_cfg.get("expected_count", None)
        if value is not None:
            return int(value)

        # If omitted, inherit marker.count from world.yaml so the controller
        # knows when all generated markers have been seen.
        world_path = self.cfg.get("map", {}).get("world_config_path", "project/config/world.yaml")
        path = resolve_project_path(world_path)
        if yaml is None or not Path(path).exists():
            return 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                world = yaml.safe_load(f) or {}
            return int((world.get("marker", {}) or {}).get("count", 0))
        except Exception:
            return 0

    @property
    def found_count(self) -> int:
        return len(self.records)

    @property
    def complete(self) -> bool:
        return bool(self.enabled and self.expected_count > 0 and self.found_count >= self.expected_count)

    def update(self, result: ArucoDetectionResult, pose, planner, frame_idx: int, state: str = "") -> List[MarkerRecord]:
        """Update memory from current-frame detections.

        The marker is associated with the canonical map node nearest to the robot
        pose. Since markers are generated on grid vertices, this produces the
        same key used by the topology planner after node centering.
        """
        if not self.enabled or result is None or not result.observations:
            return []

        new_records: List[MarkerRecord] = []
        for obs in result.observations:
            if obs.marker_id in self.records:
                rec = self.records[obs.marker_id]
                rec.last_seen_frame = frame_idx
                rec.seen_count += 1
                # IMPORTANT:
                # planner.current_node_key is intentionally allowed to lag behind
                # the physical pose while the robot is approaching/leaving a node.
                # Therefore, only a post-registration state may overwrite the
                # marker-to-node binding.  This prevents the marker debug map from
                # showing the previous node first and then jumping to the correct
                # node later.
                if state == "CENTER_NODE_CONFIRMED" and getattr(planner, "current_node_key", ""):
                    rec.node_key = planner.current_node_key
                    rec.x = float(pose.x)
                    rec.y = float(pose.y)
                    rec.yaw = float(pose.yaw)
                continue

            # Before a node has been formally registered, current_node_key can be
            # stale from the previous centered node.  For new sightings, use the
            # pose-derived canonical grid key unless runtime explicitly tells us
            # that the node was just confirmed.
            if state == "CENTER_NODE_CONFIRMED" and getattr(planner, "current_node_key", ""):
                node_key = planner.current_node_key
            else:
                node_key = planner.map.key_from_xy(float(pose.x), float(pose.y))

            rec = MarkerRecord(
                marker_id=int(obs.marker_id),
                node_key=node_key,
                x=float(pose.x),
                y=float(pose.y),
                yaw=float(pose.yaw),
                first_seen_frame=int(frame_idx),
                last_seen_frame=int(frame_idx),
                seen_count=1,
            )
            self.records[obs.marker_id] = rec
            self.first_seen_order.append(obs.marker_id)
            new_records.append(rec)

        return new_records

    def reverse_targets(self) -> List[MarkerRecord]:
        """Return marker records in reverse first-detection order."""
        return [self.records[mid] for mid in reversed(self.first_seen_order) if mid in self.records]

    def status_text(self) -> str:
        if not self.enabled:
            return "Aruco OFF"
        if self.expected_count > 0:
            return f"Aruco {self.found_count}/{self.expected_count}"
        return f"Aruco {self.found_count}/?"
