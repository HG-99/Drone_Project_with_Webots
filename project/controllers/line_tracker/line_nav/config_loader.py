"""Configuration loading utilities for the line tracker controller."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError as exc:  # pragma: no cover - depends on Webots Python env
    raise ImportError(
        "PyYAML is required. Install it in the Python environment used by Webots: "
        "pip install pyyaml"
    ) from exc


DEFAULT_CONFIG: Dict[str, Any] = {
    "camera": {
        "down_camera_name": "down_camera",
        "image_rotation": "none",  # none | rot90_cw | rot90_ccw | rot180
        "forward_dir_in_image": "up",  # up | down | left | right
    },
    "line_detection": {
        "black_threshold": 80,
        "min_mask_area": 300,
        "roi_y_min_ratio": 0.12,
        "roi_y_max_ratio": 0.88,
        "roi_x_min_ratio": 0.18,
        "roi_x_max_ratio": 0.82,
        "morph_kernel_size": 5,
    },
    "node_detection": {
        "enabled": True,
        # Use a wide ROI because T/cross/corner branches can be outside the
        # narrow line-following ROI.
        "roi_y_min_ratio": 0.02,
        "roi_y_max_ratio": 0.98,
        "roi_x_min_ratio": 0.02,
        "roi_x_max_ratio": 0.98,
        "min_area": 3500,
        "projection_threshold_ratio": 0.45,
        # A vertical axis must create a strong column peak and a horizontal axis
        # must create a strong row peak.  This prevents a plain straight line
        # from being treated as a node.
        "min_axis_peak_ratio": 0.16,
        # Arm classification around the geometric center.
        "arm_band_ratio": 0.075,
        "min_arm_length_ratio": 0.10,
        "min_row_col_occupancy": 0.18,
        "center_gap_px": 6,
        "dead_end_as_node": False,
        "require_longitudinal_arm": True,
        # Centering controller. Positive forward_error means move forward.
        "kp_forward": 0.35,
        "max_forward_speed_mps": 0.16,
        "kp_yaw": 1.0,
        "max_yaw_rate": 0.45,
        "yaw_sign": None,  # None => use control.yaw_sign
        "center_lateral_tolerance": 0.055,
        "center_forward_tolerance": 0.055,
        "hold_center_frames": 6,
        "stop_when_centered": True,
    },
    # Kept for backward compatibility with older YAML files.  If a user still
    # has `intersection:` in YAML, load_config() maps it into node_detection.
    "intersection": {},
    "planner": {
        "enabled": True,
        "mode": "true_dfs_stack",
        "use_true_dfs_stack": True,
        "decision_priority": ["front", "left", "right", "back"],
        "allow_backtracking": True,
        "allow_known_edge_reuse": False,
        "mark_attempted_edges": True,
        "node_rearm_distance_m": 0.65,
        "turn_kp_yaw": 2.2,
        "turn_max_yaw_rate": 0.85,
        "turn_yaw_tolerance_rad": 0.04,
        "leave_speed_mps": 0.18,
        "leave_distance_m": 0.45,
        "leave_snap_to_node_center": True,
        "avoid_outside_edges": True,
        "allow_outside_edges": False,
        "prefer_inward_from_boundary": True,
        "avoid_revisited_nodes": True,
        "avoid_entry_node_reentry": True,
    },
    "map": {
        "enabled": True,
        "world_config_path": "project/config/world.yaml",
        "save_path": "project/debug_frames/topology_map.json",
        "node_merge_distance_m": 0.35,
        "grid_snap_tolerance_m": 0.35,
        "coord_round_decimals": 2,
        "default_grid_dx_m": 1.0,
        "default_grid_dy_m": 1.0,
    },
    "control": {
        "forward_speed_mps": 0.25,
        "kp_yaw": 1.2,
        "max_yaw_rate": 0.6,
        "yaw_sign": -1.0,
        "fixed_z": 1.0,
    },
    "lost_line": {
        "stop_when_lost": True,
        "search_when_lost": False,
        "search_yaw_rate": 0.25,
    },
    "recovery": {
        "backtrack_on_line_lost": True,
        "backtrack_speed_mps": 0.20,
        "backtrack_max_yaw_rate": 1.2,
        "backtrack_position_tolerance_m": 0.035,
        "backtrack_yaw_tolerance_rad": 0.05,
    },
    "return_home": {
        "enabled": True,
        "target": "entry_start",  # entry_start | initial_pose
        "speed_mps": 0.20,
        "max_yaw_rate": 1.2,
        "position_tolerance_m": 0.035,
        "yaw_tolerance_rad": 0.05,
    },

    "aruco": {
        "enabled": True,
        "dictionary": "DICT_4X4_50",
        # None means: read marker.count from map.world_config_path.
        "expected_count": None,
        "min_area_px": 400.0,
        "trigger_revisit_when_all_found": True,
        "draw_debug": True,
    },
    "marker_revisit": {
        "enabled": True,
        "unique_nodes_only": True,
        "skip_duplicate_target_nodes": False,
    },
    "marker_return": {
        # Phase 2 planner used after every ArUco marker has been found.
        # This planner does not reuse DFS stack order or visited topology edges.
        "enabled": True,
        "planner": "predefined_grid_shortest_path",
        "use_predefined_grid": True,
        "use_visited_topology_edges": False,
        "ordinary_nodes_reusable": True,
        "forbid_all_non_target_marker_nodes": True,
        "route_to_entry_after_markers": True,
        "direction_priority": ["E", "N", "W", "S"],
    },
    "debug": {
        "enabled": True,
        "window_scale": 0.7,
        "show_mask": True,
        "save_frames": False,
        "save_dir": "project/debug_frames",
        "pseudo_odometry_map": {
            "enabled": True,
            "panel_size_px": 720,
            "padding_px": 36,
            "path_sample_distance_m": 0.08,
            "sample_node_radius_px": 2,
            "show_node_labels": True,
            "max_node_labels": 30,
            "save_panel": False,
            "save_panel_name": "pseudo_odometry_map_latest.png",
        },
        "marker_node_map": {
            "enabled": True,
            "draw_reverse_edges": True,
            "show_labels": True,
            "save_panel": False,
            "save_panel_name": "aruco_marker_node_map_latest.png",
        },
    },
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base and return a new dictionary."""
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def find_project_root(start: Optional[Path] = None) -> Path:
    """Find the `project` directory from this file or current working directory."""
    if start is None:
        start = Path(__file__).resolve()

    candidates = [start] + list(start.parents)
    for path in candidates:
        p = path if path.is_dir() else path.parent
        if (p / "config").is_dir() and (p / "controllers").is_dir():
            return p

    cwd = Path.cwd().resolve()
    for path in [cwd] + list(cwd.parents):
        if (path / "config").is_dir() and (path / "controllers").is_dir():
            return path

    return cwd


def default_config_path() -> Path:
    return find_project_root() / "config" / "line_tracker.yaml"


def _normalize_legacy_sections(cfg: Dict[str, Any], user_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Map old `intersection:` settings into `node_detection:` if needed."""
    if user_cfg and "intersection" in user_cfg:
        # Old config files used intersection.  New code uses node_detection.
        # Values explicitly placed in node_detection take priority.
        legacy = cfg.get("intersection", {}) or {}
        current = cfg.get("node_detection", {}) or {}
        cfg["node_detection"] = deep_merge(current, legacy)
    return cfg


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load YAML config and merge it with DEFAULT_CONFIG."""
    path = Path(config_path).expanduser() if config_path else default_config_path()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()

    user_cfg = {}
    if not path.exists():
        print(f"[WARN] Config not found: {path}")
        print("[WARN] Falling back to built-in DEFAULT_CONFIG.")
        cfg = copy.deepcopy(DEFAULT_CONFIG)
    else:
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg = deep_merge(DEFAULT_CONFIG, user_cfg)
        cfg = _normalize_legacy_sections(cfg, user_cfg)

    cfg["_meta"] = {
        "config_path": str(path),
        "project_root": str(find_project_root()),
    }
    validate_config(cfg)
    return cfg


def validate_config(cfg: Dict[str, Any]) -> None:
    """Validate the minimum fields used by the controller."""
    cam = cfg["camera"]
    if cam["image_rotation"] not in {"none", "rot90_cw", "rot90_ccw", "rot180"}:
        raise ValueError(f"Invalid image_rotation: {cam['image_rotation']}")
    if cam["forward_dir_in_image"] not in {"up", "down", "left", "right"}:
        raise ValueError(f"Invalid forward_dir_in_image: {cam['forward_dir_in_image']}")

    for section_name in ["line_detection", "node_detection"]:
        sec = cfg.get(section_name, {})
        for key in ["roi_y_min_ratio", "roi_y_max_ratio", "roi_x_min_ratio", "roi_x_max_ratio"]:
            if key not in sec:
                continue
            v = float(sec[key])
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{section_name}.{key} must be in [0, 1], got {v}")
        if sec.get("roi_y_min_ratio", 0.0) >= sec.get("roi_y_max_ratio", 1.0):
            raise ValueError(f"{section_name}: roi_y_min_ratio must be smaller than roi_y_max_ratio")
        if sec.get("roi_x_min_ratio", 0.0) >= sec.get("roi_x_max_ratio", 1.0):
            raise ValueError(f"{section_name}: roi_x_min_ratio must be smaller than roi_x_max_ratio")

    ctrl = cfg["control"]
    if float(ctrl["forward_speed_mps"]) < 0:
        raise ValueError("forward_speed_mps must be non-negative")
    if float(ctrl["max_yaw_rate"]) <= 0:
        raise ValueError("max_yaw_rate must be positive")


def resolve_project_path(path_like: str) -> Path:
    """Resolve paths relative to the project root."""
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p
    project_root = find_project_root()
    if str(p).startswith("project/"):
        return project_root / p.relative_to("project")
    return project_root / p
