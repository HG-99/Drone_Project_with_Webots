"""Lightweight topological map for a rectangular-grid line-tracking arena.

The controller already knows its Webots Supervisor pose. This module stores
centered decision points as graph nodes and stores traversed line segments as
edges. It is intentionally small and JSON-based so that it can be inspected
while debugging.

Planner v4 additions:
- classify outside/entry-road candidates so exploration does not leave the grid
- record failed edges when the robot loses the line while traversing an edge
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from .config_loader import resolve_project_path
from .robot_motion import Pose2D, normalize_angle


GLOBAL_DIRS = ("E", "N", "W", "S")
GLOBAL_DIR_TO_VEC = {
    "E": (1.0, 0.0),
    "N": (0.0, 1.0),
    "W": (-1.0, 0.0),
    "S": (0.0, -1.0),
}
GLOBAL_DIR_TO_YAW = {
    "E": 0.0,
    "N": math.pi / 2.0,
    "W": math.pi,
    "S": -math.pi / 2.0,
}


@dataclass
class MapNode:
    key: str
    x: float
    y: float
    visits: int = 0
    node_types: List[str] = field(default_factory=list)
    observed_dirs: List[str] = field(default_factory=list)


@dataclass
class MapEdge:
    key: str
    a: str
    b: str
    visits: int = 0
    last_direction: str = ""
    failed: bool = False
    failure_count: int = 0
    failure_reason: str = ""


def _edge_key(a: str, b: str) -> str:
    return "<->".join(sorted([a, b]))


def yaw_to_global_dir(yaw: float) -> str:
    """Snap a yaw angle to the nearest grid direction E/N/W/S."""
    y = normalize_angle(yaw)
    best = "E"
    best_err = 10.0
    for d, target in GLOBAL_DIR_TO_YAW.items():
        err = abs(normalize_angle(y - target))
        if err < best_err:
            best_err = err
            best = d
    return best


def relative_to_global_dir(current_yaw: float, relative_dir: str) -> str:
    heading_idx = GLOBAL_DIRS.index(yaw_to_global_dir(current_yaw))
    if relative_dir == "front":
        idx = heading_idx
    elif relative_dir == "left":
        idx = (heading_idx + 1) % 4
    elif relative_dir == "right":
        idx = (heading_idx - 1) % 4
    elif relative_dir == "back":
        idx = (heading_idx + 2) % 4
    else:
        raise ValueError(f"Unknown relative direction: {relative_dir}")
    return GLOBAL_DIRS[idx]


def global_to_yaw(global_dir: str) -> float:
    return GLOBAL_DIR_TO_YAW[global_dir]


def opposite_global_dir(global_dir: str) -> str:
    idx = GLOBAL_DIRS.index(global_dir)
    return GLOBAL_DIRS[(idx + 2) % 4]


class TopologyMap:
    """Graph map of visited decision nodes and traversed line segments."""

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        map_cfg = cfg.get("map", {})
        self.enabled = bool(map_cfg.get("enabled", True))
        self.node_merge_distance_m = float(map_cfg.get("node_merge_distance_m", 0.35))
        self.grid_snap_tolerance_m = float(map_cfg.get("grid_snap_tolerance_m", 0.35))
        self.coord_round_decimals = int(map_cfg.get("coord_round_decimals", 2))
        self.save_path = resolve_project_path(map_cfg.get("save_path", "project/debug_frames/topology_map.json"))

        self.grid_xs: List[float] = []
        self.grid_ys: List[float] = []
        self.grid_dx = float(map_cfg.get("default_grid_dx_m", 1.0))
        self.grid_dy = float(map_cfg.get("default_grid_dy_m", 1.0))
        self.entry_points: List[Tuple[float, float]] = []
        self.entry_start_xy: Tuple[float, float] | None = None
        self.entry_end_xy: Tuple[float, float] | None = None

        self.nodes: Dict[str, MapNode] = {}
        self.edges: Dict[str, MapEdge] = {}
        self._load_world_geometry()

    def _load_world_geometry(self) -> None:
        map_cfg = self.cfg.get("map", {})
        world_path = map_cfg.get("world_config_path", "project/config/world.yaml")
        path = resolve_project_path(world_path)
        if yaml is None or not path.exists():
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                world = yaml.safe_load(f) or {}
        except Exception as exc:  # pragma: no cover - defensive for Webots env
            print(f"[WARN] Failed to read world config for map snapping: {path}: {exc}")
            return

        grid = world.get("grid", {}) or {}
        try:
            w = float(grid.get("width_m"))
            h = float(grid.get("height_m"))
            cols = int(grid.get("cols"))
            rows = int(grid.get("rows"))
            self.grid_dx = w / max(cols, 1)
            self.grid_dy = h / max(rows, 1)
            self.grid_xs = [(-w / 2.0) + i * self.grid_dx for i in range(cols + 1)]
            self.grid_ys = [(-h / 2.0) + j * self.grid_dy for j in range(rows + 1)]
        except Exception:
            pass

        entry = world.get("entry_path", {}) or {}
        for key in ("start_xy", "end_xy"):
            if key in entry:
                try:
                    x, y = entry[key]
                    xy = (float(x), float(y))
                    self.entry_points.append(xy)
                    if key == "start_xy":
                        self.entry_start_xy = xy
                    elif key == "end_xy":
                        self.entry_end_xy = xy
                except Exception:
                    pass

    @property
    def grid_bounds(self) -> Tuple[float, float, float, float] | None:
        if not self.grid_xs or not self.grid_ys:
            return None
        return min(self.grid_xs), max(self.grid_xs), min(self.grid_ys), max(self.grid_ys)

    def _snap_value(self, value: float, candidates: Iterable[float]) -> float:
        candidates = list(candidates)
        if not candidates:
            return value
        nearest = min(candidates, key=lambda c: abs(float(value) - float(c)))
        if abs(float(value) - float(nearest)) <= self.grid_snap_tolerance_m:
            return float(nearest)
        return float(value)

    def _canonical_xy(self, x: float, y: float) -> Tuple[float, float]:
        # Entry-road endpoints are not always grid vertices, so snap to them first.
        for ex, ey in self.entry_points:
            if math.hypot(x - ex, y - ey) <= self.grid_snap_tolerance_m:
                return round(ex, self.coord_round_decimals), round(ey, self.coord_round_decimals)

        sx = self._snap_value(float(x), self.grid_xs)
        sy = self._snap_value(float(y), self.grid_ys)
        return round(sx, self.coord_round_decimals), round(sy, self.coord_round_decimals)

    def _find_existing_node(self, x: float, y: float) -> Optional[str]:
        for key, node in self.nodes.items():
            if math.hypot(node.x - x, node.y - y) <= self.node_merge_distance_m:
                return key
        return None

    def _node_key(self, x: float, y: float) -> str:
        return f"N:{x:.{self.coord_round_decimals}f},{y:.{self.coord_round_decimals}f}"

    def key_from_xy(self, x: float, y: float) -> str:
        """Return canonical node key for a world coordinate."""
        cx, cy = self._canonical_xy(float(x), float(y))
        return self._node_key(cx, cy)

    def entry_node_key(self) -> str:
        """Return the grid-side entry node key, if world.yaml provides entry_path.end_xy."""
        if self.entry_end_xy is None:
            return ""
        return self.key_from_xy(self.entry_end_xy[0], self.entry_end_xy[1])

    def is_entry_node_key(self, node_key: str) -> bool:
        entry_key = self.entry_node_key()
        return bool(entry_key and node_key == entry_key)

    def node_known(self, node_key: str) -> bool:
        return node_key in self.nodes

    def register_node(self, pose: Pose2D, node_type: str, observed_global_dirs: Iterable[str]) -> str:
        x, y = self._canonical_xy(pose.x, pose.y)
        existing = self._find_existing_node(x, y)
        key = existing if existing is not None else self._node_key(x, y)

        if key not in self.nodes:
            self.nodes[key] = MapNode(key=key, x=x, y=y)

        node = self.nodes[key]
        node.visits += 1
        if node_type and node_type not in node.node_types:
            node.node_types.append(node_type)
        for d in observed_global_dirs:
            if d not in node.observed_dirs:
                node.observed_dirs.append(d)
        node.observed_dirs = [d for d in GLOBAL_DIRS if d in set(node.observed_dirs)]
        return key

    def neighbor_key_from_direction(self, node_key: str, global_dir: str) -> str:
        node = self.nodes[node_key]
        vx, vy = GLOBAL_DIR_TO_VEC[global_dir]
        step_x = self.grid_dx if vx != 0 else 0.0
        step_y = self.grid_dy if vy != 0 else 0.0
        nx = node.x + vx * step_x
        ny = node.y + vy * step_y
        nx, ny = self._canonical_xy(nx, ny)
        return self._node_key(nx, ny)

    def add_edge(self, a: str, b: str, direction: str = "") -> None:
        if not a or not b or a == b:
            return
        key = _edge_key(a, b)
        if key not in self.edges:
            self.edges[key] = MapEdge(key=key, a=a, b=b)
        edge = self.edges[key]
        edge.visits += 1
        edge.last_direction = direction
        # A traversed edge is no longer considered failed.
        edge.failed = False

    def mark_planned_edge(self, a: str, b: str, direction: str = "") -> None:
        """Create an edge placeholder so repeated node detections do not re-pick it."""
        if not a or not b or a == b:
            return
        key = _edge_key(a, b)
        if key not in self.edges:
            self.edges[key] = MapEdge(key=key, a=a, b=b, visits=0, last_direction=direction)

    def mark_failed_edge(self, a: str, b: str, direction: str = "", reason: str = "") -> None:
        if not a or not b or a == b:
            return
        key = _edge_key(a, b)
        if key not in self.edges:
            self.edges[key] = MapEdge(key=key, a=a, b=b, visits=0, last_direction=direction)
        edge = self.edges[key]
        edge.failed = True
        edge.failure_count += 1
        edge.failure_reason = reason
        edge.last_direction = direction

    def edge_known(self, a: str, b: str) -> bool:
        return _edge_key(a, b) in self.edges

    def edge_visited(self, a: str, b: str) -> bool:
        edge = self.edges.get(_edge_key(a, b))
        return bool(edge and edge.visits > 0)

    def edge_failed(self, a: str, b: str) -> bool:
        edge = self.edges.get(_edge_key(a, b))
        return bool(edge and edge.failed)

    def node_is_outside_grid(self, node_key: str, margin: float = 1e-6) -> bool:
        node = self.nodes.get(node_key)
        if node is None or self.grid_bounds is None:
            return False
        xmin, xmax, ymin, ymax = self.grid_bounds
        return node.x < xmin - margin or node.x > xmax + margin or node.y < ymin - margin or node.y > ymax + margin

    def xy_is_outside_grid(self, x: float, y: float, margin: float = 1e-6) -> bool:
        if self.grid_bounds is None:
            return False
        xmin, xmax, ymin, ymax = self.grid_bounds
        return x < xmin - margin or x > xmax + margin or y < ymin - margin or y > ymax + margin

    def candidate_goes_outside_grid(self, node_key: str, global_dir: str) -> bool:
        node = self.nodes[node_key]
        vx, vy = GLOBAL_DIR_TO_VEC[global_dir]
        nx = node.x + vx * (self.grid_dx if vx != 0 else 0.0)
        ny = node.y + vy * (self.grid_dy if vy != 0 else 0.0)
        return self.xy_is_outside_grid(nx, ny, margin=self.grid_snap_tolerance_m * 0.2)

    def node_on_boundary(self, node_key: str) -> bool:
        node = self.nodes.get(node_key)
        if node is None or self.grid_bounds is None:
            return False
        xmin, xmax, ymin, ymax = self.grid_bounds
        tol = self.grid_snap_tolerance_m
        return (
            abs(node.x - xmin) <= tol
            or abs(node.x - xmax) <= tol
            or abs(node.y - ymin) <= tol
            or abs(node.y - ymax) <= tol
        )

    def candidate_goes_inward_from_boundary(self, node_key: str, global_dir: str) -> bool:
        """True if a candidate moves from a boundary node toward grid interior."""
        if not self.node_on_boundary(node_key) or self.grid_bounds is None:
            return False
        node = self.nodes[node_key]
        vx, vy = GLOBAL_DIR_TO_VEC[global_dir]
        nx = node.x + vx * (self.grid_dx if vx != 0 else 0.0)
        ny = node.y + vy * (self.grid_dy if vy != 0 else 0.0)
        xmin, xmax, ymin, ymax = self.grid_bounds
        tol = self.grid_snap_tolerance_m
        inside = (xmin - tol <= nx <= xmax + tol) and (ymin - tol <= ny <= ymax + tol)
        if not inside:
            return False
        # If the current node is on a boundary and the neighbor is not on the same boundary,
        # this is an inward candidate.
        moves_off_left = abs(node.x - xmin) <= tol and nx > node.x + tol * 0.2
        moves_off_right = abs(node.x - xmax) <= tol and nx < node.x - tol * 0.2
        moves_off_bottom = abs(node.y - ymin) <= tol and ny > node.y + tol * 0.2
        moves_off_top = abs(node.y - ymax) <= tol and ny < node.y - tol * 0.2
        return moves_off_left or moves_off_right or moves_off_bottom or moves_off_top

    def save(self) -> None:
        if not self.enabled:
            return
        try:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "nodes": {k: asdict(v) for k, v in self.nodes.items()},
                "edges": {k: asdict(v) for k, v in self.edges.items()},
                "grid": {
                    "xs": self.grid_xs,
                    "ys": self.grid_ys,
                    "dx": self.grid_dx,
                    "dy": self.grid_dy,
                },
            }
            with open(self.save_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] Failed to save topology map: {self.save_path}: {exc}")
