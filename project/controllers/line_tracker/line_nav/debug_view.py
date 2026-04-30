"""OpenCV debug visualization for line tracking and pseudo-odometry mapping.

Dashboard layout:
- top-left: down-camera image with line/node/ArUco overlays
- top-right: down-camera binary mask and node overlay
- bottom-left: full topology map with all nodes/edges
- bottom-right: ArUco marker-node map showing only marker nodes and reverse order

The maps are not used for direct low-level control; they are for visual debugging.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .config_loader import resolve_project_path


class DebugView:
    def __init__(self, cfg):
        self.cfg = cfg
        debug_cfg = cfg["debug"]
        self.enabled = bool(debug_cfg.get("enabled", True))
        self.scale = float(debug_cfg.get("window_scale", 0.7))
        self.show_mask = bool(debug_cfg.get("show_mask", True))
        self.save_frames = bool(debug_cfg.get("save_frames", False))
        self.frame_idx = 0
        self.window_name = "line_tracker_debug"

        save_dir = debug_cfg.get("save_dir", "project/debug_frames")
        self.save_dir = resolve_project_path(save_dir)
        if self.save_frames:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # Pseudo-odometry debug map settings.
        odom_cfg = debug_cfg.get("pseudo_odometry_map", {})
        self.odom_enabled = bool(odom_cfg.get("enabled", True))
        self.odom_size = int(odom_cfg.get("panel_size_px", 720))
        self.odom_padding = int(odom_cfg.get("padding_px", 36))
        self.odom_path_sample_dist = float(odom_cfg.get("path_sample_distance_m", 0.08))
        self.odom_node_draw_radius = int(odom_cfg.get("sample_node_radius_px", 2))
        self.odom_show_labels = bool(odom_cfg.get("show_node_labels", True))
        self.odom_max_labels = int(odom_cfg.get("max_node_labels", 30))
        self.odom_save_panel = bool(odom_cfg.get("save_panel", False))
        self.odom_save_name = str(odom_cfg.get("save_panel_name", "pseudo_odometry_map_latest.png"))

        marker_map_cfg = debug_cfg.get("marker_node_map", {}) or {}
        self.marker_map_enabled = bool(marker_map_cfg.get("enabled", True))
        self.marker_map_draw_reverse_edges = bool(marker_map_cfg.get("draw_reverse_edges", True))
        self.marker_map_show_labels = bool(marker_map_cfg.get("show_labels", True))
        self.marker_map_save_panel = bool(marker_map_cfg.get("save_panel", False))
        self.marker_map_save_name = str(marker_map_cfg.get("save_panel_name", "aruco_marker_node_map_latest.png"))

        self.pose_history: list[tuple[float, float]] = []
        self.pose_history_phase: list[str] = []
        self.pose_nodes: list[tuple[float, float]] = []
        self.pose_nodes_phase: list[str] = []
        self._last_history_pose: tuple[float, float] | None = None
        self._last_node_pose: tuple[float, float] | None = None
        # Latched once Phase 2 starts. This lets the full map draw the path used
        # after all ArUco markers are found in a separate color even while the
        # state machine is TURN_TO_DIRECTION / LEAVE_NODE / RETURN_HOME_*.
        self._marker_return_phase_latched = False

    def _draw_forward_arrow(self, image: np.ndarray) -> None:
        h, w = image.shape[:2]
        direction = self.cfg["camera"].get("forward_dir_in_image", "up")
        center = (w // 2, h // 2)
        length = int(min(w, h) * 0.18)

        if direction == "up":
            end = (center[0], center[1] - length)
        elif direction == "down":
            end = (center[0], center[1] + length)
        elif direction == "left":
            end = (center[0] - length, center[1])
        else:  # right
            end = (center[0] + length, center[1])

        cv2.arrowedLine(image, center, end, (0, 255, 255), 2, tipLength=0.25)
        cv2.putText(image, "front", (end[0] + 5, end[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    def _draw_node(self, image: np.ndarray, detection) -> None:
        h, w = image.shape[:2]
        if detection.node_center is None:
            return

        nx, ny = detection.node_center
        color = (255, 0, 255) if detection.node_found else (160, 160, 160)
        cv2.circle(image, (nx, ny), 10, color, 2)
        cv2.drawMarker(image, (nx, ny), color, cv2.MARKER_CROSS, 26, 2)
        cv2.line(image, (w // 2, h // 2), (nx, ny), color, 2)

        label = detection.node_type
        if detection.active_dirs:
            label += " " + ",".join(detection.active_dirs)
        cv2.putText(image, label, (nx + 10, ny - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    def _is_marker_return_phase(self, state: str, planner_info=None, revisit_mission=None) -> bool:
        """Return True after Phase 2 starts.

        Phase 2 means: all ArUco markers have been found and the controller is
        now using the marker-return/grid-return planner instead of the discovery
        DFS planner.  This value is latched so that TURN/LEAVE/HOME states also
        stay colored as return-path motion in the full map.
        """
        reason = ""
        if planner_info:
            reason = str(planner_info.get("reason", ""))

        started = bool(getattr(revisit_mission, "started", False)) if revisit_mission is not None else False
        if started or "marker_return" in reason or "marker_revisit" in reason:
            self._marker_return_phase_latched = True

        if str(state).startswith("RETURN_HOME") and self._marker_return_phase_latched:
            return True
        return bool(self._marker_return_phase_latched)

    def _append_pose_history(self, pose, marker_return_phase: bool = False) -> None:
        p = (float(pose.x), float(pose.y))
        phase = "marker_return" if marker_return_phase else "explore"
        if self._last_history_pose is None:
            self.pose_history.append(p)
            self.pose_history_phase.append(phase)
            self.pose_nodes.append(p)
            self.pose_nodes_phase.append(phase)
            self._last_history_pose = p
            self._last_node_pose = p
            return

        dx = p[0] - self._last_history_pose[0]
        dy = p[1] - self._last_history_pose[1]
        if math.hypot(dx, dy) >= 0.01:
            self.pose_history.append(p)
            self.pose_history_phase.append(phase)
            self._last_history_pose = p

        if self._last_node_pose is None:
            self.pose_nodes.append(p)
            self.pose_nodes_phase.append(phase)
            self._last_node_pose = p
        else:
            ndx = p[0] - self._last_node_pose[0]
            ndy = p[1] - self._last_node_pose[1]
            if math.hypot(ndx, ndy) >= self.odom_path_sample_dist:
                self.pose_nodes.append(p)
                self.pose_nodes_phase.append(phase)
                self._last_node_pose = p

    def _collect_world_bounds(self, planner=None):
        xs = []
        ys = []

        for x, y in self.pose_history:
            xs.append(float(x))
            ys.append(float(y))
        for x, y in self.pose_nodes:
            xs.append(float(x))
            ys.append(float(y))

        topo = getattr(planner, "map", None) if planner is not None else None
        if topo is not None:
            for x in getattr(topo, "grid_xs", []):
                xs.append(float(x))
            for y in getattr(topo, "grid_ys", []):
                ys.append(float(y))
            for ex, ey in getattr(topo, "entry_points", []):
                xs.append(float(ex))
                ys.append(float(ey))
            for node in getattr(topo, "nodes", {}).values():
                xs.append(float(node.x))
                ys.append(float(node.y))

        if not xs or not ys:
            xs = [-5.0, 5.0]
            ys = [-5.0, 5.0]

        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)

        # keep square-ish and add margins
        span_x = max(0.1, max_x - min_x)
        span_y = max(0.1, max_y - min_y)
        span = max(span_x, span_y)
        cx = 0.5 * (min_x + max_x)
        cy = 0.5 * (min_y + max_y)
        half = 0.5 * span + 0.5
        return cx - half, cx + half, cy - half, cy + half

    def _world_to_canvas(self, x, y, bounds, size, padding):
        min_x, max_x, min_y, max_y = bounds
        inner_w = max(1, size - 2 * padding)
        inner_h = max(1, size - 2 * padding)

        nx = (float(x) - min_x) / max(max_x - min_x, 1e-6)
        ny = (float(y) - min_y) / max(max_y - min_y, 1e-6)
        px = int(round(padding + nx * inner_w))
        py = int(round(size - (padding + ny * inner_h)))
        return px, py

    def _draw_grid(self, canvas, planner, bounds):
        topo = getattr(planner, "map", None) if planner is not None else None
        if topo is None:
            return

        size = canvas.shape[0]
        pad = self.odom_padding
        for gx in getattr(topo, "grid_xs", []):
            x0, y0 = self._world_to_canvas(gx, bounds[2], bounds, size, pad)
            x1, y1 = self._world_to_canvas(gx, bounds[3], bounds, size, pad)
            cv2.line(canvas, (x0, y0), (x1, y1), (230, 230, 230), 1)
        for gy in getattr(topo, "grid_ys", []):
            x0, y0 = self._world_to_canvas(bounds[0], gy, bounds, size, pad)
            x1, y1 = self._world_to_canvas(bounds[1], gy, bounds, size, pad)
            cv2.line(canvas, (x0, y0), (x1, y1), (230, 230, 230), 1)

    def _make_pseudo_odom_map(self, pose, state, planner=None, planner_info=None, revisit_mission=None):
        marker_return_phase = self._is_marker_return_phase(state, planner_info=planner_info, revisit_mission=revisit_mission)
        self._append_pose_history(pose, marker_return_phase=marker_return_phase)

        size = self.odom_size
        canvas = np.full((size, size, 3), 250, dtype=np.uint8)
        bounds = self._collect_world_bounds(planner)
        pad = self.odom_padding

        # Title and axes box
        cv2.rectangle(canvas, (pad // 2, pad // 2), (size - pad // 2, size - pad // 2), (180, 180, 180), 1)
        cv2.putText(canvas, "Pseudo Odometry Map", (18, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2)

        self._draw_grid(canvas, planner, bounds)

        # Entry path / planned topology edges first.
        topo = getattr(planner, "map", None) if planner is not None else None
        if topo is not None:
            for edge in topo.edges.values():
                na = topo.nodes.get(edge.a)
                nb = topo.nodes.get(edge.b)
                if na is None or nb is None:
                    continue
                a = self._world_to_canvas(na.x, na.y, bounds, size, pad)
                b = self._world_to_canvas(nb.x, nb.y, bounds, size, pad)
                if getattr(edge, "failed", False):
                    color = (40, 40, 220)  # red-ish: failed/blocked edge
                    thickness = 2
                elif edge.visits > 0:
                    color = (180, 140, 220)
                    thickness = 3
                else:
                    color = (205, 205, 205)
                    thickness = 1
                cv2.line(canvas, a, b, color, thickness)

        # Actual traversed path.
        # Exploration path is gray; after all ArUco markers are found, the
        # marker-return / home-return path is drawn in red for quick debugging.
        if len(self.pose_history) >= 2:
            for i in range(1, len(self.pose_history)):
                p0 = self._world_to_canvas(*self.pose_history[i - 1], bounds, size, pad)
                p1 = self._world_to_canvas(*self.pose_history[i], bounds, size, pad)
                phase0 = self.pose_history_phase[i - 1] if i - 1 < len(self.pose_history_phase) else "explore"
                phase1 = self.pose_history_phase[i] if i < len(self.pose_history_phase) else "explore"
                is_return_segment = phase0 == "marker_return" or phase1 == "marker_return"
                color = (0, 0, 255) if is_return_segment else (70, 70, 70)
                thickness = 3 if is_return_segment else 2
                cv2.line(canvas, p0, p1, color, thickness, lineType=cv2.LINE_AA)

        # Sampled pseudo-odometry nodes.
        for i, (x, y) in enumerate(self.pose_nodes):
            p = self._world_to_canvas(x, y, bounds, size, pad)
            phase = self.pose_nodes_phase[i] if i < len(self.pose_nodes_phase) else "explore"
            color = (0, 0, 255) if phase == "marker_return" else (60, 170, 60)
            cv2.circle(canvas, p, self.odom_node_draw_radius, color, -1)

        # Planner topology nodes.
        label_budget = self.odom_max_labels
        if topo is not None:
            current_node_key = getattr(planner, "current_node_key", "")
            for key, node in topo.nodes.items():
                p = self._world_to_canvas(node.x, node.y, bounds, size, pad)
                is_current = key == current_node_key
                radius = 7 if is_current else 5
                fill = (0, 0, 255) if is_current else (255, 0, 255)
                cv2.circle(canvas, p, radius, fill, -1)
                cv2.circle(canvas, p, radius + 2, (255, 255, 255), 1)
                if self.odom_show_labels and label_budget > 0:
                    label = key.replace("N:", "")
                    cv2.putText(canvas, label, (p[0] + 8, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (40, 40, 40), 1)
                    label_budget -= 1

        # Current robot pose.
        current = self._world_to_canvas(pose.x, pose.y, bounds, size, pad)
        arrow_len_m = 0.35
        hx = float(pose.x) + arrow_len_m * math.cos(float(pose.yaw))
        hy = float(pose.y) + arrow_len_m * math.sin(float(pose.yaw))
        head = self._world_to_canvas(hx, hy, bounds, size, pad)
        cv2.circle(canvas, current, 6, (0, 0, 200), -1)
        cv2.arrowedLine(canvas, current, head, (0, 120, 255), 2, tipLength=0.28)

        # Optional target/next node.
        if planner_info and topo is not None:
            next_key = planner_info.get("next", "")
            if next_key and next_key in topo.nodes:
                node = topo.nodes[next_key]
                p = self._world_to_canvas(node.x, node.y, bounds, size, pad)
                cv2.circle(canvas, p, 9, (255, 165, 0), 2)
                cv2.putText(canvas, "NEXT", (p[0] + 10, p[1] + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 165, 0), 2)

        # Legend / info.
        legend = [
            "gray line: exploration path",
            "red line: marker-return/home path",
            "green/red dots: sampled odom nodes",
            "magenta nodes: planner/topology nodes",
            "red edges: failed/blocked edges",
            "orange ring: next target node",
            "red dot + arrow: current pose",
            f"state: {state}",
            f"pose: ({pose.x:+.2f}, {pose.y:+.2f}, {pose.yaw:+.2f})",
        ]
        if planner_info:
            legend.append(
                f"plan: {planner_info.get('decision', 'none')}/{planner_info.get('global', '')}"
            )
        y0 = size - 18 - 18 * len(legend)
        y = max(40, y0)
        for text in legend:
            cv2.putText(canvas, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (25, 25, 25), 1)
            y += 18

        # Save separate latest map panel if requested.
        if self.odom_save_panel:
            out_path = Path(self.save_dir) / self.odom_save_name
            try:
                cv2.imwrite(str(out_path), canvas)
            except Exception:
                pass

        return canvas

    def _resize_panel(self, image: np.ndarray, width: int, height: int) -> np.ndarray:
        """Resize a panel to the common 2x2 dashboard cell size."""
        if image.shape[1] == width and image.shape[0] == height:
            return image
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    def _draw_panel_title(self, image: np.ndarray, title: str, subtitle: str = "") -> None:
        """Draw a dark title band at the top of a debug panel."""
        cv2.rectangle(image, (0, 0), (image.shape[1], 34), (20, 20, 20), -1)
        cv2.putText(image, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        if subtitle:
            cv2.putText(image, subtitle, (260, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1)

    def _marker_records_in_first_seen_order(self, aruco_memory):
        if aruco_memory is None:
            return []
        records = getattr(aruco_memory, "records", {}) or {}
        order = list(getattr(aruco_memory, "first_seen_order", []) or [])
        out = []
        for marker_id in order:
            rec = records.get(marker_id)
            if rec is not None:
                out.append(rec)
        if not out and records:
            out = sorted(records.values(), key=lambda r: getattr(r, "first_seen_frame", 0))
        return out

    def _marker_records_in_reverse_order(self, aruco_memory, revisit_mission=None):
        # Once the mission starts, use its actual target list because it may have
        # unique-node filtering applied. Before that, show raw reverse discovery order.
        if revisit_mission is not None and getattr(revisit_mission, "targets", None):
            return list(revisit_mission.targets)
        if aruco_memory is not None and hasattr(aruco_memory, "reverse_targets"):
            return list(aruco_memory.reverse_targets())
        return list(reversed(self._marker_records_in_first_seen_order(aruco_memory)))

    def _marker_node_xy(self, rec, planner=None):
        key = getattr(rec, "node_key", "")
        topo = getattr(planner, "map", None) if planner is not None else None
        if topo is not None and key in getattr(topo, "nodes", {}):
            node = topo.nodes[key]
            return float(node.x), float(node.y)
        return float(getattr(rec, "x", 0.0)), float(getattr(rec, "y", 0.0))

    def _draw_current_pose_on_map(self, canvas, pose, bounds, size, pad) -> None:
        current = self._world_to_canvas(pose.x, pose.y, bounds, size, pad)
        arrow_len_m = 0.35
        hx = float(pose.x) + arrow_len_m * math.cos(float(pose.yaw))
        hy = float(pose.y) + arrow_len_m * math.sin(float(pose.yaw))
        head = self._world_to_canvas(hx, hy, bounds, size, pad)
        cv2.circle(canvas, current, 6, (0, 0, 200), -1)
        cv2.arrowedLine(canvas, current, head, (0, 120, 255), 2, tipLength=0.28)

    def _make_marker_node_map(self, pose, state, planner=None, planner_info=None, aruco_memory=None, revisit_mission=None):
        """Map panel that displays only ArUco marker nodes.

        Ordinary topology nodes are intentionally hidden here. The purpose is
        to visually separate the marker target sequence from the full DFS map.
        """
        size = self.odom_size
        canvas = np.full((size, size, 3), 252, dtype=np.uint8)
        bounds = self._collect_world_bounds(planner)
        pad = self.odom_padding

        cv2.rectangle(canvas, (pad // 2, pad // 2), (size - pad // 2, size - pad // 2), (180, 180, 180), 1)
        self._draw_panel_title(canvas, "ArUco Marker Node Map", "marker nodes only")
        self._draw_grid(canvas, planner, bounds)

        first_records = self._marker_records_in_first_seen_order(aruco_memory)
        reverse_records = self._marker_records_in_reverse_order(aruco_memory, revisit_mission=revisit_mission)

        if self.marker_map_draw_reverse_edges and len(reverse_records) >= 2:
            pts = []
            last_key = None
            for rec in reverse_records:
                key = getattr(rec, "node_key", "")
                if not key or key == last_key:
                    continue
                x, y = self._marker_node_xy(rec, planner=planner)
                pts.append(self._world_to_canvas(x, y, bounds, size, pad))
                last_key = key
            for a, b in zip(pts, pts[1:]):
                cv2.arrowedLine(canvas, a, b, (255, 120, 0), 2, tipLength=0.18)

        by_node = defaultdict(list)
        for rec in first_records:
            key = getattr(rec, "node_key", "")
            if key:
                by_node[key].append(rec)

        visited_keys = set(getattr(revisit_mission, "visited_node_keys", []) or []) if revisit_mission is not None else set()
        target_key = ""
        if revisit_mission is not None and hasattr(revisit_mission, "current"):
            try:
                cur = revisit_mission.current()
                target_key = getattr(cur, "node_key", "") if cur is not None else ""
            except Exception:
                target_key = ""

        for node_key, records in by_node.items():
            x, y = self._marker_node_xy(records[0], planner=planner)
            p = self._world_to_canvas(x, y, bounds, size, pad)

            if node_key == target_key:
                fill = (0, 180, 255)       # current target: orange
                radius = 14
            elif node_key in visited_keys:
                fill = (60, 180, 75)       # revisited marker node: green
                radius = 11
            else:
                fill = (40, 40, 220)       # found/future marker node: red
                radius = 11

            cv2.circle(canvas, p, radius, fill, -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, p, radius + 3, (255, 255, 255), 2, lineType=cv2.LINE_AA)
            cv2.circle(canvas, p, radius + 5, (50, 50, 50), 1, lineType=cv2.LINE_AA)

            if self.marker_map_show_labels:
                ids = ",".join(str(getattr(r, "marker_id", "?")) for r in records)
                cv2.putText(canvas, f"ID {ids}", (p[0] + 14, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (20, 20, 20), 2)
                cv2.putText(canvas, node_key.replace("N:", ""), (p[0] + 14, p[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (60, 60, 60), 1)

        self._draw_current_pose_on_map(canvas, pose, bounds, size, pad)

        reverse_text = " -> ".join(str(getattr(r, "marker_id", "?")) for r in reverse_records)
        if not reverse_text:
            reverse_text = "not ready"
        status_text = aruco_memory.status_text() if aruco_memory is not None and hasattr(aruco_memory, "status_text") else "Aruco ?"
        revisit_text = revisit_mission.status_text() if revisit_mission is not None and hasattr(revisit_mission, "status_text") else "revisit: not_started"

        legend = [
            status_text,
            revisit_text,
            f"reverse IDs: {reverse_text[:60]}",
            "orange: current target",
            "green: already revisited marker node",
            "red: found/future marker node",
            "ordinary nodes are hidden",
        ]
        if planner_info and planner_info.get("next"):
            legend.append(f"next grid node: {str(planner_info.get('next')).replace('N:', '')}")

        y0 = size - 18 - 18 * len(legend)
        y = max(42, y0)
        for text in legend:
            cv2.putText(canvas, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (25, 25, 25), 1)
            y += 18

        if self.marker_map_save_panel:
            out_path = Path(self.save_dir) / self.marker_map_save_name
            try:
                cv2.imwrite(str(out_path), canvas)
            except Exception:
                pass

        return canvas

    def _make_debug_image(self, bgr, detection, forward, yaw_rate, pose, state, planner_info=None, planner=None, aruco_memory=None, revisit_mission=None) -> np.ndarray:
        vis = bgr.copy()
        h, w = vis.shape[:2]
        x1, y1, x2, y2 = detection.roi

        # ROI and image center
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.line(vis, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)
        cv2.line(vis, (0, h // 2), (w, h // 2), (255, 255, 0), 1)
        self._draw_forward_arrow(vis)

        if detection.found and detection.center is not None:
            cx, cy = detection.center
            cv2.circle(vis, (cx, cy), 7, (0, 0, 255), -1)
            cv2.line(vis, (w // 2, h // 2), (cx, cy), (0, 0, 255), 2)

        self._draw_node(vis, detection)

        lines = [
            f"STATE: {state}",
            f"line found: {detection.found}  reason: {detection.reason}",
            f"line error: {detection.error:+.3f}  area: {detection.mask_area}",
            f"node found: {detection.node_found}  type: {detection.node_type}",
            f"node reason: {detection.node_reason}",
            f"node dirs: {list(detection.active_dirs)}  img: {list(detection.active_image_dirs)}",
            f"node err lat/fwd: {detection.node_lateral_error:+.3f}/{detection.node_forward_error:+.3f}",
            f"node area/span: {detection.node_area} / {detection.node_span_x:.2f},{detection.node_span_y:.2f}",
        ]
        if planner_info:
            lines.append(
                "plan: "
                f"decision={planner_info.get('decision', 'none')} "
                f"global={planner_info.get('global', '')} "
                f"reason={planner_info.get('reason', '')}"
            )
            if planner_info.get("node") or planner_info.get("next"):
                lines.append(f"map: node={planner_info.get('node', '')} next={planner_info.get('next', '')}")

        topo = getattr(planner, "map", None) if planner is not None else None
        node_count = len(getattr(topo, "nodes", {})) if topo is not None else 0
        edge_count = len(getattr(topo, "edges", {})) if topo is not None else 0

        lines += [
            f"cmd: forward={forward:+.3f} m/s  yaw_rate={yaw_rate:+.3f} rad/s",
            f"pose: x={pose.x:+.2f}, y={pose.y:+.2f}, yaw={pose.yaw:+.2f}",
            f"odom map: trace={len(self.pose_history)} sampled_nodes={len(self.pose_nodes)} topo_nodes={node_count} topo_edges={edge_count}",
        ]
        if detection.angle_deg is not None:
            lines.append(f"line_angle_img: {detection.angle_deg:+.1f} deg")

        y = 24
        for text in lines:
            cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 4)
            cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
            y += 22

        # 2x2 dashboard:
        #   top-left     : down camera overlay
        #   top-right    : mask / node view
        #   bottom-left  : full topology map
        #   bottom-right : ArUco marker-node-only map
        self._draw_panel_title(vis, "Down Camera + Line/Node Overlay")

        if self.show_mask:
            mask_bgr = cv2.cvtColor(detection.mask, cv2.COLOR_GRAY2BGR)
        else:
            mask_bgr = np.full_like(vis, 40)
            cv2.putText(mask_bgr, "Mask view disabled", (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)

        cv2.rectangle(mask_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)
        if detection.node_center is not None:
            nx, ny = detection.node_center
            color = (255, 0, 255) if detection.node_found else (160, 160, 160)
            cv2.circle(mask_bgr, (nx, ny), 10, color, 2)
            cv2.drawMarker(mask_bgr, (nx, ny), color, cv2.MARKER_CROSS, 26, 2)
        self._draw_panel_title(mask_bgr, "Down Camera Mask / Node View", f"found={detection.found} node={detection.node_found}")

        if self.odom_enabled:
            full_map = self._make_pseudo_odom_map(
                pose,
                state,
                planner=planner,
                planner_info=planner_info,
                revisit_mission=revisit_mission,
            )
        else:
            full_map = np.full((h, w, 3), 45, dtype=np.uint8)
            self._draw_panel_title(full_map, "Full Topology Map", "disabled")

        if self.marker_map_enabled:
            marker_map = self._make_marker_node_map(
                pose,
                state,
                planner=planner,
                planner_info=planner_info,
                aruco_memory=aruco_memory,
                revisit_mission=revisit_mission,
            )
        else:
            marker_map = np.full((h, w, 3), 45, dtype=np.uint8)
            self._draw_panel_title(marker_map, "ArUco Marker Node Map", "disabled")

        full_map = self._resize_panel(full_map, w, h)
        marker_map = self._resize_panel(marker_map, w, h)

        top_row = np.hstack([vis, mask_bgr])
        bottom_row = np.hstack([full_map, marker_map])
        combined = np.vstack([top_row, bottom_row])

        if self.scale != 1.0:
            combined = cv2.resize(combined, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        return combined

    def update(self, bgr, detection, forward, yaw_rate, pose, state, planner_info=None, planner=None, aruco_memory=None, revisit_mission=None) -> bool:
        """Update debug window. Return False if user requested quit."""
        if not self.enabled:
            return True

        try:
            debug_img = self._make_debug_image(
                bgr,
                detection,
                forward,
                yaw_rate,
                pose,
                state,
                planner_info=planner_info,
                planner=planner,
                aruco_memory=aruco_memory,
                revisit_mission=revisit_mission,
            )

            if self.save_frames:
                out_path = self.save_dir / f"line_tracker_{self.frame_idx:06d}.png"
                cv2.imwrite(str(out_path), debug_img)

            cv2.imshow(self.window_name, debug_img)
            key = cv2.waitKey(1) & 0xFF
            self.frame_idx += 1
            if key in (ord("q"), 27):
                return False
        except cv2.error as exc:
            print(f"[WARN] OpenCV debug window disabled due to error: {exc}")
            self.enabled = False
        return True

    def close(self) -> None:
        if self.enabled:
            try:
                cv2.destroyWindow(self.window_name)
            except cv2.error:
                pass
