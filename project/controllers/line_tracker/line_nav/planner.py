"""True DFS next-direction planner for the grid line tracker.

Planner v5:
- keeps a real DFS stack of the current exploration path
- explores only unvisited/unfailed candidate edges first
- backtracks only to the DFS parent node, not to any arbitrary visible "back" edge
- stops at the DFS root when no unvisited outgoing edge remains
- avoids the A <-> B infinite bouncing that appeared in v4 local backtracking
- v6: optionally blocks newly selecting already-discovered nodes and entry-node reentry
"""

from __future__ import annotations

import math
import heapq
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .map_graph import (
    GLOBAL_DIRS,
    TopologyMap,
    global_to_yaw,
    relative_to_global_dir,
    yaw_to_global_dir,
)
from .robot_motion import Pose2D


@dataclass
class PlannerDecision:
    action: str  # GO | STOP
    relative_dir: str = "stop"
    global_dir: str = ""
    target_yaw: float = 0.0
    current_node_key: str = ""
    next_node_key: str = ""
    reason: str = ""


class DirectionPlanner:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.planner_cfg = cfg.get("planner", {})
        self.enabled = bool(self.planner_cfg.get("enabled", True))
        self.priority = list(self.planner_cfg.get("decision_priority", ["front", "left", "right", "back"]))
        self.allow_backtracking = bool(self.planner_cfg.get("allow_backtracking", True))
        self.mark_attempted_edges = bool(self.planner_cfg.get("mark_attempted_edges", True))
        self.node_rearm_distance_m = float(self.planner_cfg.get("node_rearm_distance_m", 0.65))

        # v4/v5 exploration policy.
        self.avoid_outside_edges = bool(self.planner_cfg.get("avoid_outside_edges", True))
        self.allow_outside_edges = bool(self.planner_cfg.get("allow_outside_edges", False))
        self.prefer_inward_from_boundary = bool(self.planner_cfg.get("prefer_inward_from_boundary", True))

        # v6 exploration policy.
        # True means: do not treat an edge to an already discovered node as a new exploration target.
        # Backtracking to the DFS parent is still allowed via _dfs_parent_decision().
        self.avoid_revisited_nodes = bool(self.planner_cfg.get("avoid_revisited_nodes", True))
        # Prevent returning to the grid-side entry node during normal exploration,
        # except when DFS stack explicitly backtracks to it.
        self.avoid_entry_node_reentry = bool(self.planner_cfg.get("avoid_entry_node_reentry", True))

        # v5: true DFS stack.  The last element is the current DFS node.
        self.use_true_dfs_stack = bool(self.planner_cfg.get("use_true_dfs_stack", True))
        self.dfs_stack: List[str] = []

        self.map = TopologyMap(cfg)
        self.last_center_pose: Optional[Pose2D] = None
        self.current_node_key: str = ""
        self.departed_from_node_key: str = ""
        self.last_depart_global_dir: str = ""
        self.last_decision: PlannerDecision = PlannerDecision(action="STOP", reason="not_started")

    def should_accept_node(self, pose: Pose2D) -> bool:
        """Prevent immediately re-detecting the same node after leaving it."""
        if self.last_center_pose is None:
            return True
        dist = math.hypot(pose.x - self.last_center_pose.x, pose.y - self.last_center_pose.y)
        return dist >= self.node_rearm_distance_m

    def force_rearm_node_detection(self) -> None:
        """Allow the same node to be accepted again, used after backtracking."""
        self.last_center_pose = None

    def _relative_dirs_to_global(self, active_relative_dirs: Iterable[str], yaw: float) -> List[Tuple[str, str]]:
        pairs = []
        for rel in active_relative_dirs:
            if rel not in {"front", "back", "left", "right"}:
                continue
            pairs.append((rel, relative_to_global_dir(yaw, rel)))
        return pairs

    def _global_to_relative_dir(self, current_yaw: float, global_dir: str) -> str:
        """Convert a global E/N/W/S direction to current robot-relative direction."""
        heading_idx = GLOBAL_DIRS.index(yaw_to_global_dir(current_yaw))
        target_idx = GLOBAL_DIRS.index(global_dir)
        diff = (target_idx - heading_idx) % 4
        if diff == 0:
            return "front"
        if diff == 1:
            return "left"
        if diff == 2:
            return "back"
        return "right"

    def _global_dir_between_nodes(self, a_key: str, b_key: str) -> str:
        """Return the dominant global direction from node a to node b."""
        a = self.map.nodes[a_key]
        b = self.map.nodes[b_key]
        dx = float(b.x) - float(a.x)
        dy = float(b.y) - float(a.y)
        if abs(dx) >= abs(dy):
            return "E" if dx >= 0 else "W"
        return "N" if dy >= 0 else "S"

    def _sync_dfs_stack(self, node_key: str) -> None:
        """Synchronize the DFS stack with the node that has just been centered.

        - first node: initialize root
        - new child: push
        - backtracked ancestor: pop until ancestor is top
        - same node: keep stack unchanged
        """
        if not self.use_true_dfs_stack:
            return

        if not self.dfs_stack:
            self.dfs_stack = [node_key]
            return

        if self.dfs_stack[-1] == node_key:
            return

        if node_key in self.dfs_stack:
            while self.dfs_stack and self.dfs_stack[-1] != node_key:
                self.dfs_stack.pop()
            return

        # We reached a newly discovered child.
        self.dfs_stack.append(node_key)

    def on_node_centered(self, pose: Pose2D, detection) -> PlannerDecision:
        """Update map at a centered node and choose the next direction."""
        active_rel = list(detection.active_dirs or [])
        active_global = [g for _, g in self._relative_dirs_to_global(active_rel, pose.yaw)]

        node_key = self.map.register_node(
            pose=pose,
            node_type=getattr(detection, "node_type", "UNKNOWN"),
            observed_global_dirs=active_global,
        )

        # If we previously departed from another node and have now centered a
        # different node, confirm that traversed edge.
        if self.departed_from_node_key and self.departed_from_node_key != node_key:
            self.map.add_edge(self.departed_from_node_key, node_key, self.last_depart_global_dir)

        self.current_node_key = node_key
        self.last_center_pose = Pose2D(pose.x, pose.y, pose.z, pose.yaw)

        # v5: maintain actual DFS path stack before choosing.
        self._sync_dfs_stack(node_key)

        decision = self.choose_next_direction(node_key, pose.yaw, active_rel)
        self.last_decision = decision

        if decision.action == "GO":
            self.departed_from_node_key = node_key
            self.last_depart_global_dir = decision.global_dir
            if self.mark_attempted_edges and decision.next_node_key:
                self.map.mark_planned_edge(node_key, decision.next_node_key, decision.global_dir)
        else:
            self.departed_from_node_key = ""
            self.last_depart_global_dir = ""

        self.map.save()
        return decision

    def mark_decision_failed(self, decision: PlannerDecision, reason: str = "line_lost") -> None:
        if not decision or decision.action != "GO":
            return
        if not decision.current_node_key or not decision.next_node_key:
            return

        # If a forward exploration edge failed before reaching the child, keep
        # the DFS stack at the current/source node and block this edge.
        self.map.mark_failed_edge(
            decision.current_node_key,
            decision.next_node_key,
            decision.global_dir,
            reason=reason,
        )
        self.departed_from_node_key = ""
        self.last_depart_global_dir = ""
        self.map.save()

    def _candidate_is_blocked(self, node_key: str, neighbor: str) -> tuple[bool, str]:
        """Return whether a candidate should be blocked before normal edge testing.

        This is the v6 anti-revisit guard:
        - Known/discovered neighbor nodes are not selected as *new* exploration targets.
        - Entry node is not re-entered as a normal candidate.
        - DFS parent backtracking is handled separately, so it is not blocked here.
        """
        if self.avoid_entry_node_reentry and self.map.is_entry_node_key(neighbor):
            # Allow only if this is the current root entry node. Otherwise, do not re-enter it
            # as a normal unvisited branch. Parent backtracking can still return to root.
            if not (self.dfs_stack and len(self.dfs_stack) <= 1 and self.dfs_stack[-1] == neighbor):
                return True, "entry_node_reentry"

        if self.avoid_revisited_nodes and self.map.node_known(neighbor):
            return True, "neighbor_node_already_known"

        return False, ""

    def _candidate_sort_key(self, node_key: str, rel: str, global_dir: str, neighbor: str) -> tuple:
        """Sort unvisited candidates.

        Lower is better:
        1. inward branches from boundary nodes
        2. normal in-grid edges
        3. outside/entry road edges only if explicitly allowed
        4. user priority within the same class
        """
        priority_idx = self.priority.index(rel) if rel in self.priority else 999
        outside = self.map.candidate_goes_outside_grid(node_key, global_dir)
        inward = self.map.candidate_goes_inward_from_boundary(node_key, global_dir)

        if outside:
            outside_rank = 99 if self.avoid_outside_edges else 10
        elif inward and self.prefer_inward_from_boundary:
            outside_rank = -1
        else:
            outside_rank = 0
        return (outside_rank, priority_idx)

    def _iter_unvisited_candidates(self, node_key: str, yaw: float, active_rel_set: set[str]):
        candidates = []
        for rel in self.priority:
            if rel not in active_rel_set:
                continue

            # In true DFS, backtracking is handled only by dfs_stack parent,
            # not by the locally visible relative "back" direction.
            if rel == "back":
                continue

            g = relative_to_global_dir(yaw, rel)
            neighbor = self.map.neighbor_key_from_direction(node_key, g)

            blocked, _block_reason = self._candidate_is_blocked(node_key, neighbor)
            if blocked:
                continue

            if self.map.edge_failed(node_key, neighbor):
                continue
            if self.map.edge_known(node_key, neighbor):
                continue

            outside = self.map.candidate_goes_outside_grid(node_key, g)
            if outside and self.avoid_outside_edges and not self.allow_outside_edges:
                continue

            candidates.append((rel, g, neighbor))

        candidates.sort(key=lambda item: self._candidate_sort_key(node_key, item[0], item[1], item[2]))
        return candidates

    def _make_go_decision(self, node_key: str, rel: str, g: str, neighbor: str, reason: str) -> PlannerDecision:
        return PlannerDecision(
            action="GO",
            relative_dir=rel,
            global_dir=g,
            target_yaw=global_to_yaw(g),
            current_node_key=node_key,
            next_node_key=neighbor,
            reason=reason,
        )

    def _dfs_parent_decision(self, node_key: str, yaw: float) -> PlannerDecision | None:
        """Return a decision to move to the DFS parent, or None at root."""
        if not self.use_true_dfs_stack or len(self.dfs_stack) <= 1:
            return None
        if self.dfs_stack[-1] != node_key:
            # Defensive sync in case a snapped node key changed.
            self._sync_dfs_stack(node_key)
            if len(self.dfs_stack) <= 1 or self.dfs_stack[-1] != node_key:
                return None

        parent = self.dfs_stack[-2]
        g = self._global_dir_between_nodes(node_key, parent)
        rel = self._global_to_relative_dir(yaw, g)

        # Do not require rel to be in active_dirs; the stack is the source of
        # truth for DFS backtracking. If perception temporarily misses the back
        # arm, the global leave step still keeps the robot on the grid axis.
        return self._make_go_decision(node_key, rel, g, parent, "dfs_backtrack")

    def choose_next_direction(self, node_key: str, yaw: float, active_relative_dirs: Iterable[str]) -> PlannerDecision:
        if not self.enabled:
            return PlannerDecision(action="STOP", current_node_key=node_key, reason="planner_disabled")

        active_rel_set = set(active_relative_dirs or [])
        if not active_rel_set:
            return PlannerDecision(action="STOP", current_node_key=node_key, reason="no_available_dirs")

        # First pass: choose an unvisited edge.
        candidates = self._iter_unvisited_candidates(node_key, yaw, active_rel_set)
        if candidates:
            rel, g, neighbor = candidates[0]
            reason = "dfs_unvisited_edge"
            if self.map.candidate_goes_inward_from_boundary(node_key, g):
                reason = "dfs_inward_unvisited_edge"
            return self._make_go_decision(node_key, rel, g, neighbor, reason)

        # Second pass: true DFS backtracking to stack parent only.
        if self.allow_backtracking:
            parent_decision = self._dfs_parent_decision(node_key, yaw)
            if parent_decision is not None:
                return parent_decision

        # Third pass: optionally allow known edge reuse for non-DFS debug modes.
        # In true DFS mode, keep this disabled to avoid infinite loops.
        if (not self.use_true_dfs_stack) and bool(self.planner_cfg.get("allow_known_edge_reuse", False)):
            for rel in self.priority:
                if rel not in active_rel_set:
                    continue
                g = relative_to_global_dir(yaw, rel)
                neighbor = self.map.neighbor_key_from_direction(node_key, g)
                if self.map.edge_failed(node_key, neighbor):
                    continue
                outside = self.map.candidate_goes_outside_grid(node_key, g)
                if outside and self.avoid_outside_edges and not self.allow_outside_edges:
                    continue
                return self._make_go_decision(node_key, rel, g, neighbor, "reuse_known_edge")

        # If root has no unvisited candidates left, DFS is complete.
        if self.use_true_dfs_stack and len(self.dfs_stack) <= 1:
            return PlannerDecision(action="STOP", current_node_key=node_key, reason="dfs_complete")

        return PlannerDecision(action="STOP", current_node_key=node_key, reason="no_unvisited_edge")


class GridRoutePlanner:
    """Shortest-path planner for the post-ArUco return mission.

    This planner is intentionally separate from ``DirectionPlanner``.

    DirectionPlanner is used only during the exploration phase, where the drone
    discovers ArUco markers with DFS and records a topology map.

    GridRoutePlanner is used after all ArUco markers have been found.  It does
    not use DFS stack order and does not use the previously traversed topology
    edges.  Instead, it builds a route on the known rectangular grid described
    by world.yaml and treats ArUco marker nodes as ordered mission targets.
    Ordinary grid nodes may be reused freely; marker nodes other than the route
    start and current goal are blocked so that the reverse marker order cannot
    be skipped or revisited accidentally.
    """

    def __init__(self, cfg: Dict, topology_map: TopologyMap):
        self.cfg = cfg
        self.map = topology_map
        self.route_cfg = cfg.get("marker_return", cfg.get("marker_revisit", {})) or {}
        self.enabled = bool(self.route_cfg.get("enabled", True))
        self.priority = list(self.route_cfg.get("direction_priority", ["E", "N", "W", "S"]))

    def _parse_node_key_xy(self, node_key: str) -> Tuple[float, float]:
        if node_key in self.map.nodes:
            node = self.map.nodes[node_key]
            return float(node.x), float(node.y)
        try:
            body = str(node_key).replace("N:", "", 1)
            x_s, y_s = body.split(",", 1)
            return float(x_s), float(y_s)
        except Exception:
            return 0.0, 0.0

    def _grid_node_key(self, x: float, y: float) -> str:
        return self.map.key_from_xy(float(x), float(y))

    def nearest_grid_node_key(self, pose_or_key) -> str:
        """Return a canonical grid/entry key for a pose or an existing key.

        ``TopologyMap.key_from_xy`` snaps near entry_path.start/end before
        snapping to rectangular grid vertices.  This matters because the
        entry-road T-junction can be located at the middle of a boundary segment
        such as ``N:0.00,-4.00``.  That node is not always in ``grid_xs ×
        grid_ys``, but it is a real line node and must be reachable during the
        final return phase.
        """
        if isinstance(pose_or_key, str):
            x, y = self._parse_node_key_xy(pose_or_key)
        else:
            x, y = float(pose_or_key.x), float(pose_or_key.y)
        return self._grid_node_key(x, y)

    def entry_node_key(self) -> str:
        return self.map.entry_node_key()

    def _same_xy(self, a: float, b: float) -> bool:
        return abs(float(a) - float(b)) <= max(1e-6, self.map.grid_snap_tolerance_m * 0.5)

    def _entry_node_xy(self) -> Tuple[float, float] | None:
        entry_key = self.entry_node_key()
        if not entry_key:
            return None
        return self._parse_node_key_xy(entry_key)

    def _is_entry_node_key(self, node_key: str) -> bool:
        entry_key = self.entry_node_key()
        return bool(entry_key and node_key == entry_key)

    def _is_inside_grid_key(self, node_key: str) -> bool:
        # entry_path.end_xy is allowed as a route node even when it is not a
        # rectangular grid vertex, e.g. bottom-center N:0.00,-4.00.
        if self._is_entry_node_key(node_key):
            return True
        x, y = self._parse_node_key_xy(node_key)
        return not self.map.xy_is_outside_grid(x, y, margin=self.map.grid_snap_tolerance_m * 0.2)

    def _entry_adjacent_grid_neighbors(self, node_key: str) -> List[Tuple[str, str, float]]:
        """Special edges between entry_path.end_xy and nearby grid vertices.

        The rectangular planner normally moves by ``grid_dx``/``grid_dy``.  The
        entry-road T-junction may sit between two boundary grid vertices, so the
        normal step rule cannot reach it.  This method adds short edges such as:

            N:-0.80,-4.00 <-> N:0.00,-4.00 <-> N:0.80,-4.00

        This fixes the final stop at ``grid_route_no_path;target=N:0.00,-4.00``.
        """
        entry_key = self.entry_node_key()
        entry_xy = self._entry_node_xy()
        if not entry_key or entry_xy is None or not self.map.grid_xs or not self.map.grid_ys:
            return []

        ex, ey = entry_xy
        x, y = self._parse_node_key_xy(node_key)
        out: List[Tuple[str, str, float]] = []

        def direction_and_dist(x0: float, y0: float, x1: float, y1: float) -> Tuple[str, float]:
            dx = x1 - x0
            dy = y1 - y0
            dist = math.hypot(dx, dy)
            if abs(dx) >= abs(dy):
                return ("E" if dx >= 0.0 else "W"), dist
            return ("N" if dy >= 0.0 else "S"), dist

        # normal grid node -> entry node
        if node_key != entry_key:
            if self._same_xy(y, ey) and abs(x - ex) <= float(self.map.grid_dx) + self.map.grid_snap_tolerance_m:
                g, dist = direction_and_dist(x, y, ex, ey)
                if dist > 1e-9:
                    out.append((entry_key, g, dist))
            if self._same_xy(x, ex) and abs(y - ey) <= float(self.map.grid_dy) + self.map.grid_snap_tolerance_m:
                g, dist = direction_and_dist(x, y, ex, ey)
                if dist > 1e-9:
                    out.append((entry_key, g, dist))
            return out

        # entry node -> nearby rectangular grid vertices on the boundary line.
        # Bottom/top entry: connect horizontally to the nearest boundary vertices.
        if self._same_xy(ey, min(self.map.grid_ys)) or self._same_xy(ey, max(self.map.grid_ys)):
            for gx in self.map.grid_xs:
                gx = float(gx)
                if abs(gx - ex) <= float(self.map.grid_dx) + self.map.grid_snap_tolerance_m:
                    nk = self._grid_node_key(gx, ey)
                    if nk == entry_key:
                        continue
                    g, dist = direction_and_dist(ex, ey, gx, ey)
                    if dist > 1e-9:
                        out.append((nk, g, dist))

        # Left/right entry: connect vertically to the nearest boundary vertices.
        if self._same_xy(ex, min(self.map.grid_xs)) or self._same_xy(ex, max(self.map.grid_xs)):
            for gy in self.map.grid_ys:
                gy = float(gy)
                if abs(gy - ey) <= float(self.map.grid_dy) + self.map.grid_snap_tolerance_m:
                    nk = self._grid_node_key(ex, gy)
                    if nk == entry_key:
                        continue
                    g, dist = direction_and_dist(ex, ey, ex, gy)
                    if dist > 1e-9:
                        out.append((nk, g, dist))

        dedup: List[Tuple[str, str, float]] = []
        seen = set()
        for nk, g, cost in out:
            if nk in seen:
                continue
            seen.add(nk)
            dedup.append((nk, g, cost))
        return dedup

    def _axis_step(self, global_dir: str) -> Tuple[float, float]:
        d = str(global_dir).upper()
        if d == "E":
            return float(self.map.grid_dx), 0.0
        if d == "W":
            return -float(self.map.grid_dx), 0.0
        if d == "N":
            return 0.0, float(self.map.grid_dy)
        if d == "S":
            return 0.0, -float(self.map.grid_dy)
        return 0.0, 0.0

    def neighbors(self, node_key: str) -> List[Tuple[str, str, float]]:
        """Return ``[(neighbor_key, global_dir, cost), ...]`` on the full grid.

        In addition to rectangular grid vertices, this supports the grid-side
        entry node from ``entry_path.end_xy`` as a special pseudo-grid node.
        """
        if not self.enabled:
            return []
        if not self.map.grid_xs or not self.map.grid_ys:
            return []

        x, y = self._parse_node_key_xy(node_key)
        out: List[Tuple[str, str, float]] = []

        if self._is_entry_node_key(node_key):
            return self._entry_adjacent_grid_neighbors(node_key)

        for g in self.priority:
            dx, dy = self._axis_step(g)
            if dx == 0.0 and dy == 0.0:
                continue
            nk = self._grid_node_key(x + dx, y + dy)
            if not self._is_inside_grid_key(nk):
                continue
            nx, ny = self._parse_node_key_xy(nk)
            cost = math.hypot(nx - x, ny - y)
            if cost <= 1e-9:
                continue
            out.append((nk, str(g).upper(), cost))

        # Extra edge to the entry T-junction if this node is a neighboring
        # boundary vertex.
        out.extend(self._entry_adjacent_grid_neighbors(node_key))

        dedup: List[Tuple[str, str, float]] = []
        seen = set()
        for nk, g, cost in out:
            if nk in seen:
                continue
            seen.add(nk)
            dedup.append((nk, g, cost))
        return dedup

    def _heuristic(self, a_key: str, b_key: str) -> float:
        ax, ay = self._parse_node_key_xy(a_key)
        bx, by = self._parse_node_key_xy(b_key)
        return abs(ax - bx) + abs(ay - by)

    def shortest_path(
        self,
        start_key: str,
        goal_key: str,
        forbidden_nodes: set[str] | None = None,
    ) -> List[str]:
        """Shortest path on the predefined full grid, not on visited DFS edges."""
        if not self.enabled or not start_key or not goal_key:
            return []

        start_key = self.nearest_grid_node_key(start_key)
        goal_key = self.nearest_grid_node_key(goal_key)

        forbidden = set(forbidden_nodes or set())
        # The robot may already be standing on a marker node, and the current
        # target marker must be reachable.  All other marker nodes remain blocked.
        forbidden.discard(start_key)
        forbidden.discard(goal_key)

        if start_key == goal_key:
            return [start_key]
        if start_key in forbidden or goal_key in forbidden:
            return []
        if not self._is_inside_grid_key(start_key) or not self._is_inside_grid_key(goal_key):
            return []

        frontier: List[Tuple[float, float, str]] = []
        heapq.heappush(frontier, (self._heuristic(start_key, goal_key), 0.0, start_key))
        parent: Dict[str, Optional[str]] = {start_key: None}
        cost_so_far: Dict[str, float] = {start_key: 0.0}

        while frontier:
            _, cur_cost, cur = heapq.heappop(frontier)
            if cur == goal_key:
                break
            if cur_cost > cost_so_far.get(cur, float("inf")) + 1e-9:
                continue
            for nxt, _g, step_cost in self.neighbors(cur):
                if nxt in forbidden:
                    continue
                new_cost = cur_cost + step_cost
                if new_cost + 1e-9 < cost_so_far.get(nxt, float("inf")):
                    cost_so_far[nxt] = new_cost
                    parent[nxt] = cur
                    priority = new_cost + self._heuristic(nxt, goal_key)
                    heapq.heappush(frontier, (priority, new_cost, nxt))

        if goal_key not in parent:
            return []

        path = []
        cur = goal_key
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return path

    def global_dir_between_nodes(self, a_key: str, b_key: str) -> str:
        ax, ay = self._parse_node_key_xy(a_key)
        bx, by = self._parse_node_key_xy(b_key)
        dx = bx - ax
        dy = by - ay
        if abs(dx) >= abs(dy):
            return "E" if dx >= 0 else "W"
        return "N" if dy >= 0 else "S"

    def global_to_relative_dir(self, current_yaw: float, global_dir: str) -> str:
        heading_idx = GLOBAL_DIRS.index(yaw_to_global_dir(current_yaw))
        target_idx = GLOBAL_DIRS.index(global_dir)
        diff = (target_idx - heading_idx) % 4
        if diff == 0:
            return "front"
        if diff == 1:
            return "left"
        if diff == 2:
            return "back"
        return "right"

    def make_decision_to_node(
        self,
        current_key: str,
        target_key: str,
        pose: Pose2D,
        reason: str,
        forbidden_nodes: set[str] | None = None,
    ) -> PlannerDecision:
        current_key = self.nearest_grid_node_key(current_key)
        target_key = self.nearest_grid_node_key(target_key)
        forbidden = set(forbidden_nodes or set())
        path = self.shortest_path(current_key, target_key, forbidden_nodes=forbidden)
        if len(path) == 1:
            return PlannerDecision(
                action="STOP",
                current_node_key=current_key,
                next_node_key=target_key,
                reason=f"grid_route_already_at_target;target={target_key}",
            )
        if len(path) < 2:
            forbidden_text = ",".join(sorted(forbidden)) if forbidden else "none"
            return PlannerDecision(
                action="STOP",
                current_node_key=current_key,
                next_node_key=target_key,
                reason=f"grid_route_no_path;target={target_key};forbidden={forbidden_text}",
            )

        next_key = path[1]
        g = self.global_dir_between_nodes(current_key, next_key)
        rel = self.global_to_relative_dir(pose.yaw, g)
        forbidden_text = ",".join(sorted(forbidden)) if forbidden else "none"
        path_text = "->".join(path)
        return PlannerDecision(
            action="GO",
            relative_dir=rel,
            global_dir=g,
            target_yaw=global_to_yaw(g),
            current_node_key=current_key,
            next_node_key=next_key,
            reason=f"{reason};grid_path_len={len(path)};target={target_key};forbidden={forbidden_text};grid_path={path_text}",
        )
