"""Runtime loop for the rectangular line-tracker Webots controller.

Planner v2 yaw/leave fix:
- follow black line with the down camera
- classify decision nodes: T, cross, ㄱ/ㄴ corners
- center on a node
- record the node/edge in a topological map
- choose next direction using a simple DFS-like planner
- ALWAYS align to the selected GLOBAL yaw before leaving the node
- leave the node along the selected GLOBAL grid direction, not current body yaw
"""

from __future__ import annotations

import argparse
import math
import time
from controller import Supervisor

from .camera_io import enable_camera, read_processed_bgr
from .config_loader import load_config
from .debug_view import DebugView
from .line_detector import detect_line
from .aruco_detector import ArucoMarkerMemory, detect_aruco, draw_aruco_observations
from .planner import DirectionPlanner, GridRoutePlanner, PlannerDecision
from .robot_motion import (
    KinematicRobotMotion,
    Pose2D,
    clamp,
    compute_motion_command,
    normalize_angle,
    shortest_angle_error,
)
from .map_graph import global_to_yaw, opposite_global_dir


STATE_FOLLOW_LINE = "FOLLOW_LINE"
STATE_CENTER_NODE = "CENTER_NODE"
STATE_TURN_TO_DIRECTION = "TURN_TO_DIRECTION"
STATE_LEAVE_NODE = "LEAVE_NODE"
STATE_STOPPED_NO_PLAN = "STOPPED_NO_PLAN"
STATE_LINE_LOST_STOP = "LINE_LOST_STOP"
STATE_BACKTRACK_TO_LAST_NODE = "BACKTRACK_TO_LAST_NODE"
STATE_RETURN_HOME_TURN = "RETURN_HOME_TURN"
STATE_RETURN_HOME_MOVE = "RETURN_HOME_MOVE"
STATE_RETURN_HOME_DONE = "RETURN_HOME_DONE"
STATE_MARKER_REVISIT_ROUTE = "MARKER_REVISIT_ROUTE"
STATE_MARKER_REVISIT_DONE = "MARKER_REVISIT_DONE"


def parse_args():
    parser = argparse.ArgumentParser(description="Webots line tracker controller with node planner")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to line_tracker.yaml. Default: project/config/line_tracker.yaml",
    )
    return parser.parse_args()


def _node_cfg(cfg):
    return cfg.get("node_detection", cfg.get("intersection", {}))


def compute_node_centering_command(detection, cfg):
    """Return forward speed, yaw rate, centered flag for node centering."""
    node = _node_cfg(cfg)
    ctrl = cfg.get("control", {})

    forward_error = float(detection.node_forward_error)
    lateral_error = float(detection.node_lateral_error)

    forward = float(node.get("kp_forward", 0.35)) * forward_error
    max_forward = float(node.get("max_forward_speed_mps", 0.16))
    forward = clamp(forward, -max_forward, max_forward)

    yaw_sign = node.get("yaw_sign", None)
    if yaw_sign is None:
        yaw_sign = float(ctrl.get("yaw_sign", -1.0))
    yaw_rate = float(yaw_sign) * float(node.get("kp_yaw", 1.0)) * lateral_error
    max_yaw = float(node.get("max_yaw_rate", 0.45))
    yaw_rate = clamp(yaw_rate, -max_yaw, max_yaw)

    lat_tol = float(node.get("center_lateral_tolerance", 0.055))
    fwd_tol = float(node.get("center_forward_tolerance", 0.055))
    centered = abs(lateral_error) <= lat_tol and abs(forward_error) <= fwd_tol

    if centered:
        forward = 0.0
        yaw_rate = 0.0

    return forward, yaw_rate, centered


def compute_turn_command(current_yaw: float, target_yaw: float, cfg):
    planner_cfg = cfg.get("planner", {})
    yaw_err = shortest_angle_error(target_yaw, current_yaw)
    tol = float(planner_cfg.get("turn_yaw_tolerance_rad", 0.04))
    if abs(yaw_err) <= tol:
        return 0.0, True, yaw_err
    yaw_rate = float(planner_cfg.get("turn_kp_yaw", 2.2)) * yaw_err
    max_yaw = float(planner_cfg.get("turn_max_yaw_rate", 0.85))
    yaw_rate = clamp(yaw_rate, -max_yaw, max_yaw)
    return yaw_rate, False, yaw_err


def _get_node_xy(planner: DirectionPlanner, node_key: str, fallback_pose: Pose2D):
    node = planner.map.nodes.get(node_key) if node_key else None
    if node is None:
        return float(fallback_pose.x), float(fallback_pose.y)
    return float(node.x), float(node.y)


def _prepare_leave_start_pose(
    motion: KinematicRobotMotion,
    planner: DirectionPlanner,
    decision: PlannerDecision,
    cfg,
) -> Pose2D:
    """Prepare a clean, grid-aligned pose at the start of LEAVE_NODE.

    This is the key stabilization step.  After node centering, the robot may have
    a small yaw/position bias caused by visual centering.  Before leaving, we
    enforce the selected global yaw and optionally snap x/y to the canonical map
    node coordinate.  Then LEAVE_NODE moves along a global axis.
    """
    pose = motion.get_pose()
    target_yaw = normalize_angle(decision.target_yaw)
    planner_cfg = cfg.get("planner", {})

    if bool(planner_cfg.get("leave_snap_to_node_center", True)):
        x, y = _get_node_xy(planner, decision.current_node_key, pose)
    else:
        x, y = pose.x, pose.y

    motion.set_pose(x, y, motion.fixed_z, target_yaw)
    return motion.get_pose()


def traveled_along_global(pose: Pose2D, start: Pose2D, global_dir: str) -> float:
    d = str(global_dir).upper()
    dx = float(pose.x) - float(start.x)
    dy = float(pose.y) - float(start.y)
    if d == "E":
        return max(0.0, dx)
    if d == "W":
        return max(0.0, -dx)
    if d == "N":
        return max(0.0, dy)
    if d == "S":
        return max(0.0, -dy)
    return math.hypot(dx, dy)



def make_backtrack_target(planner: DirectionPlanner, decision: PlannerDecision, motion: KinematicRobotMotion) -> Pose2D | None:
    """Return the previous node pose for line-lost backtracking."""
    if not decision or decision.action != "GO" or not decision.current_node_key:
        return None
    node = planner.map.nodes.get(decision.current_node_key)
    if node is None:
        return None
    # Face back into the grid/previous node while returning.
    return_yaw = global_to_yaw(opposite_global_dir(decision.global_dir)) if decision.global_dir else motion.get_pose().yaw
    return Pose2D(float(node.x), float(node.y), motion.fixed_z, return_yaw)

def _global_dir_from_to(src: Pose2D, dst: Pose2D) -> str:
    """Dominant global direction from src to dst."""
    dx = float(dst.x) - float(src.x)
    dy = float(dst.y) - float(src.y)
    if abs(dx) >= abs(dy):
        return "E" if dx >= 0 else "W"
    return "N" if dy >= 0 else "S"


def make_return_home_target(planner: DirectionPlanner, initial_pose: Pose2D, motion: KinematicRobotMotion, cfg) -> Pose2D:
    """Return the final home pose after DFS completes.

    For the generated world, this is normally entry_path.start_xy, i.e. the
    outside end of the entry road. If world.yaml is unavailable, fall back to
    the robot's initial Supervisor pose.
    """
    rh = cfg.get("return_home", {})
    target_mode = str(rh.get("target", "entry_start"))

    if target_mode == "entry_start" and getattr(planner.map, "entry_start_xy", None) is not None:
        x, y = planner.map.entry_start_xy
        # Face along the direction from the grid-side entry node toward the home point.
        src_pose = motion.get_pose()
        if getattr(planner.map, "entry_end_xy", None) is not None:
            ex, ey = planner.map.entry_end_xy
            src_pose = Pose2D(float(ex), float(ey), motion.fixed_z, src_pose.yaw)
        dst_pose = Pose2D(float(x), float(y), motion.fixed_z, src_pose.yaw)
        yaw = global_to_yaw(_global_dir_from_to(src_pose, dst_pose))
        return Pose2D(float(x), float(y), motion.fixed_z, yaw)

    return Pose2D(float(initial_pose.x), float(initial_pose.y), motion.fixed_z, float(initial_pose.yaw))


def register_centered_node_only(planner: DirectionPlanner, pose: Pose2D, detection):
    """Register the current centered node without immediately choosing DFS.

    This allows the marker-revisit mission to reuse the same topology map update
    path while overriding the next movement target.
    """
    active_rel = list(detection.active_dirs or [])
    active_global = [g for _, g in planner._relative_dirs_to_global(active_rel, pose.yaw)]

    node_key = planner.map.register_node(
        pose=pose,
        node_type=getattr(detection, "node_type", "UNKNOWN"),
        observed_global_dirs=active_global,
    )

    if planner.departed_from_node_key and planner.departed_from_node_key != node_key:
        planner.map.add_edge(planner.departed_from_node_key, node_key, planner.last_depart_global_dir)

    planner.current_node_key = node_key
    planner.last_center_pose = Pose2D(pose.x, pose.y, pose.z, pose.yaw)
    planner._sync_dfs_stack(node_key)
    planner.map.save()
    return node_key, active_rel


def commit_decision_departure(planner: DirectionPlanner, decision: PlannerDecision, mark_attempted: bool = True) -> None:
    """Update planner bookkeeping after selecting a GO/STOP decision."""
    planner.last_decision = decision
    if decision.action == "GO":
        planner.departed_from_node_key = decision.current_node_key
        planner.last_depart_global_dir = decision.global_dir
        if mark_attempted and planner.mark_attempted_edges and decision.next_node_key:
            planner.map.mark_planned_edge(decision.current_node_key, decision.next_node_key, decision.global_dir)
    else:
        planner.departed_from_node_key = ""
        planner.last_depart_global_dir = ""
    planner.map.save()


def choose_normal_dfs_decision(planner: DirectionPlanner, node_key: str, pose: Pose2D, active_rel) -> PlannerDecision:
    decision = planner.choose_next_direction(node_key, pose.yaw, active_rel)
    commit_decision_departure(planner, decision, mark_attempted=True)
    return decision


def build_reverse_marker_node_targets(aruco_memory: ArucoMarkerMemory, cfg):
    """Return only ArUco-marker nodes in reverse discovery order.

    Before all markers are found, DFS may visit many ordinary grid nodes.
    After all markers are found, the revisit mission ignores those ordinary
    nodes as mission targets.  This helper builds a target list consisting only
    of nodes where at least one ArUco marker was recorded.

    If multiple marker IDs are bound to the same node, the node is visited once
    by default.  The representative record keeps the marker ID that appears
    latest in the reverse discovery order, which is useful for logging.
    """
    marker_revisit_cfg = cfg.get("marker_revisit", {}) or {}
    unique_nodes_only = bool(marker_revisit_cfg.get("unique_nodes_only", True))

    targets = []
    seen_node_keys = set()
    for rec in aruco_memory.reverse_targets():
        if not rec.node_key:
            continue
        if unique_nodes_only and rec.node_key in seen_node_keys:
            continue
        seen_node_keys.add(rec.node_key)
        targets.append(rec)
    return targets


class MarkerRevisitMission:
    def __init__(self, cfg):
        self.cfg = cfg
        self.enabled = bool((cfg.get("marker_revisit", {}) or {}).get("enabled", True))
        self.started = False
        self.completed = False
        self.targets = []
        self.index = 0
        self.visited_marker_ids = []
        self.visited_node_keys = []

    def start(self, targets) -> None:
        self.targets = list(targets or [])
        self.index = 0
        self.started = True
        self.completed = len(self.targets) == 0
        self.visited_marker_ids = []
        self.visited_node_keys = []

    def current(self):
        if self.completed or self.index >= len(self.targets):
            return None
        return self.targets[self.index]

    def all_marker_node_keys(self) -> set[str]:
        """All ArUco marker node keys in the reverse mission target set."""
        out = set()
        for rec in self.targets:
            key = getattr(rec, "node_key", "")
            if key:
                out.add(key)
        return out

    def forbidden_node_keys_for_route(self, start_key: str = "", goal_key: str = "") -> set[str]:
        """Marker nodes that cannot be used as intermediate grid nodes.

        Phase 2 rule:
        - ordinary grid nodes are freely reusable;
        - the current start node is allowed because the drone may already be on it;
        - the current goal node is allowed because it is the next marker target;
        - every other ArUco marker node is forbidden.

        This prevents both order skipping and revisiting an already completed
        marker node as a pass-through node.
        """
        forbidden = self.all_marker_node_keys()
        if start_key:
            forbidden.discard(start_key)
        if goal_key:
            forbidden.discard(goal_key)
        return forbidden

    def future_forbidden_node_keys(self) -> set[str]:
        """Backward-compatible alias for debug views.

        Older code only blocked future marker nodes.  The new grid-return planner
        blocks all non-start/non-goal marker nodes, but this method is kept so
        existing debug code can still call it safely.
        """
        current = self.current()
        current_key = getattr(current, "node_key", "") if current is not None else ""
        return self.forbidden_node_keys_for_route(start_key="", goal_key=current_key)

    def is_non_current_marker_node(self, node_key: str) -> bool:
        if not node_key or node_key not in self.all_marker_node_keys():
            return False
        cur = self.current()
        cur_key = getattr(cur, "node_key", "") if cur is not None else ""
        return bool(node_key != cur_key)

    def is_future_marker_node(self, node_key: str) -> bool:
        # Kept for compatibility with earlier runtime logic.
        return self.is_non_current_marker_node(node_key)

    def advance_if_current_node(self, node_key: str):
        arrived = []
        while self.index < len(self.targets) and self.targets[self.index].node_key == node_key:
            rec = self.targets[self.index]
            self.visited_marker_ids.append(rec.marker_id)
            self.visited_node_keys.append(rec.node_key)
            arrived.append(rec)
            self.index += 1
        if self.index >= len(self.targets):
            self.completed = True
        return arrived

    def status_text(self) -> str:
        if not self.started:
            return "revisit: not_started"
        return f"revisit: {self.index}/{len(self.targets)} marker-nodes done"

def _return_home_enabled(cfg) -> bool:
    return bool(cfg.get("return_home", {}).get("enabled", True))

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    dt = timestep / 1000.0

    camera_name = cfg["camera"]["down_camera_name"]
    down_camera = enable_camera(robot, camera_name, timestep)

    motion = KinematicRobotMotion(robot, fixed_z=float(cfg["control"]["fixed_z"]))
    debug = DebugView(cfg)
    planner = DirectionPlanner(cfg)
    # Phase 2 planner: after all ArUco markers are found, ignore the DFS route
    # and plan on the known rectangular grid from world.yaml.
    marker_return_planner = GridRoutePlanner(cfg, planner.map)
    aruco_memory = ArucoMarkerMemory(cfg)
    revisit_mission = MarkerRevisitMission(cfg)

    node = _node_cfg(cfg)
    hold_required = int(node.get("hold_center_frames", 6))
    centered_count = 0
    state = STATE_FOLLOW_LINE

    active_decision = PlannerDecision(action="STOP", reason="not_started")
    leave_start_pose: Pose2D | None = None
    backtrack_target: Pose2D | None = None
    planner_info = {"decision": "none"}
    initial_home_pose = motion.get_pose()
    return_home_target: Pose2D | None = None

    print("[INFO] Line tracker controller started.")
    print(f"[INFO] config_path={cfg['_meta']['config_path']}")
    print(f"[INFO] camera={camera_name}, timestep={timestep} ms, dt={dt:.3f}s")
    print("[INFO] Current scope: planner v5 true DFS stack + ArUco marker mission, down_camera only.")
    print(
        "[INFO] planner: "
        f"enabled={planner.enabled}, priority={planner.priority}, "
        f"rearm={planner.node_rearm_distance_m:.2f} m, "
        f"map={planner.map.save_path}"
    )
    print("[INFO] v2 fixes: always TURN_TO_DIRECTION; LEAVE_NODE uses global direction + cross-track lock.")
    print("[INFO] debug: pseudo odometry map enabled inside debug_view.")
    print("[INFO] v6 fixes: anti-revisit node filter; entry-node reentry blocked during exploration.")
    print("[INFO] v7 fixes: dfs_complete triggers RETURN_HOME to entry_path.start_xy / initial pose.")
    print("[INFO] v8: split planners: DFS exploration planner + predefined-grid marker return planner.")
    print(f"[INFO] marker_return: enabled={marker_return_planner.enabled}, entry_node={marker_return_planner.entry_node_key()}")
    print(f"[INFO] aruco: enabled={aruco_memory.enabled}, expected_count={aruco_memory.expected_count}")

    last_log_time = time.time()
    frame_idx = 0

    try:
        while robot.step(timestep) != -1:
            bgr = read_processed_bgr(down_camera, cfg["camera"]["image_rotation"])
            if bgr is None:
                continue

            detection = detect_line(bgr, cfg)
            aruco_detection = detect_aruco(bgr, cfg)
            pose_before = motion.get_pose()
            new_aruco_records = aruco_memory.update(aruco_detection, pose_before, planner, frame_idx, state)
            for rec in new_aruco_records:
                print(
                    "[ARUCO] "
                    f"new id={rec.marker_id} node={rec.node_key} "
                    f"seen={aruco_memory.found_count}/{aruco_memory.expected_count or '?'} "
                    f"pose=({rec.x:+.2f},{rec.y:+.2f},{rec.yaw:+.2f})"
                )
            forward = 0.0
            yaw_rate = 0.0

            if state == STATE_CENTER_NODE:
                if detection.node_found:
                    forward, yaw_rate, centered = compute_node_centering_command(detection, cfg)
                    pose = motion.step(forward, yaw_rate, dt)
                    if centered:
                        centered_count += 1
                    else:
                        centered_count = 0

                    if centered_count >= hold_required:
                        centered_node_key, active_rel = register_centered_node_only(planner, pose, detection)

                        # Re-bind any marker visible at the exact centered node to the current node key.
                        new_at_node = aruco_memory.update(aruco_detection, pose, planner, frame_idx, "CENTER_NODE_CONFIRMED")
                        for rec in new_at_node:
                            print(
                                "[ARUCO] "
                                f"new id={rec.marker_id} node={rec.node_key} "
                                f"seen={aruco_memory.found_count}/{aruco_memory.expected_count or '?'} "
                                f"pose=({rec.x:+.2f},{rec.y:+.2f},{rec.yaw:+.2f})"
                            )

                        should_start_revisit = (
                            bool(cfg.get("aruco", {}).get("trigger_revisit_when_all_found", True))
                            and aruco_memory.complete
                            and revisit_mission.enabled
                            and not revisit_mission.started
                        )
                        if should_start_revisit:
                            # Stop DFS immediately and build a revisit list that contains
                            # ONLY ArUco marker nodes, not every DFS/topology node.
                            targets = build_reverse_marker_node_targets(aruco_memory, cfg)
                            revisit_mission.start(targets)
                            seq = " -> ".join(f"id={r.marker_id}@{r.node_key}" for r in targets)
                            print(f"[MISSION] All ArUco markers found. Reverse marker-node sequence: {seq}")

                        if revisit_mission.started:
                            # Phase 2 uses the separate grid-return planner.
                            # It deliberately ignores the DFS stack and the visited topology edges.
                            # Ordinary grid nodes may be reused; non-target ArUco marker nodes are blocked.
                            current_target = revisit_mission.current()
                            if (not revisit_mission.completed) and revisit_mission.is_non_current_marker_node(centered_node_key):
                                target_key = getattr(current_target, "node_key", "") if current_target is not None else ""
                                active_decision = PlannerDecision(
                                    action="STOP",
                                    current_node_key=centered_node_key,
                                    next_node_key=target_key,
                                    reason=(
                                        "illegal_non_target_marker_node_pass_through;"
                                        f"hit={centered_node_key};target={target_key}"
                                    ),
                                )
                                print(
                                    "[MISSION][ERROR] Non-target marker-node reached: "
                                    f"hit={centered_node_key}, target={target_key}"
                                )
                            else:
                                arrived = revisit_mission.advance_if_current_node(centered_node_key)
                                for rec in arrived:
                                    print(f"[MISSION] revisited marker-node id={rec.marker_id} at node={centered_node_key}")

                                if not revisit_mission.completed:
                                    target = revisit_mission.current()
                                    forbidden_nodes = revisit_mission.forbidden_node_keys_for_route(
                                        start_key=centered_node_key,
                                        goal_key=target.node_key,
                                    )
                                    active_decision = marker_return_planner.make_decision_to_node(
                                        centered_node_key,
                                        target.node_key,
                                        pose,
                                        reason=f"marker_return_grid_id={target.marker_id}",
                                        forbidden_nodes=forbidden_nodes,
                                    )
                                    commit_decision_departure(planner, active_decision, mark_attempted=False)
                                else:
                                    entry_key = marker_return_planner.entry_node_key()
                                    use_grid_to_entry = bool((cfg.get("return_home", {}) or {}).get("use_grid_route_to_entry_node", True))
                                    if use_grid_to_entry and entry_key and centered_node_key != entry_key:
                                        forbidden_nodes = revisit_mission.forbidden_node_keys_for_route(
                                            start_key=centered_node_key,
                                            goal_key=entry_key,
                                        )
                                        active_decision = marker_return_planner.make_decision_to_node(
                                            centered_node_key,
                                            entry_key,
                                            pose,
                                            reason="marker_return_grid_to_entry",
                                            forbidden_nodes=forbidden_nodes,
                                        )
                                        commit_decision_departure(planner, active_decision, mark_attempted=False)
                                    else:
                                        active_decision = PlannerDecision(
                                            action="STOP",
                                            current_node_key=centered_node_key,
                                            next_node_key=entry_key or "HOME",
                                            reason="marker_revisit_complete",
                                        )
                        else:
                            active_decision = choose_normal_dfs_decision(planner, centered_node_key, pose, active_rel)

                        planner_info = {
                            "decision": active_decision.relative_dir,
                            "global": active_decision.global_dir,
                            "reason": active_decision.reason,
                            "node": active_decision.current_node_key,
                            "next": active_decision.next_node_key,
                            "aruco": aruco_memory.status_text(),
                            "revisit": revisit_mission.status_text(),
                        }
                        print(
                            "[PLAN] "
                            f"node={active_decision.current_node_key} "
                            f"type={detection.node_type} dirs={list(detection.active_dirs)} "
                            f"decision={active_decision.relative_dir}/{active_decision.global_dir} "
                            f"reason={active_decision.reason} next={active_decision.next_node_key} "
                            f"{aruco_memory.status_text()} {revisit_mission.status_text()}"
                        )

                        if active_decision.action != "GO":
                            # Direct HOME movement is allowed only after the marker-return
                            # planner has already reached the grid-side entry node.  If the
                            # grid route itself fails, stop instead of falling back to the
                            # old DFS/visited-edge behavior.
                            can_direct_home_after_marker_return = (
                                revisit_mission.started
                                and revisit_mission.completed
                                and _return_home_enabled(cfg)
                                and (
                                    active_decision.reason == "marker_revisit_complete"
                                    or active_decision.reason.startswith("grid_route_already_at_target")
                                )
                            )
                            if can_direct_home_after_marker_return:
                                return_home_target = make_return_home_target(planner, initial_home_pose, motion, cfg)
                                state = STATE_RETURN_HOME_TURN
                                planner_info = {
                                    "decision": "return_home",
                                    "global": _global_dir_from_to(motion.get_pose(), return_home_target),
                                    "reason": "marker_return_grid_complete_return_home",
                                    "node": active_decision.current_node_key,
                                    "next": "HOME",
                                    "aruco": aruco_memory.status_text(),
                                    "revisit": revisit_mission.status_text(),
                                }
                            elif active_decision.reason == "dfs_complete" and _return_home_enabled(cfg):
                                return_home_target = make_return_home_target(planner, initial_home_pose, motion, cfg)
                                state = STATE_RETURN_HOME_TURN
                                planner_info = {
                                    "decision": "return_home",
                                    "global": _global_dir_from_to(motion.get_pose(), return_home_target),
                                    "reason": "dfs_complete_return_home",
                                    "node": active_decision.current_node_key,
                                    "next": "HOME",
                                    "aruco": aruco_memory.status_text(),
                                    "revisit": revisit_mission.status_text(),
                                }
                            else:
                                state = STATE_STOPPED_NO_PLAN
                        else:
                            # v2 fix #1: even if relative_dir == "front", force
                            # validation/realignment against the selected global yaw.
                            leave_start_pose = None
                            state = STATE_TURN_TO_DIRECTION
                        centered_count = 0
                # else:
                #     # While centering a node, do not fall back to normal line following.
                #     pose = motion.step(0.0, 0.0, dt)
                #     centered_count = 0

                else:
                    # CENTER_NODE 중 node를 놓친 경우:
                    # 라인이 아직 보이면 CENTER_NODE에 갇히지 말고 FOLLOW_LINE으로 복귀한다.
                    centered_count = 0

                    if detection.found:
                        forward, yaw_rate, state = compute_motion_command(detection, cfg)
                        pose = motion.step(forward, yaw_rate, dt)
                        planner_info = {
                            "decision": "resume_follow",
                            "global": active_decision.global_dir if active_decision else "",
                            "reason": "center_node_lost_but_line_found",
                            "node": active_decision.current_node_key if active_decision else "",
                            "next": active_decision.next_node_key if active_decision else "",
                        }
                    else:
                        state = STATE_LINE_LOST_STOP
                        forward = 0.0
                        yaw_rate = 0.0
                        pose = motion.step(0.0, 0.0, dt)
                        planner_info = {
                            "decision": "stop",
                            "global": "",
                            "reason": "center_node_lost_and_line_lost",
                            "node": active_decision.current_node_key if active_decision else "",
                            "next": active_decision.next_node_key if active_decision else "",
                        }

            elif state == STATE_TURN_TO_DIRECTION:
                target_yaw = normalize_angle(active_decision.target_yaw)
                pose_now = motion.get_pose()
                yaw_rate, turned, yaw_err = compute_turn_command(pose_now.yaw, target_yaw, cfg)
                pose = motion.step(0.0, yaw_rate, dt)
                planner_info = {
                    "decision": active_decision.relative_dir,
                    "global": active_decision.global_dir,
                    "reason": active_decision.reason,
                    "yaw_err": f"{yaw_err:+.3f}",
                    "node": active_decision.current_node_key,
                    "next": active_decision.next_node_key,
                }
                if turned:
                    # Snap yaw and optionally snap x/y to the canonical centered node.
                    leave_start_pose = _prepare_leave_start_pose(motion, planner, active_decision, cfg)
                    pose = leave_start_pose
                    state = STATE_LEAVE_NODE

            elif state == STATE_LEAVE_NODE:
                planner_cfg = cfg.get("planner", {})
                leave_speed = float(planner_cfg.get("leave_speed_mps", 0.18))
                leave_distance = float(planner_cfg.get("leave_distance_m", 0.45))
                if leave_start_pose is None:
                    leave_start_pose = _prepare_leave_start_pose(motion, planner, active_decision, cfg)

                # v2 fix #2: leave along the selected global grid direction.
                # This locks cross-track coordinate and prevents diagonal drift.
                pose = motion.step_global_direction(
                    active_decision.global_dir,
                    leave_speed,
                    dt,
                    anchor_x=leave_start_pose.x,
                    anchor_y=leave_start_pose.y,
                    force_yaw=True,
                )
                dist = traveled_along_global(pose, leave_start_pose, active_decision.global_dir)
                forward = leave_speed
                yaw_rate = 0.0
                planner_info = {
                    "decision": active_decision.relative_dir,
                    "global": active_decision.global_dir,
                    "reason": f"leaving_global {dist:.2f}/{leave_distance:.2f}m",
                    "node": active_decision.current_node_key,
                    "next": active_decision.next_node_key,
                    "anchor": f"({leave_start_pose.x:+.2f},{leave_start_pose.y:+.2f})",
                }
                if dist >= leave_distance:
                    state = STATE_FOLLOW_LINE
                    leave_start_pose = None

            elif state == STATE_BACKTRACK_TO_LAST_NODE:
                recovery_cfg = cfg.get("recovery", {})
                if backtrack_target is None:
                    backtrack_target = make_backtrack_target(planner, active_decision, motion)

                if backtrack_target is None:
                    state = STATE_LINE_LOST_STOP
                    pose = motion.step(0.0, 0.0, dt)
                else:
                    pose, arrived = motion.move_towards_pose(
                        backtrack_target,
                        linear_speed_mps=float(recovery_cfg.get("backtrack_speed_mps", 0.20)),
                        max_yaw_rate_radps=float(recovery_cfg.get("backtrack_max_yaw_rate", 1.2)),
                        dt=dt,
                        position_tolerance_m=float(recovery_cfg.get("backtrack_position_tolerance_m", 0.035)),
                        yaw_tolerance_rad=float(recovery_cfg.get("backtrack_yaw_tolerance_rad", 0.05)),
                    )
                    forward = 0.0
                    yaw_rate = 0.0
                    planner_info = {
                        "decision": "backtrack",
                        "global": opposite_global_dir(active_decision.global_dir) if active_decision.global_dir else "",
                        "reason": "line_lost_return_to_last_node",
                        "node": active_decision.current_node_key,
                        "next": active_decision.next_node_key,
                    }
                    if arrived:
                        # Let the same node be accepted immediately, then re-run planner there.
                        planner.force_rearm_node_detection()
                        backtrack_target = None
                        leave_start_pose = None
                        state = STATE_FOLLOW_LINE

            elif state == STATE_RETURN_HOME_TURN:
                if return_home_target is None:
                    return_home_target = make_return_home_target(planner, initial_home_pose, motion, cfg)
                pose_now = motion.get_pose()
                yaw_rate, turned, yaw_err = compute_turn_command(pose_now.yaw, return_home_target.yaw, cfg)
                pose = motion.step(0.0, yaw_rate, dt)
                planner_info = {
                    "decision": "return_home",
                    "global": _global_dir_from_to(pose_now, return_home_target),
                    "reason": f"turn_home yaw_err={yaw_err:+.3f}",
                    "node": planner.current_node_key,
                    "next": "HOME",
                }
                if turned:
                    state = STATE_RETURN_HOME_MOVE

            elif state == STATE_RETURN_HOME_MOVE:
                if return_home_target is None:
                    return_home_target = make_return_home_target(planner, initial_home_pose, motion, cfg)
                rh = cfg.get("return_home", {})
                pose, arrived = motion.move_towards_pose(
                    return_home_target,
                    linear_speed_mps=float(rh.get("speed_mps", 0.20)),
                    max_yaw_rate_radps=float(rh.get("max_yaw_rate", 1.2)),
                    dt=dt,
                    position_tolerance_m=float(rh.get("position_tolerance_m", 0.035)),
                    yaw_tolerance_rad=float(rh.get("yaw_tolerance_rad", 0.05)),
                )
                forward = float(rh.get("speed_mps", 0.20))
                yaw_rate = 0.0
                dist_home = math.hypot(return_home_target.x - pose.x, return_home_target.y - pose.y)
                planner_info = {
                    "decision": "return_home",
                    "global": _global_dir_from_to(pose, return_home_target),
                    "reason": f"moving_home dist={dist_home:.2f}",
                    "node": planner.current_node_key,
                    "next": "HOME",
                }
                if arrived:
                    state = STATE_RETURN_HOME_DONE
                    planner_info["reason"] = "home_reached"

            elif state == STATE_RETURN_HOME_DONE:
                pose = motion.step(0.0, 0.0, dt)
                planner_info = {
                    "decision": "return_home",
                    "global": "",
                    "reason": "home_reached",
                    "node": planner.current_node_key,
                    "next": "HOME",
                }

            elif state == STATE_STOPPED_NO_PLAN:
                pose = motion.step(0.0, 0.0, dt)

            else:
                # # FOLLOW_LINE or LINE_LOST_STOP. If a node is visible and we are
                # # far enough from the last centered node, center it first.
                # if detection.node_found and planner.should_accept_node(pose_before):
                #     state = STATE_CENTER_NODE
                #     centered_count = 0
                #     forward, yaw_rate, centered = compute_node_centering_command(detection, cfg)
                #     pose = motion.step(forward, yaw_rate, dt)
                #     if centered:
                #         centered_count = 1

                node_accept_min_fwd = float(
                    _node_cfg(cfg).get("accept_forward_error_min", -0.10)
                )

                node_is_in_front_or_near_center = (
                    float(detection.node_forward_error) >= node_accept_min_fwd
                )

                if (
                    detection.node_found
                    and node_is_in_front_or_near_center
                    and planner.should_accept_node(pose_before)
                ):
                    state = STATE_CENTER_NODE
                    centered_count = 0
                    forward, yaw_rate, centered = compute_node_centering_command(detection, cfg)
                    pose = motion.step(forward, yaw_rate, dt)
                    if centered:
                        centered_count = 1

                elif detection.found:
                    forward, yaw_rate, state = compute_motion_command(detection, cfg)
                    pose = motion.step(forward, yaw_rate, dt)
                else:
                    recovery_cfg = cfg.get("recovery", {})
                    can_backtrack = (
                        bool(recovery_cfg.get("backtrack_on_line_lost", True))
                        and active_decision.action == "GO"
                        and bool(active_decision.current_node_key)
                        and bool(active_decision.next_node_key)
                    )
                    if can_backtrack:
                        planner.mark_decision_failed(active_decision, reason="line_lost")
                        backtrack_target = make_backtrack_target(planner, active_decision, motion)
                        state = STATE_BACKTRACK_TO_LAST_NODE if backtrack_target is not None else STATE_LINE_LOST_STOP
                        planner_info = {
                            "decision": "backtrack",
                            "global": opposite_global_dir(active_decision.global_dir) if active_decision.global_dir else "",
                            "reason": "line_lost_mark_failed_edge",
                            "node": active_decision.current_node_key,
                            "next": active_decision.next_node_key,
                        }
                    else:
                        state = STATE_LINE_LOST_STOP
                    forward = 0.0
                    yaw_rate = 0.0
                    pose = motion.step(0.0, 0.0, dt)

            if bool(cfg.get("aruco", {}).get("draw_debug", True)):
                draw_aruco_observations(bgr, aruco_detection)
            keep_running = debug.update(
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
            if not keep_running:
                print("[INFO] Quit requested from debug window.")
                break

            now = time.time()
            if now - last_log_time > 1.0:
                node_text = "none"
                if detection.node_center is not None:
                    nx, ny = detection.node_center
                    node_text = (
                        f"type={detection.node_type} center=({nx},{ny}) "
                        f"lat={detection.node_lateral_error:+.3f} "
                        f"fwd={detection.node_forward_error:+.3f} "
                        f"dirs={list(detection.active_dirs)} "
                        f"area={detection.node_area} "
                        f"reason={detection.node_reason}"
                    )
                print(
                    f"[LINE] frame={frame_idx} state={state} found={detection.found} "
                    f"err={detection.error:+.3f} area={detection.mask_area} "
                    f"node_found={detection.node_found} {node_text} "
                    f"plan={planner_info} "
                    f"v={forward:.2f} yaw_rate={yaw_rate:+.2f} "
                    f"pose=({pose.x:+.2f},{pose.y:+.2f},{pose.yaw:+.2f}) "
                    f"{aruco_memory.status_text()} {revisit_mission.status_text()} "
                    f"hold={centered_count}/{hold_required}"
                )
                last_log_time = now
            frame_idx += 1
    finally:
        debug.close()
        planner.map.save()
        print("[INFO] Line tracker controller stopped.")


if __name__ == "__main__":
    main()
