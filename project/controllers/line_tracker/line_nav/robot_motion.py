"""Supervisor-based kinematic motion for the rectangular line tracker."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def shortest_angle_error(target: float, current: float) -> float:
    """Return target-current normalized to [-pi, pi]."""
    return normalize_angle(float(target) - float(current))


@dataclass
class Pose2D:
    x: float
    y: float
    z: float
    yaw: float


class KinematicRobotMotion:
    """Move the Webots Robot node by directly updating translation/rotation."""

    def __init__(self, supervisor, fixed_z: float = 1.0):
        self.supervisor = supervisor
        self.node = supervisor.getSelf()
        if self.node is None:
            raise RuntimeError("Supervisor.getSelf() returned None. Is supervisor TRUE set in the Robot node?")

        self.translation_field = self.node.getField("translation")
        self.rotation_field = self.node.getField("rotation")
        self.fixed_z = float(fixed_z)

    def get_pose(self) -> Pose2D:
        t = self.translation_field.getSFVec3f()
        r = self.rotation_field.getSFRotation()

        # Generated world uses rotation 0 0 1 yaw.
        yaw = float(r[3])
        if len(r) >= 3 and float(r[2]) < 0:
            yaw = -yaw
        yaw = normalize_angle(yaw)
        return Pose2D(float(t[0]), float(t[1]), float(t[2]), yaw)

    def set_pose(self, x: float, y: float, z: float, yaw: float) -> None:
        yaw = normalize_angle(float(yaw))
        self.translation_field.setSFVec3f([float(x), float(y), float(z)])
        self.rotation_field.setSFRotation([0.0, 0.0, 1.0, yaw])

    def step(self, forward_speed_mps: float, yaw_rate_radps: float, dt: float) -> Pose2D:
        """Body-frame forward motion. Used for normal line following/centering."""
        pose = self.get_pose()
        yaw = normalize_angle(pose.yaw + float(yaw_rate_radps) * float(dt))

        x = pose.x + float(forward_speed_mps) * math.cos(yaw) * float(dt)
        y = pose.y + float(forward_speed_mps) * math.sin(yaw) * float(dt)
        z = self.fixed_z

        self.set_pose(x, y, z, yaw)
        return Pose2D(x, y, z, yaw)

    def step_global_direction(
        self,
        global_dir: str,
        speed_mps: float,
        dt: float,
        anchor_x: float | None = None,
        anchor_y: float | None = None,
        force_yaw: bool = True,
    ) -> Pose2D:
        """Move along the arena-grid direction rather than current body yaw.

        This is used in LEAVE_NODE. It prevents diagonal departures by locking
        the cross-track coordinate to the centered node coordinate:

        - N/S: keep x = anchor_x
        - E/W: keep y = anchor_y
        """
        pose = self.get_pose()
        step = max(0.0, float(speed_mps)) * float(dt)
        d = str(global_dir).upper()

        x, y = pose.x, pose.y
        if d == "E":
            x = pose.x + step
            if anchor_y is not None:
                y = float(anchor_y)
            yaw = 0.0
        elif d == "W":
            x = pose.x - step
            if anchor_y is not None:
                y = float(anchor_y)
            yaw = math.pi
        elif d == "N":
            if anchor_x is not None:
                x = float(anchor_x)
            y = pose.y + step
            yaw = math.pi / 2.0
        elif d == "S":
            if anchor_x is not None:
                x = float(anchor_x)
            y = pose.y - step
            yaw = -math.pi / 2.0
        else:
            # Unknown direction: stay still instead of drifting.
            yaw = pose.yaw

        if not force_yaw:
            yaw = pose.yaw

        self.set_pose(x, y, self.fixed_z, yaw)
        return Pose2D(x, y, self.fixed_z, normalize_angle(yaw))

    def move_towards_pose(
        self,
        target: Pose2D,
        linear_speed_mps: float,
        max_yaw_rate_radps: float,
        dt: float,
        position_tolerance_m: float = 0.02,
        yaw_tolerance_rad: float = 0.03,
    ) -> Tuple[Pose2D, bool]:
        """Move toward a stored safe pose and return (new_pose, arrived)."""
        pose = self.get_pose()
        dx = float(target.x) - pose.x
        dy = float(target.y) - pose.y
        dist = math.hypot(dx, dy)

        yaw_err = shortest_angle_error(target.yaw, pose.yaw)

        if dist <= position_tolerance_m and abs(yaw_err) <= yaw_tolerance_rad:
            self.set_pose(target.x, target.y, self.fixed_z, target.yaw)
            return Pose2D(target.x, target.y, self.fixed_z, target.yaw), True

        max_step = max(0.0, float(linear_speed_mps)) * float(dt)
        if dist > 1e-9 and max_step > 0.0:
            step = min(dist, max_step)
            x = pose.x + dx / dist * step
            y = pose.y + dy / dist * step
        else:
            x, y = pose.x, pose.y

        max_yaw_step = max(0.0, float(max_yaw_rate_radps)) * float(dt)
        if abs(yaw_err) <= max_yaw_step or max_yaw_step <= 0.0:
            yaw = target.yaw
        else:
            yaw = normalize_angle(pose.yaw + math.copysign(max_yaw_step, yaw_err))

        self.set_pose(x, y, self.fixed_z, yaw)
        new_pose = Pose2D(x, y, self.fixed_z, yaw)

        remaining_dist = math.hypot(target.x - x, target.y - y)
        remaining_yaw = abs(shortest_angle_error(target.yaw, yaw))
        arrived = remaining_dist <= position_tolerance_m and remaining_yaw <= yaw_tolerance_rad
        if arrived:
            self.set_pose(target.x, target.y, self.fixed_z, target.yaw)
            new_pose = Pose2D(target.x, target.y, self.fixed_z, target.yaw)

        return new_pose, arrived


def compute_motion_command(detection, cfg) -> Tuple[float, float, str]:
    """Convert line detection result to forward speed and yaw rate."""
    ctrl = cfg["control"]
    lost = cfg["lost_line"]

    if detection.found:
        forward = float(ctrl["forward_speed_mps"])
        yaw_rate = (
            float(ctrl["yaw_sign"])
            * float(ctrl["kp_yaw"])
            * float(detection.error)
        )
        max_yaw = float(ctrl["max_yaw_rate"])
        yaw_rate = clamp(yaw_rate, -max_yaw, max_yaw)
        state = "FOLLOW_LINE"
        return forward, yaw_rate, state

    if bool(lost.get("search_when_lost", False)) and not bool(lost.get("stop_when_lost", True)):
        forward = 0.0
        yaw_rate = float(lost.get("search_yaw_rate", 0.25))
        state = "SEARCH_LINE"
        return forward, yaw_rate, state

    return 0.0, 0.0, "LINE_LOST_STOP"
