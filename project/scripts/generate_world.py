#!/usr/bin/env python3
"""Generate a Webots world for a square camera-based line-tracker robot.

Design goal for the reset version:
  - keep the black grid lines
  - keep the optional entry road/path
  - remove the start pad
  - place the robot at the outer end of the entry road
  - rotate the robot so its local +X direction faces along the entry road
  - keep the robot Supervisor-friendly for kinematic debugging

Typical usage:
    python3 generate_world_entry_robot.py --config project/config/config.yaml --no-markers
    python3 generate_world_entry_robot.py --config project/config/config.yaml
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import random
import re
from typing import Any, Optional, Sequence, Tuple

import yaml


HEADER = """#VRML_SIM R2025a utf8

EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/backgrounds/protos/TexturedBackground.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/backgrounds/protos/TexturedBackgroundLight.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/floors/protos/RectangleArena.proto"

WorldInfo {
  basicTimeStep 32
}

Viewpoint {
  orientation -0.5267542163807309 0.30187841562043327 0.7946064546098395 2.2869564011379837
  position 10 -18 35
}

TexturedBackground {
}

TexturedBackgroundLight {
}
"""


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config file: {config_path}")
    return cfg


def get_section(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    value = cfg.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"config['{key}'] must be a dictionary")
    return value


def fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:.6f}".rstrip("0").rstrip(".")
    return str(v)


def fmt_vec(values: Sequence[Any]) -> str:
    return " ".join(fmt(v) for v in values)


def read_tuple(section: dict[str, Any], key: str, default: Sequence[float]) -> Tuple[float, ...]:
    value = section.get(key, default)
    if value is None:
        value = default
    return tuple(float(v) for v in value)


def read_color(section: dict[str, Any], key: str, default: Sequence[float]) -> Tuple[float, float, float]:
    value = read_tuple(section, key, default)
    if len(value) != 3:
        raise ValueError(f"'{key}' must contain exactly 3 values")
    return value  # type: ignore[return-value]


def clamp(v: float, low: float, high: float) -> float:
    return max(low, min(high, v))


def z_center_on_floor(thickness: float, requested_z: Optional[float] = None) -> float:
    """Return the center-z of a thin floor object.

    Webots Box translation is its center.  If old configs use z=0.0 to mean
    "on the floor", this converts it to thickness/2.
    """
    if requested_z is None or abs(requested_z) < 1e-12:
        return thickness / 2.0
    return requested_z


def yaw_to_rotation(yaw_rad: float) -> Tuple[float, float, float, float]:
    return (0.0, 0.0, 1.0, yaw_rad)


def yaw_between(start_xy: Sequence[float], target_xy: Sequence[float]) -> float:
    dx = float(target_xy[0]) - float(start_xy[0])
    dy = float(target_xy[1]) - float(start_xy[1])
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return 0.0
    return math.atan2(dy, dx)


# -----------------------------------------------------------------------------
# Basic world geometry
# -----------------------------------------------------------------------------

def make_arena(arena_cfg: dict[str, Any]) -> str:
    width = float(arena_cfg.get("width_m", 15.0))
    height = float(arena_cfg.get("height_m", 15.0))
    wall_height = float(arena_cfg.get("wall_height_m", 0.5))
    tile = float(arena_cfg.get("floor_tile_size_m", 1.0))

    return f"""
RectangleArena {{
  floorSize {fmt(width)} {fmt(height)}
  floorTileSize {fmt(tile)} {fmt(tile)}
  floorAppearance PBRAppearance {{
    baseColor 1 1 1
    roughness 1
    metalness 0
  }}
  wallHeight {fmt(wall_height)}
}}
"""


def make_box_solid(
    name: str,
    translation: Sequence[float],
    size: Sequence[float],
    color: Sequence[float] = (0, 0, 0),
    texture_url: Optional[str] = None,
    rotation: Optional[Sequence[float]] = None,
) -> str:
    rotation_text = f"\n  rotation {fmt_vec(rotation)}" if rotation is not None else ""

    if texture_url:
        appearance = f"""PBRAppearance {{
        baseColorMap ImageTexture {{
          url [ \"{texture_url}\" ]
        }}
        roughness 1
        metalness 0
      }}"""
    else:
        appearance = f"""PBRAppearance {{
        baseColor {fmt_vec(color)}
        roughness 1
        metalness 0
      }}"""

    return f"""
Solid {{{rotation_text}
  translation {fmt_vec(translation)}
  children [
    Shape {{
      appearance {appearance}
      geometry Box {{
        size {fmt_vec(size)}
      }}
    }}
  ]
  name \"{name}\"
}}
"""


def make_grid_lines(grid_cfg: dict[str, Any]) -> str:
    grid_w = float(grid_cfg.get("width_m", 8.0))
    grid_h = float(grid_cfg.get("height_m", 8.0))
    cols = int(grid_cfg.get("cols", 5))
    rows = int(grid_cfg.get("rows", 5))
    line_w = float(grid_cfg.get("line_width_m", 0.15))
    thickness = float(grid_cfg.get("line_thickness_m", 0.01))

    if cols <= 0 or rows <= 0:
        raise ValueError("grid.cols and grid.rows must be positive")

    parts: list[str] = []
    dx = grid_w / cols
    dy = grid_h / rows

    for j in range(rows + 1):
        y = -grid_h / 2.0 + j * dy
        parts.append(
            make_box_solid(
                name=f"line_h_{j}",
                translation=(0, y, z_center_on_floor(thickness)),
                size=(grid_w, line_w, thickness),
                color=(0, 0, 0),
            )
        )

    for i in range(cols + 1):
        x = -grid_w / 2.0 + i * dx
        parts.append(
            make_box_solid(
                name=f"line_v_{i}",
                translation=(x, 0, z_center_on_floor(thickness) + 0.0002),
                size=(line_w, grid_h, thickness),
                color=(0, 0, 0),
            )
        )

    return "".join(parts)


# -----------------------------------------------------------------------------
# Entry road/path
# -----------------------------------------------------------------------------

def compute_entry_path_segment(cfg: dict[str, Any]) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Return (outer_end_xy, grid_end_xy) for the entry road.

    The returned order is important:
      - outer_end_xy: where the robot starts
      - grid_end_xy: the direction the robot initially faces

    For manual paths:
      - entry_path.start_xy is treated as the robot/outer end
      - entry_path.end_xy is treated as the grid-facing end
    """
    entry_cfg = get_section(cfg, "entry_path")
    if not bool(entry_cfg.get("enabled", False)):
        return None

    grid_cfg = get_section(cfg, "grid")
    start_cfg = get_section(cfg, "start")

    grid_w = float(grid_cfg.get("width_m", 8.0))
    grid_h = float(grid_cfg.get("height_m", 8.0))

    # Entry road policy:
    # - The outer end is the robot start position.
    # - The grid-side end touches the grid boundary.
    # - It does NOT extend into the grid interior.
    #
    # The previous version applied overlap_m to the grid side, producing
    # ymin + overlap / ymax - overlap / xmin + overlap / xmax - overlap.
    # That created an unwanted black segment inside the grid.
    #
    # Keep optional extensions explicit:
    # - outer_extension_m: extend farther outward from entry_origin_xy
    # - grid_overlap_m: intentionally extend into the grid, default 0.0
    outer_extension = float(entry_cfg.get("outer_extension_m", 0.0))
    grid_overlap = float(entry_cfg.get("grid_overlap_m", 0.0))

    if "start_xy" in entry_cfg and "end_xy" in entry_cfg:
        start_xy = tuple(float(v) for v in entry_cfg["start_xy"][:2])
        end_xy = tuple(float(v) for v in entry_cfg["end_xy"][:2])
        if abs(start_xy[0] - end_xy[0]) > 1e-9 and abs(start_xy[1] - end_xy[1]) > 1e-9:
            raise ValueError("entry_path.start_xy -> end_xy must be horizontal or vertical")
        return start_xy, end_xy

    # Backward compatible fallback:
    # old configs used start.pad_translation to mean the port / entry origin.
    # If pad_translation is removed, use start.robot_translation or drone_translation xy.
    if "entry_origin_xy" in entry_cfg:
        px, py = (float(entry_cfg["entry_origin_xy"][0]), float(entry_cfg["entry_origin_xy"][1]))
    elif "pad_translation" in start_cfg:
        pad_translation = read_tuple(start_cfg, "pad_translation", (0, -5, 0))
        px, py = pad_translation[0], pad_translation[1]
    elif "robot_translation" in start_cfg:
        robot_translation = read_tuple(start_cfg, "robot_translation", (0, -5, 1))
        px, py = robot_translation[0], robot_translation[1]
    else:
        robot_translation = read_tuple(start_cfg, "drone_translation", (0, -5, 1))
        px, py = robot_translation[0], robot_translation[1]

    half_w = grid_w / 2.0
    half_h = grid_h / 2.0
    xmin, xmax = -half_w, half_w
    ymin, ymax = -half_h, half_h

    if py < ymin:
        x = clamp(px, xmin, xmax)
        outer_end = (x, py - outer_extension)
        grid_end = (x, ymin + grid_overlap)
    elif py > ymax:
        x = clamp(px, xmin, xmax)
        outer_end = (x, py + outer_extension)
        grid_end = (x, ymax - grid_overlap)
    elif px < xmin:
        y = clamp(py, ymin, ymax)
        outer_end = (px - outer_extension, y)
        grid_end = (xmin + grid_overlap, y)
    elif px > xmax:
        y = clamp(py, ymin, ymax)
        outer_end = (px + outer_extension, y)
        grid_end = (xmax - grid_overlap, y)
    else:
        print("[WARN] entry origin is inside the grid rectangle. Entry path skipped.")
        return None

    return outer_end, grid_end


def make_axis_aligned_line(
    name: str,
    start_xy: Sequence[float],
    end_xy: Sequence[float],
    line_width: float,
    thickness: float,
    color: Sequence[float] = (0, 0, 0),
    z_offset: float = 0.0006,
) -> str:
    x1, y1 = float(start_xy[0]), float(start_xy[1])
    x2, y2 = float(end_xy[0]), float(end_xy[1])
    eps = 1e-9

    if abs(x1 - x2) < eps:
        cx = x1
        cy = (y1 + y2) / 2.0
        sx = line_width
        sy = abs(y2 - y1)
    elif abs(y1 - y2) < eps:
        cx = (x1 + x2) / 2.0
        cy = y1
        sx = abs(x2 - x1)
        sy = line_width
    else:
        raise ValueError(f"Entry road must be horizontal or vertical: got ({x1}, {y1}) -> ({x2}, {y2})")

    if sx <= 0 or sy <= 0:
        return ""

    return make_box_solid(
        name=name,
        translation=(cx, cy, z_center_on_floor(thickness) + z_offset),
        size=(sx, sy, thickness),
        color=color,
    )


def make_entry_path(cfg: dict[str, Any]) -> str:
    entry_cfg = get_section(cfg, "entry_path")
    segment = compute_entry_path_segment(cfg)
    if segment is None:
        return ""

    grid_cfg = get_section(cfg, "grid")
    line_width = float(entry_cfg.get("line_width_m", grid_cfg.get("line_width_m", 0.15)))
    thickness = float(entry_cfg.get("thickness_m", grid_cfg.get("line_thickness_m", 0.01)))
    z_offset = float(entry_cfg.get("z_offset_m", 0.0006))
    color = read_color(entry_cfg, "color", (0, 0, 0))
    name = str(entry_cfg.get("name", "entry_path"))

    outer_end, grid_end = segment
    return make_axis_aligned_line(
        name=name,
        start_xy=outer_end,
        end_xy=grid_end,
        line_width=line_width,
        thickness=thickness,
        color=color,
        z_offset=z_offset,
    )


# -----------------------------------------------------------------------------
# ArUco marker placement
# -----------------------------------------------------------------------------

def vertex_to_world(i: int, j: int, grid_w: float, grid_h: float, cols: int, rows: int) -> Tuple[float, float]:
    dx = grid_w / cols
    dy = grid_h / rows
    x = -grid_w / 2.0 + i * dx
    y = -grid_h / 2.0 + j * dy
    return x, y


def all_grid_vertices(cols: int, rows: int) -> list[Tuple[int, int]]:
    return [(i, j) for i in range(cols + 1) for j in range(rows + 1)]


def marker_id_from_path(path: str) -> Optional[int]:
    name = os.path.basename(path)
    m = re.search(r"_(\d+)\.png$", name)
    return int(m.group(1)) if m else None


def marker_name_from_path(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def collect_marker_files(marker_cfg: dict[str, Any], texture_cfg: dict[str, Any]) -> list[str]:
    texture_dir = str(texture_cfg.get("save_dir", "project/textures/aruco_marker"))
    random_from_folder = bool(marker_cfg.get("random_select_from_folder", True))

    if random_from_folder:
        return sorted(glob.glob(os.path.join(texture_dir, "aruco_*.png")))

    dictionary = str(marker_cfg.get("dictionary", "DICT_4X4_50"))
    ids = marker_cfg.get("ids", []) or []
    files = [os.path.join(texture_dir, f"aruco_{dictionary}_{int(marker_id)}.png") for marker_id in ids]
    return [path for path in files if os.path.exists(path)]


def make_markers(cfg: dict[str, Any], world_dir: str, no_markers: bool = False) -> str:
    if no_markers:
        print("[INFO] ArUco markers disabled by --no-markers")
        return ""

    marker_cfg = get_section(cfg, "marker")
    texture_cfg = get_section(cfg, "texture")
    grid_cfg = get_section(cfg, "grid")

    count = int(marker_cfg.get("count", 0))
    if count <= 0:
        print("[INFO] marker.count <= 0. ArUco markers skipped.")
        return ""

    grid_w = float(grid_cfg.get("width_m", 8.0))
    grid_h = float(grid_cfg.get("height_m", 8.0))
    cols = int(grid_cfg.get("cols", 5))
    rows = int(grid_cfg.get("rows", 5))

    outer_size = float(marker_cfg.get("outer_size_m", 0.5))
    thickness = float(marker_cfg.get("thickness_m", 0.01))
    seed = marker_cfg.get("seed", None)
    rng = random.Random(seed)

    marker_files = collect_marker_files(marker_cfg, texture_cfg)
    if len(marker_files) < count:
        raise FileNotFoundError(
            f"marker.count={count}, but only {len(marker_files)} marker texture files were found. "
            f"Check texture.save_dir or use --no-markers for line-tracking-only tests."
        )

    selected_files = rng.sample(marker_files, count)
    forbidden = {tuple(map(int, v)) for v in (marker_cfg.get("forbidden_vertices", []) or [])}
    candidates = [v for v in all_grid_vertices(cols, rows) if v not in forbidden]
    if len(candidates) < count:
        raise ValueError(f"marker.count={count}, but only {len(candidates)} candidate grid vertices are available")

    selected_vertices = rng.sample(candidates, count)
    z = z_center_on_floor(thickness) + 0.002

    print("Selected ArUco markers:")
    parts: list[str] = []
    for marker_path, (i, j) in zip(selected_files, selected_vertices):
        x, y = vertex_to_world(i, j, grid_w, grid_h, cols, rows)
        rel_path = os.path.relpath(marker_path, world_dir).replace("\\", "/")
        marker_name = marker_name_from_path(marker_path)
        marker_id = marker_id_from_path(marker_path)
        id_text = f"id={marker_id}" if marker_id is not None else "id=?"
        print(f"  {os.path.basename(marker_path)} ({id_text}) -> vertex=({i},{j}) -> world=({x:.3f},{y:.3f})")

        parts.append(
            make_box_solid(
                name=marker_name,
                translation=(x, y, z),
                size=(outer_size, outer_size, thickness),
                texture_url=rel_path,
            )
        )

    return "".join(parts)


# -----------------------------------------------------------------------------
# Robot
# -----------------------------------------------------------------------------

def resolve_robot_pose(cfg: dict[str, Any]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    """Resolve robot translation and rotation.

    Default behavior:
      - if entry_path.enabled: place robot at the outer end of the entry road
      - yaw points from outer end toward the grid-side end
      - z is taken from start.robot_translation or old start.drone_translation
    """
    start_cfg = get_section(cfg, "start")
    robot_cfg = get_section(cfg, "robot")
    entry_cfg = get_section(cfg, "entry_path")

    base_translation = read_tuple(
        start_cfg,
        "robot_translation",
        start_cfg.get("drone_translation", (0.0, -5.0, 1.0)),
    )
    z = base_translation[2] if len(base_translation) >= 3 else 1.0

    auto_pose = bool(robot_cfg.get("auto_pose_from_entry", True))
    segment = compute_entry_path_segment(cfg) if auto_pose else None

    if segment is not None:
        outer_end, grid_end = segment
        start_mode = str(entry_cfg.get("robot_start_end", "outer")).lower()

        if start_mode in {"grid", "inner", "end"}:
            robot_xy = grid_end
            target_xy = outer_end
        else:
            robot_xy = outer_end
            target_xy = grid_end

        yaw = yaw_between(robot_xy, target_xy)
        return (robot_xy[0], robot_xy[1], z), yaw_to_rotation(yaw)

    robot_rotation = read_tuple(robot_cfg, "rotation", (0, 0, 1, 0))
    return (base_translation[0], base_translation[1], z), robot_rotation  # type: ignore[return-value]


def make_line_tracker_robot(cfg: dict[str, Any]) -> str:
    robot_cfg = get_section(cfg, "robot")
    camera_cfg = get_section(robot_cfg, "camera")

    def_name = str(robot_cfg.get("def_name", "LINE_TRACKER"))
    name = str(robot_cfg.get("name", "line_tracker"))
    controller = str(robot_cfg.get("controller", "line_tracker"))
    supervisor = bool(robot_cfg.get("supervisor", True))
    include_physics = bool(robot_cfg.get("include_physics", False))

    robot_translation, robot_rotation = resolve_robot_pose(cfg)

    body_size = read_tuple(robot_cfg, "body_size", (0.36, 0.36, 0.10))
    body_color = read_color(robot_cfg, "body_color", (0.15, 0.15, 0.18))

    front_marker_translation = read_tuple(robot_cfg, "front_marker_translation", (0.16, 0, 0.07))
    front_marker_size = read_tuple(robot_cfg, "front_marker_size", (0.12, 0.16, 0.03))
    front_marker_color = read_color(robot_cfg, "front_marker_color", (0.9, 0.2, 0.2))

    side_marker_translation = read_tuple(robot_cfg, "side_marker_translation", (0, 0.16, 0.07))
    side_marker_size = read_tuple(robot_cfg, "side_marker_size", (0.10, 0.04, 0.03))
    side_marker_color = read_color(robot_cfg, "side_marker_color", (0.2, 0.2, 0.9))
    show_side_marker = bool(robot_cfg.get("show_side_marker", True))

    front_cam_translation = read_tuple(robot_cfg, "front_camera_translation", (0.20, 0, 0.02))
    front_cam_rotation = read_tuple(robot_cfg, "front_camera_rotation", (0, 0, 1, 0))
    down_cam_translation = read_tuple(robot_cfg, "down_camera_translation", (0, 0, -0.14))
    down_cam_rotation = read_tuple(robot_cfg, "down_camera_rotation", (0, 1, 0, 1.5708))

    cam_width = int(camera_cfg.get("width", 640))
    cam_height = int(camera_cfg.get("height", 480))
    cam_fov = float(camera_cfg.get("fieldOfView", 1.2))
    cam_near = float(camera_cfg.get("near", 0.02))

    front_marker = f"""
    Transform {{
      translation {fmt_vec(front_marker_translation)}
      children [
        Shape {{
          appearance PBRAppearance {{
            baseColor {fmt_vec(front_marker_color)}
            roughness 1
            metalness 0
          }}
          geometry Box {{
            size {fmt_vec(front_marker_size)}
          }}
        }}
      ]
    }}"""

    side_marker = ""
    if show_side_marker:
        side_marker = f"""
    Transform {{
      translation {fmt_vec(side_marker_translation)}
      children [
        Shape {{
          appearance PBRAppearance {{
            baseColor {fmt_vec(side_marker_color)}
            roughness 1
            metalness 0
          }}
          geometry Box {{
            size {fmt_vec(side_marker_size)}
          }}
        }}
      ]
    }}"""

    physics = ""
    if include_physics:
        mass = float(robot_cfg.get("mass", 1.0))
        physics = f"""
  physics Physics {{
    mass {fmt(mass)}
  }}"""

    return f"""
DEF {def_name} Robot {{
  translation {fmt_vec(robot_translation)}
  rotation {fmt_vec(robot_rotation)}
  controller \"{controller}\"
  supervisor {fmt(supervisor)}

  children [
    Shape {{
      appearance PBRAppearance {{
        baseColor {fmt_vec(body_color)}
        roughness 1
        metalness 0
      }}
      geometry Box {{
        size {fmt_vec(body_size)}
      }}
    }}
{front_marker}
{side_marker}

    DEF FRONT_CAMERA Camera {{
      translation {fmt_vec(front_cam_translation)}
      rotation {fmt_vec(front_cam_rotation)}
      name \"front_camera\"
      width {fmt(cam_width)}
      height {fmt(cam_height)}
      fieldOfView {fmt(cam_fov)}
      near {fmt(cam_near)}
    }}

    DEF DOWN_CAMERA Camera {{
      translation {fmt_vec(down_cam_translation)}
      rotation {fmt_vec(down_cam_rotation)}
      name \"down_camera\"
      width {fmt(cam_width)}
      height {fmt(cam_height)}
      fieldOfView {fmt(cam_fov)}
      near {fmt(cam_near)}
    }}
  ]

  name \"{name}\"
  boundingObject Box {{
    size {fmt_vec(body_size)}
  }}{physics}
}}
"""


# -----------------------------------------------------------------------------
# World composition
# -----------------------------------------------------------------------------

def validate_config(cfg: dict[str, Any]) -> None:
    arena_cfg = get_section(cfg, "arena")
    grid_cfg = get_section(cfg, "grid")
    arena_w = float(arena_cfg.get("width_m", 15.0))
    arena_h = float(arena_cfg.get("height_m", 15.0))
    grid_w = float(grid_cfg.get("width_m", 8.0))
    grid_h = float(grid_cfg.get("height_m", 8.0))

    if grid_w > arena_w or grid_h > arena_h:
        raise ValueError(f"grid size ({grid_w}, {grid_h}) must be smaller than arena size ({arena_w}, {arena_h})")


def build_world(cfg: dict[str, Any], no_markers: bool = False, output_override: Optional[str] = None) -> str:
    validate_config(cfg)

    output_cfg = get_section(cfg, "output")
    out_path = output_override or str(output_cfg.get("world_path", "project/worlds/Drone_Project.wbt"))
    world_dir = os.path.dirname(out_path) or "."

    parts = [HEADER]
    parts.append(make_arena(get_section(cfg, "arena")))
    parts.append(make_grid_lines(get_section(cfg, "grid")))
    parts.append(make_markers(cfg, world_dir=world_dir, no_markers=no_markers))
    parts.append(make_entry_path(cfg))
    parts.append(make_line_tracker_robot(cfg))

    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Webots world for the square line-tracker robot.")
    parser.add_argument("--config", default="project/config/world.yaml", help="Path to config.yaml")
    parser.add_argument("--output", default=None, help="Optional output .wbt path override")
    parser.add_argument("--no-markers", action="store_true", help="Skip ArUco marker placement for line-tracking-only tests")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_path = args.output or str(get_section(cfg, "output").get("world_path", "project/worlds/Drone_Project.wbt"))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    world_text = build_world(cfg, no_markers=args.no_markers, output_override=out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(world_text)

    robot_translation, robot_rotation = resolve_robot_pose(cfg)
    print(f"World generated: {out_path}")
    print(f"Robot start translation: {fmt_vec(robot_translation)}")
    print(f"Robot start rotation: {fmt_vec(robot_rotation)}")
    print("Start pad: disabled / not generated")
    if args.no_markers:
        print("Mode: line-tracking only, ArUco markers skipped")


if __name__ == "__main__":
    main()
