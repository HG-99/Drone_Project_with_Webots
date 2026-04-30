"""Camera helpers for Webots Camera -> OpenCV image conversion."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def enable_camera(robot, camera_name: str, timestep: int):
    """Get and enable a Webots camera device."""
    camera = robot.getDevice(camera_name)
    if camera is None:
        raise RuntimeError(f"Camera device not found: {camera_name}")
    camera.enable(timestep)
    return camera


def read_camera_bgr(camera) -> Optional[np.ndarray]:
    """Read a Webots Camera frame as BGR image.

    Webots camera buffers are commonly exposed as BGRA bytes. OpenCV uses BGR,
    so we drop the alpha channel.
    """
    raw = camera.getImage()
    if raw is None:
        return None

    width = camera.getWidth()
    height = camera.getHeight()
    frame = np.frombuffer(raw, dtype=np.uint8)

    expected = width * height * 4
    if frame.size != expected:
        raise RuntimeError(
            f"Unexpected camera buffer size: got {frame.size}, expected {expected} "
            f"for {width}x{height}x4"
        )

    bgra = frame.reshape((height, width, 4))
    bgr = bgra[:, :, :3].copy()
    return bgr


def rotate_image(image: np.ndarray, mode: str) -> np.ndarray:
    """Rotate image according to config value."""
    if mode == "none":
        return image
    if mode == "rot90_cw":
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if mode == "rot90_ccw":
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if mode == "rot180":
        return cv2.rotate(image, cv2.ROTATE_180)
    raise ValueError(f"Unsupported image_rotation mode: {mode}")


def read_processed_bgr(camera, image_rotation: str) -> Optional[np.ndarray]:
    """Read and rotate the camera image."""
    bgr = read_camera_bgr(camera)
    if bgr is None:
        return None
    return rotate_image(bgr, image_rotation)
