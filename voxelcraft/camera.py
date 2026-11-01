"""Matrix maths and frustum culling.

Matrices are plain 4x4 NumPy arrays in row-major order (``v_clip = M @ v``),
uploaded to OpenGL with the transpose flag, so the maths here reads exactly
like the textbook formulas.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

# --- basic constructors ---------------------------------------------------
def identity() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Right-handed projection looking down -Z (OpenGL convention)."""
    f = 1.0 / math.tan(math.radians(fov_y_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = f / max(aspect, 1e-6)
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def look_at(eye, target, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    forward = target - eye
    norm = np.linalg.norm(forward)
    if norm < 1e-9:
        forward = np.array([0.0, 0.0, -1.0])
    else:
        forward /= norm
    right = np.cross(forward, up)
    rn = np.linalg.norm(right)
    if rn < 1e-6:  # looking straight up/down
        right = np.array([1.0, 0.0, 0.0])
    else:
        right /= rn
    true_up = np.cross(right, forward)

    m = np.eye(4, dtype=np.float64)
    m[0, :3] = right
    m[1, :3] = true_up
    m[2, :3] = -forward
    m[0, 3] = -right.dot(eye)
    m[1, 3] = -true_up.dot(eye)
    m[2, 3] = forward.dot(eye)
    return m


def translation(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[0, 3], m[1, 3], m[2, 3] = x, y, z
    return m


def scaling(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[0, 0], m[1, 1], m[2, 2] = x, y, z
    return m


def to_float32(m: np.ndarray) -> np.ndarray:
    """Row-major float32 copy suitable for ``glUniformMatrix4fv(..., GL_TRUE)``."""
    return np.ascontiguousarray(m, dtype=np.float32)


# --- camera / frustum -----------------------------------------------------
def direction_from_angles(yaw: float, pitch: float) -> np.ndarray:
    """Unit view direction. yaw=0 looks down -Z, +yaw turns right; pitch up is +."""
    cp = math.cos(pitch)
    return np.array(
        [math.sin(yaw) * cp, math.sin(pitch), -math.cos(yaw) * cp],
        dtype=np.float64,
    )


def horizontal_basis(yaw: float) -> tuple[np.ndarray, np.ndarray]:
    """(forward, right) on the XZ plane for a given yaw."""
    return (
        np.array([math.sin(yaw), 0.0, -math.cos(yaw)]),
        np.array([math.cos(yaw), 0.0, math.sin(yaw)]),
    )


class Frustum:
    """Six planes extracted from a view-projection matrix (Gribb/Hartmann)."""

    def __init__(self, matrix: np.ndarray):
        m = matrix
        rows = [m[0], m[1], m[2], m[3]]
        planes = np.empty((6, 4), dtype=np.float64)
        planes[0] = rows[3] + rows[0]      # left
        planes[1] = rows[3] - rows[0]      # right
        planes[2] = rows[3] + rows[1]      # bottom
        planes[3] = rows[3] - rows[1]      # top
        planes[4] = rows[3] + rows[2]      # near
        planes[5] = rows[3] - rows[2]      # far
        lengths = np.linalg.norm(planes[:, :3], axis=1, keepdims=True)
        lengths[lengths == 0] = 1.0
        self.planes = planes / lengths

    def intersects_aabb(self, min_corner: Iterable[float],
                        max_corner: Iterable[float]) -> bool:
        """``False`` only if the box is completely outside (cheap and conservative)."""
        mn = np.asarray(min_corner, dtype=np.float64)
        mx = np.asarray(max_corner, dtype=np.float64)
        p = self.planes
        # For each plane pick the box corner that is furthest along the normal.
        px = np.where(p[:, 0] >= 0, mx[0], mn[0])
        py = np.where(p[:, 1] >= 0, mx[1], mn[1])
        pz = np.where(p[:, 2] >= 0, mx[2], mn[2])
        dist = p[:, 0] * px + p[:, 1] * py + p[:, 2] * pz + p[:, 3]
        return bool(np.all(dist >= 0.0))
