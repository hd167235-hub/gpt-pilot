"""First person player: look angles, AABB voxel physics and block interaction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import blocks as B
from .camera import direction_from_angles, horizontal_basis
from .constants import (
    FLY_SPEED,
    GRAVITY,
    JUMP_SPEED,
    PHYSICS_STEP,
    PLAYER_EYE,
    PLAYER_HEIGHT,
    PLAYER_WIDTH,
    REACH,
    SPRINT_SPEED,
    SWIM_SPEED,
    TERMINAL_VELOCITY,
    WALK_SPEED,
    WORLD_HEIGHT,
)
from .world import RaycastHit

_EPS = 1e-4


@dataclass
class MoveInput:
    """Player intentions for one frame."""

    forward: float = 0.0     # +1 = away from the camera, -1 = towards it
    strafe: float = 0.0      # +1 = right
    jump: bool = False
    sneak: bool = False
    sprint: bool = False
    up: bool = False         # fly up / swim up
    down: bool = False       # fly down


class Player:
    """A capsule-ish AABB walking on a voxel grid.

    ``position`` is the centre of the feet (the bottom face of the AABB), so the
    camera sits at ``position.y + eye_height``.
    """

    def __init__(self, world, position=(0.5, 70.0, 0.5), yaw: float = 0.0,
                 pitch: float = 0.0):
        self.world = world
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)
        self.yaw = float(yaw)
        self.pitch = float(pitch)
        self.on_ground = False
        self.flying = False
        self.in_water = False
        self.head_in_water = False
        self.sprinting = False
        self.sneaking = False
        self.eye_height = PLAYER_EYE
        self.height = PLAYER_HEIGHT
        self.width = PLAYER_WIDTH
        self.selected_block = B.GRASS
        self.spawn = np.array(position, dtype=np.float64)
        self._step_accumulator = 0.0

    # -- view -----------------------------------------------------------
    @property
    def eye_position(self) -> np.ndarray:
        return self.position + np.array([0.0, self.eye_height, 0.0])

    def look_direction(self) -> np.ndarray:
        return direction_from_angles(self.yaw, self.pitch)

    def look_at(self, target) -> None:
        """Point the camera at a world position (used by tests / the photo mode)."""
        d = np.asarray(target, dtype=np.float64) - self.eye_position
        n = np.linalg.norm(d)
        if n < 1e-9:
            return
        d /= n
        self.yaw = math.atan2(d[0], -d[2])
        self.pitch = math.asin(max(-1.0, min(1.0, d[1])))

    def rotate(self, dyaw: float, dpitch: float) -> None:
        """Mouse look with the pitch clamped to straight up/down."""
        self.yaw = (self.yaw + dyaw) % (2.0 * math.pi)
        limit = math.pi / 2 - 1e-3
        self.pitch = max(-limit, min(limit, self.pitch + dpitch))

    # -- AABB helpers ----------------------------------------------------
    def _extents(self):
        """(lo_offset, hi_offset) of the AABB for each axis, relative to position."""
        half = self.width / 2.0
        return (-half, half), (0.0, self.height), (-half, half)

    def _collides(self, position: np.ndarray) -> bool:
        (xlo, xhi), (ylo, yhi), (zlo, zhi) = self._extents()
        x0, y0, z0 = position[0] + xlo, position[1] + ylo, position[2] + zlo
        x1, y1, z1 = position[0] + xhi, position[1] + yhi, position[2] + zhi
        world = self.world
        for bx in range(int(math.floor(x0 + _EPS)), int(math.floor(x1 - _EPS)) + 1):
            for by in range(int(math.floor(y0 + _EPS)), int(math.floor(y1 - _EPS)) + 1):
                for bz in range(int(math.floor(z0 + _EPS)), int(math.floor(z1 - _EPS)) + 1):
                    if world.is_solid(bx, by, bz):
                        return True
        return False

    def intersects_block(self, block: tuple[int, int, int]) -> bool:
        """True if the player's box overlaps the given block cell."""
        (xlo, xhi), (ylo, yhi), (zlo, zhi) = self._extents()
        x0, y0, z0 = self.position + np.array([xlo, ylo, zlo])
        x1, y1, z1 = self.position + np.array([xhi, yhi, zhi])
        bx, by, bz = block
        return not (x1 <= bx or x0 >= bx + 1 or y1 <= by or y0 >= by + 1
                    or z1 <= bz or z0 >= bz + 1)

    # -- simulation ------------------------------------------------------
    def update(self, dt: float, move: MoveInput) -> None:
        """Advance the player with fixed physics sub-steps."""
        dt = min(dt, 0.25)
        self._step_accumulator += dt
        steps = 0
        while self._step_accumulator >= PHYSICS_STEP and steps < 12:
            self._step(PHYSICS_STEP, move)
            self._step_accumulator -= PHYSICS_STEP
            steps += 1
        if steps == 12:  # avoid a spiral of death on very slow frames
            self._step_accumulator = 0.0

    def _step(self, dt: float, move: MoveInput) -> None:
        self._update_environment()
        self.sprinting = move.sprint and not self.sneaking and not self.in_water

        # --- desired horizontal velocity -------------------------------
        forward, right = horizontal_basis(self.yaw)
        wish = right * move.strafe + forward * move.forward
        length = np.linalg.norm(wish)
        if length > 1e-6:
            wish /= length
        else:
            wish[:] = 0.0

        if self.flying:
            speed = FLY_SPEED * (1.6 if self.sprinting else 1.0)
        elif self.in_water:
            speed = SWIM_SPEED
        else:
            speed = SPRINT_SPEED if self.sprinting else WALK_SPEED
            if self.sneaking:
                speed = WALK_SPEED * 0.35

        target = wish * speed
        # Accelerate towards the target velocity (air control is weaker).
        if self.on_ground or self.flying or self.in_water:
            rate = 18.0 if self.in_water else 16.0
        else:
            rate = 5.0
        blend = min(1.0, rate * dt)
        self.velocity[0] += (target[0] - self.velocity[0]) * blend
        self.velocity[2] += (target[2] - self.velocity[2]) * blend

        # --- vertical ---------------------------------------------------
        if self.flying:
            vertical = 0.0
            if move.jump or move.up:
                vertical += 1.0
            if move.down or move.sneak:
                vertical -= 1.0
            self.velocity[1] = vertical * speed
        elif self.in_water:
            if move.jump or move.up:
                self.velocity[1] = 3.2
            else:
                self.velocity[1] = max(self.velocity[1] - GRAVITY * 0.25 * dt, -1.6)
        else:
            self.velocity[1] -= GRAVITY * dt
            if self.velocity[1] < -TERMINAL_VELOCITY:
                self.velocity[1] = -TERMINAL_VELOCITY
            if move.jump and self.on_ground:
                self.velocity[1] = JUMP_SPEED

        # --- integrate with collision ----------------------------------
        self.on_ground = False
        for axis in (0, 1, 2):
            delta = self.velocity[axis] * dt
            if abs(delta) < 1e-9:
                continue
            self._move_axis(axis, delta)

        # --- safety net -------------------------------------------------
        if self.position[1] < -24.0:
            self.respawn()
        if self.position[1] > WORLD_HEIGHT + 40:
            self.position[1] = float(WORLD_HEIGHT + 40)
            self.velocity[1] = min(self.velocity[1], 0.0)

    def _move_axis(self, axis: int, delta: float) -> None:
        """Move along one axis, snapping to the block face on collision."""
        test = self.position.copy()
        test[axis] += delta
        if not self._collides(test):
            self.position[axis] = test[axis]
            return

        (xlo, xhi), (ylo, yhi), (zlo, zhi) = self._extents()
        lo_offset = (xlo, ylo, zlo)[axis]
        hi_offset = (xhi, yhi, zhi)[axis]
        value = test[axis]
        if delta > 0:
            limit = math.floor(value + hi_offset)
            self.position[axis] = limit - hi_offset - _EPS
        else:
            limit = math.floor(value + lo_offset) + 1
            self.position[axis] = limit - lo_offset + _EPS
        self.velocity[axis] = 0.0
        if axis == 1 and delta < 0:
            self.on_ground = True

    def _update_environment(self) -> None:
        p = self.position
        feet = self.world.get_block(int(math.floor(p[0])), int(math.floor(p[1] + 0.2)),
                                    int(math.floor(p[2])))
        waist = self.world.get_block(int(math.floor(p[0])), int(math.floor(p[1] + 0.9)),
                                     int(math.floor(p[2])))
        head = self.world.get_block(int(math.floor(p[0])),
                                    int(math.floor(p[1] + self.eye_height)),
                                    int(math.floor(p[2])))
        self.in_water = B.BY_ID[feet].liquid or B.BY_ID[waist].liquid
        self.head_in_water = B.BY_ID[head].liquid
        self.eye_height += ((PLAYER_EYE - 0.12 if self.sneaking else PLAYER_EYE)
                            - self.eye_height) * 0.25

    # -- interaction -----------------------------------------------------
    def raycast(self, max_distance: float = REACH) -> Optional[RaycastHit]:
        return self.world.raycast(self.eye_position, self.look_direction(), max_distance)

    def break_block(self) -> Optional[RaycastHit]:
        """Left click: remove the targeted block (bedrock is immune)."""
        hit = self.raycast()
        if hit is None:
            return None
        if not B.BY_ID[hit.block_id].breakable:
            return hit
        if self.world.set_block(*hit.block, B.AIR):
            return hit
        return None

    def place_block(self, block_id: Optional[int] = None) -> Optional[tuple[int, int, int]]:
        """Right click: put a block against the targeted face."""
        block_id = self.selected_block if block_id is None else block_id
        if block_id == B.AIR:
            return None
        hit = self.raycast()
        if hit is None:
            return None
        target = hit.placement
        if target[1] < 0 or target[1] >= WORLD_HEIGHT:
            return None
        existing = self.world.get_block(*target)
        defn = B.BY_ID[existing]
        if not (existing == B.AIR or defn.liquid):
            return None
        if B.BY_ID[block_id].solid and self.intersects_block(target):
            return None
        if self.world.set_block(*target, block_id):
            return target
        return None

    # -- misc ------------------------------------------------------------
    def toggle_fly(self) -> bool:
        self.flying = not self.flying
        if self.flying:
            self.velocity[1] = 0.0
        return self.flying

    def respawn(self) -> None:
        self.position = self.spawn.copy()
        self.velocity[:] = 0.0
        self.on_ground = False

    def teleport(self, position) -> None:
        self.position = np.array(position, dtype=np.float64)
        self.velocity[:] = 0.0

    def place_on_surface(self, x: Optional[float] = None, z: Optional[float] = None) -> None:
        """Drop the player onto the highest *solid* block at ``x, z``."""
        x = self.position[0] if x is None else x
        z = self.position[2] if z is None else z
        bx, bz = int(math.floor(x)), int(math.floor(z))
        y = self.world.top_solid_y(bx, bz, default=60)
        while y > 0 and not self.world.is_solid(bx, y, bz):
            y -= 1
        self.position = np.array([x, y + 1.02, z], dtype=np.float64)
        self.velocity[:] = 0.0
        self.spawn = self.position.copy()

    @property
    def position_tuple(self) -> tuple[float, float, float]:
        return (float(self.position[0]), float(self.position[1]), float(self.position[2]))
