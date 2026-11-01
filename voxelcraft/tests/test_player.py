"""Tests for the first person controller: physics, jumping, collisions, blocks."""

from __future__ import annotations

import math
import unittest

import numpy as np

from voxelcraft import blocks as B
from voxelcraft.constants import (
    GRAVITY,
    JUMP_SPEED,
    PLAYER_HEIGHT,
    REACH,
)
from voxelcraft.player import MoveInput, Player
from voxelcraft.world import World


def flat_world(level: int = 10, radius: int = 2) -> World:
    """A flat stone plateau with air above it."""
    world = World(seed=5)
    for cz in range(-radius, radius + 1):
        for cx in range(-radius, radius + 1):
            world.generate_chunk(cx, cz)
    for chunk in world.iter_chunks():
        chunk.blocks[:] = B.AIR
        chunk.blocks[:, :level, :] = B.STONE
        chunk.blocks[:, 0, :] = B.BEDROCK
    return world


def simulate(player: Player, seconds: float, move: MoveInput | None = None,
             dt: float = 1 / 60.0) -> None:
    move = move or MoveInput()
    steps = int(seconds / dt)
    for _ in range(steps):
        player.update(dt, move)


class SpawnTests(unittest.TestCase):
    def test_eye_height(self):
        player = Player(flat_world(), position=(0.5, 11.0, 0.5))
        self.assertAlmostEqual(player.eye_position[1], 11.0 + player.eye_height)

    def test_place_on_surface(self):
        world = flat_world(level=12)          # stone fills y = 0..11
        player = Player(world, position=(0.5, 40.0, 0.5))
        player.place_on_surface()
        self.assertAlmostEqual(player.position[1], 12.02, places=2)

    def test_look_at(self):
        player = Player(flat_world(), position=(0.5, 20.0, 0.5))
        eye = player.eye_position
        player.look_at(eye + np.array([10.0, 0.0, 0.0]))
        np.testing.assert_allclose(player.look_direction(), [1.0, 0.0, 0.0], atol=1e-6)
        player.look_at(eye + np.array([0.0, 0.0, -10.0]))
        np.testing.assert_allclose(player.look_direction(), [0.0, 0.0, -1.0], atol=1e-6)
        player.look_at(eye + np.array([0.0, 10.0, 0.0]))
        np.testing.assert_allclose(player.look_direction(), [0.0, 1.0, 0.0], atol=1e-3)


class GravityTests(unittest.TestCase):
    def test_falls_and_lands(self):
        world = flat_world(level=10)
        player = Player(world, position=(0.5, 20.0, 0.5))
        self.assertFalse(player.on_ground)
        simulate(player, 3.0)
        self.assertTrue(player.on_ground)
        self.assertAlmostEqual(player.position[1], 10.0, delta=0.05)
        self.assertAlmostEqual(player.velocity[1], 0.0, delta=1e-6)

    def test_acceleration_matches_gravity(self):
        player = Player(flat_world(level=1), position=(0.5, 60.0, 0.5))
        start = player.velocity[1]
        player.update(0.1, MoveInput())
        self.assertLess(player.velocity[1], start)
        self.assertAlmostEqual(player.velocity[1] - start, -GRAVITY * 0.1, delta=1.5)

    def test_does_not_fall_through_the_floor(self):
        player = Player(flat_world(level=10), position=(0.5, 11.0, 0.5))
        simulate(player, 5.0)
        self.assertGreaterEqual(player.position[1], 10.0 - 1e-3)

    def test_ceiling_stops_the_player(self):
        world = flat_world(level=1)
        for chunk in world.iter_chunks():
            chunk.blocks[:, 5, :] = B.STONE        # ceiling
        player = Player(world, position=(0.5, 2.0, 0.5))
        player.velocity[1] = 6.0
        simulate(player, 0.5)
        # the head (y + 1.8) must stay below the ceiling at y = 5
        self.assertLessEqual(player.position[1] + PLAYER_HEIGHT, 5.0 + 1e-3)


class JumpTests(unittest.TestCase):
    def test_jump_leaves_the_ground(self):
        player = Player(flat_world(level=10), position=(0.5, 10.0, 0.5))
        simulate(player, 0.2)                              # settle
        self.assertTrue(player.on_ground)
        player.update(1 / 60, MoveInput(jump=True))
        self.assertGreater(player.velocity[1], 0)
        simulate(player, 0.25, MoveInput(jump=True))
        self.assertGreater(player.position[1], 10.5)

    def test_jump_height_is_about_one_block(self):
        player = Player(flat_world(level=10), position=(0.5, 10.0, 0.5))
        simulate(player, 0.2)
        peak = player.position[1]
        # hold jump so the player leaves the ground immediately
        for _ in range(120):
            player.update(1 / 120, MoveInput(jump=True))
            peak = max(peak, player.position[1])
        height = peak - 10.0
        self.assertGreater(height, 0.9)
        self.assertLess(height, 2.0)

    def test_cannot_jump_in_mid_air(self):
        player = Player(flat_world(level=10), position=(0.5, 14.0, 0.5))
        player.update(1 / 60, MoveInput(jump=True))
        self.assertLess(player.velocity[1], JUMP_SPEED)     # still only gravity

    def test_jump_onto_a_block(self):
        """A one block step must be climbable by jumping."""
        world = flat_world(level=10)
        world.set_block(2, 10, 0, B.STONE)                 # step block (top at y=11)
        player = Player(world, position=(0.5, 10.0, 0.5), yaw=math.pi / 2)  # +X
        simulate(player, 0.3)
        for _ in range(90):
            player.update(1 / 60, MoveInput(forward=1.0, jump=True))
        self.assertGreaterEqual(player.position[1], 10.9)

    def test_walking_into_a_wall_stops(self):
        world = flat_world(level=10)
        for chunk in world.iter_chunks():
            chunk.blocks[:, 10:13, :] = B.STONE           # a 3 high wall
        # carve a corridor so the player only hits the wall
        for chunk in world.iter_chunks():
            chunk.blocks[0:2, 10:13, 0:2] = B.AIR
        player = Player(world, position=(0.5, 10.0, 0.5), yaw=math.pi / 2)
        simulate(player, 2.0, MoveInput(forward=1.0))
        self.assertLess(player.position[0], 3.0)
        self.assertAlmostEqual(player.position[1], 10.0, delta=0.05)


class MovementTests(unittest.TestCase):
    def test_walks_forward(self):
        player = Player(flat_world(level=10), position=(0.5, 10.0, 0.5), yaw=0.0)
        simulate(player, 1.0, MoveInput(forward=1.0))
        # yaw 0 looks down -Z
        self.assertLess(player.position[2], 0.0)
        self.assertAlmostEqual(player.position[0], 0.5, delta=0.05)

    def test_strafes_right(self):
        player = Player(flat_world(level=10), position=(0.5, 10.0, 0.5), yaw=0.0)
        simulate(player, 0.5, MoveInput(strafe=1.0))
        self.assertGreater(player.position[0], 1.0)

    def test_walk_speed_is_limited(self):
        player = Player(flat_world(level=10), position=(0.5, 10.0, 0.5), yaw=1.2)
        simulate(player, 3.0, MoveInput(forward=1.0))
        speed = math.hypot(player.velocity[0], player.velocity[2])
        self.assertLess(speed, 6.0)
        self.assertGreater(speed, 2.0)

    def test_sprint_is_faster(self):
        slow = Player(flat_world(level=10), position=(0.5, 10.0, 0.5), yaw=1.2)
        fast = Player(flat_world(level=10), position=(0.5, 10.0, 0.5), yaw=1.2)
        simulate(slow, 2.0, MoveInput(forward=1.0))
        simulate(fast, 2.0, MoveInput(forward=1.0, sprint=True))
        self.assertGreater(math.hypot(fast.velocity[0], fast.velocity[2]),
                           math.hypot(slow.velocity[0], slow.velocity[2]))

    def test_sneak_is_slower(self):
        player = Player(flat_world(level=10), position=(0.5, 10.0, 0.5), yaw=1.2)
        player.sneaking = True
        simulate(player, 2.0, MoveInput(forward=1.0, sneak=True))
        self.assertLess(math.hypot(player.velocity[0], player.velocity[2]), 3.0)

    def test_diagonal_input_is_normalised(self):
        a = Player(flat_world(level=10), position=(0.5, 10.0, 0.5), yaw=0.0)
        b = Player(flat_world(level=10), position=(0.5, 10.0, 0.5), yaw=0.0)
        simulate(a, 1.0, MoveInput(forward=1.0))
        simulate(b, 1.0, MoveInput(forward=1.0, strafe=1.0))
        speed_a = math.hypot(a.velocity[0], a.velocity[2])
        speed_b = math.hypot(b.velocity[0], b.velocity[2])
        self.assertAlmostEqual(speed_a, speed_b, delta=0.15)

    def test_rotation_is_clamped(self):
        player = Player(flat_world(), position=(0.5, 11.0, 0.5))
        player.rotate(0.0, 10.0)
        self.assertLessEqual(player.pitch, math.pi / 2)
        player.rotate(0.0, -20.0)
        self.assertGreaterEqual(player.pitch, -math.pi / 2)
        player.rotate(2 * math.pi * 3 + 0.5, 0.0)
        self.assertLess(player.yaw, 2 * math.pi)


class FlightAndSwimTests(unittest.TestCase):
    def test_fly_hovers(self):
        player = Player(flat_world(level=10), position=(0.5, 30.0, 0.5))
        player.toggle_fly()
        simulate(player, 1.0, MoveInput(jump=True))
        self.assertGreater(player.position[1], 30.0)

    def test_fly_does_not_fall(self):
        player = Player(flat_world(level=10), position=(0.5, 30.0, 0.5))
        player.toggle_fly()
        simulate(player, 2.0)
        self.assertAlmostEqual(player.position[1], 30.0, delta=0.2)

    def test_water_slows_the_fall(self):
        world = flat_world(level=2)
        for chunk in world.iter_chunks():
            chunk.blocks[:, 2:8, :] = B.WATER
        player = Player(world, position=(0.5, 20.0, 0.5))
        simulate(player, 1.0)
        self.assertTrue(player.in_water)
        self.assertGreater(player.velocity[1], -3.0)

    def test_swim_up(self):
        world = flat_world(level=2)
        for chunk in world.iter_chunks():
            chunk.blocks[:, 2:12, :] = B.WATER
        player = Player(world, position=(0.5, 6.0, 0.5))
        simulate(player, 1.0, MoveInput(jump=True))
        self.assertGreater(player.position[1], 6.0)

    def test_head_in_water_flag(self):
        world = flat_world(level=2)
        for chunk in world.iter_chunks():
            chunk.blocks[:, 2:12, :] = B.WATER
        player = Player(world, position=(0.5, 5.0, 0.5))
        player.update(1 / 60, MoveInput())
        self.assertTrue(player.head_in_water)


class InteractionTests(unittest.TestCase):
    def setUp(self):
        self.world = flat_world(level=10)
        # a block to target at eye level, three blocks away in front (+X)
        self.world.set_block(3, 11, 0, B.BRICK)
        self.player = Player(self.world, position=(0.5, 10.0, 0.5))
        self.player.look_at((3.5, 11.5, 0.5))

    def test_raycast_target(self):
        hit = self.player.raycast()
        self.assertIsNotNone(hit)
        self.assertEqual(hit.block, (3, 11, 0))

    def test_break_block(self):
        hit = self.player.break_block()
        self.assertIsNotNone(hit)
        self.assertEqual(self.world.get_block(3, 11, 0), B.AIR)

    def test_break_records_edit(self):
        self.player.break_block()
        self.assertEqual(self.world.edit_count(), 1)

    def test_cannot_break_bedrock(self):
        self.world.set_block(3, 11, 0, B.BEDROCK)
        self.player.break_block()
        self.assertEqual(self.world.get_block(3, 11, 0), B.BEDROCK)

    def test_break_nothing_when_aiming_at_sky(self):
        before = self.world.edit_count()
        self.player.look_at((0.5, 200.0, 0.5))
        self.assertIsNone(self.player.break_block())
        self.assertEqual(self.world.edit_count(), before)

    def test_place_block(self):
        self.player.selected_block = B.PLANKS
        placed = self.player.place_block()
        self.assertIsNotNone(placed)
        self.assertEqual(self.world.get_block(*placed), B.PLANKS)
        # placing happens against the face we are looking at
        self.assertEqual(placed, (2, 11, 0))

    def test_place_on_top_of_the_ground(self):
        # look at the top face of the block next to us -> a block goes on top of it
        self.player.look_at((-0.5, 9.5, 0.5))
        placed = self.player.place_block(B.GLASS)
        self.assertEqual(placed, (-1, 10, 0))
        self.assertEqual(self.world.get_block(-1, 10, 0), B.GLASS)

    def test_cannot_place_inside_the_player(self):
        self.player.look_at((0.5, 9.0, 0.5))
        # the player occupies (0, 10..11, 0) so a placement there must be refused
        placed = self.player.place_block(B.STONE)
        self.assertIsNone(placed)
        self.assertEqual(self.world.get_block(0, 10, 0), B.AIR)

    def test_cannot_place_inside_a_block(self):
        self.player.look_at((3.5, 11.5, 0.5))
        before = self.world.get_block(1, 11, 0)
        self.world.set_block(1, 11, 0, B.STONE)
        placed = self.player.place_block(B.STONE)
        self.assertIsNone(placed)
        self.assertEqual(self.world.get_block(1, 11, 0), B.STONE)
        self.world.set_block(1, 11, 0, before)

    def test_place_replaces_water(self):
        self.world.set_block(2, 11, 0, B.WATER)
        placed = self.player.place_block(B.SAND)
        self.assertEqual(placed, (2, 11, 0))
        self.assertEqual(self.world.get_block(2, 11, 0), B.SAND)

    def test_reach_limit(self):
        far = Player(self.world, position=(0.5, 10.0, 0.5))
        far.look_at((3.5, 11.5, 0.5))
        self.assertIsNotNone(far.raycast())
        # move further away than the reach distance
        far.teleport((-6.0, 10.0, 0.5))
        far.look_at((3.5, 11.5, 0.5))
        self.assertIsNone(far.raycast())
        self.assertLess(REACH, 8.0)

    def test_build_a_tower(self):
        """Place blocks on top of each other by looking up after each placement."""
        player = Player(self.world, position=(0.5, 10.0, 0.5))
        player.look_at((3.5, 11.5, 0.5))
        placed = []
        for _ in range(3):
            hit = player.raycast()
            if hit is None:
                break
            player.look_at(np.asarray(hit.placement, dtype=float) + 0.5)
            spot = player.place_block(B.STONE)
            if spot:
                placed.append(spot)
        self.assertGreaterEqual(len(placed), 2)
        for spot in placed:
            self.assertEqual(self.world.get_block(*spot), B.STONE)

    def test_respawn(self):
        self.player.teleport((0.5, 30.0, 0.5))
        self.player.velocity[1] = -5.0
        self.player.respawn()
        self.assertAlmostEqual(self.player.position[1], 10.0, delta=0.01)
        self.assertEqual(self.player.velocity[1], 0.0)

    def test_falling_out_of_the_world_respawns(self):
        self.player.teleport((0.5, -30.0, 0.5))
        self.player.update(1 / 60, MoveInput())
        self.assertGreater(self.player.position[1], 0.0)


if __name__ == "__main__":
    unittest.main()
