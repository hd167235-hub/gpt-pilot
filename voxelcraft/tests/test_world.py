"""Tests for chunk storage, block edits, raycasting and persistence."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from voxelcraft import blocks as B
from voxelcraft.constants import CHUNK_SIZE, SEA_LEVEL, WORLD_HEIGHT
from voxelcraft.world import World


def make_world(seed: int = 5, radius: int = 2) -> World:
    world = World(seed=seed)
    for cz in range(-radius, radius + 1):
        for cx in range(-radius, radius + 1):
            world.generate_chunk(cx, cz)
    return world


def flat_world(level: int = 10, radius: int = 1) -> World:
    """A world of one stone plateau (used by physics tests)."""
    world = make_world(seed=3, radius=radius)
    for chunk in world.iter_chunks():
        chunk.blocks[:] = B.AIR
        chunk.blocks[:, :level, :] = B.STONE
        chunk.blocks[:, 0, :] = B.BEDROCK
    return world


class BlockAccessTests(unittest.TestCase):
    def setUp(self):
        self.world = make_world()

    def test_get_block_roundtrip(self):
        x, z = 3, 5
        y = self.world.top_solid_y(x, z) + 1
        self.assertEqual(self.world.get_block(x, y, z), B.AIR)
        self.assertTrue(self.world.set_block(x, y, z, B.BRICK))
        self.assertEqual(self.world.get_block(x, y, z), B.BRICK)
        self.assertTrue(self.world.is_solid(x, y, z))
        self.assertTrue(self.world.is_opaque(x, y, z))

    def test_set_block_is_idempotent(self):
        self.world.set_block(1, 60, 1, B.STONE)
        self.assertFalse(self.world.set_block(1, 60, 1, B.STONE))

    def test_out_of_bounds(self):
        self.assertEqual(self.world.get_block(0, -1, 0), B.AIR)
        self.assertEqual(self.world.get_block(0, WORLD_HEIGHT, 0), B.AIR)
        self.assertFalse(self.world.set_block(0, -5, 0, B.STONE))
        self.assertFalse(self.world.set_block(0, WORLD_HEIGHT + 1, 0, B.STONE))

    def test_unloaded_chunk_is_air(self):
        self.assertEqual(self.world.get_block(9999, 40, 9999), B.AIR)

    def test_liquid_and_transparent_flags(self):
        self.world.set_block(2, 50, 2, B.WATER)
        self.assertTrue(self.world.is_liquid(2, 50, 2))
        self.assertFalse(self.world.is_solid(2, 50, 2))
        self.world.set_block(2, 51, 2, B.GLASS)
        self.assertFalse(self.world.is_opaque(2, 51, 2))
        self.assertTrue(self.world.is_solid(2, 51, 2))

    def test_edits_are_recorded_and_reapplied(self):
        x, z = 4, 6
        y = self.world.top_solid_y(x, z) + 1
        self.world.set_block(x, y, z, B.PLANKS)
        self.assertEqual(self.world.edit_count(), 1)
        # unload and regenerate: the edit must come back
        self.world.unload_far(100, 100, keep_radius=0)
        self.assertEqual(self.world.loaded_count(), 0)
        self.world.generate_chunk(x >> 4, z >> 4)
        self.assertEqual(self.world.get_block(x, y, z), B.PLANKS)

    def test_neighbour_version_bumped_on_border_edit(self):
        world = make_world(seed=9)
        a = world.get_chunk(0, 0)
        east = world.get_chunk(1, 0)
        before = east.version
        world.set_block(CHUNK_SIZE - 1, 40, 8, B.BRICK)
        self.assertGreater(east.version, before)
        self.assertGreater(a.version, 0)

    def test_top_solid_y(self):
        x, z = 7, 7
        top = self.world.top_solid_y(x, z)
        self.assertEqual(self.world.get_block(x, top, z) != B.AIR, True)
        self.assertEqual(self.world.get_block(x, top + 1, z), B.AIR)

    def test_unload_far_keeps_near_chunks(self):
        world = make_world(seed=2, radius=3)
        total = world.loaded_count()
        removed = world.unload_far(0, 0, keep_radius=1)
        self.assertGreater(len(removed), 0)
        self.assertLess(world.loaded_count(), total)
        self.assertIsNotNone(world.get_chunk(0, 0))

    def test_missing_chunks_sorted_by_distance(self):
        world = World(seed=1)
        missing = world.missing_chunks(0, 0, 3)
        distances = [d for d, _, _ in missing]
        self.assertEqual(distances, sorted(distances))
        self.assertTrue(all(d <= 3.5 ** 2 for d in distances))

    def test_stream_creates_closest_first(self):
        world = World(seed=1)
        world.stream(0, 0, radius=3, budget=3)
        self.assertEqual(world.loaded_count(), 3)
        self.assertIsNotNone(world.get_chunk(0, 0))


class PaddedBlocksTests(unittest.TestCase):
    def test_borders_come_from_neighbours(self):
        world = make_world(seed=4, radius=2)
        chunk = world.get_chunk(0, 0)
        padded = world.padded_blocks(chunk)
        self.assertEqual(padded.shape, (CHUNK_SIZE + 2, WORLD_HEIGHT + 2, CHUNK_SIZE + 2))
        np.testing.assert_array_equal(padded[1:-1, 1:-1, 1:-1], chunk.blocks)
        east = world.get_chunk(1, 0)
        np.testing.assert_array_equal(padded[-1, 1:-1, 1:-1], east.blocks[0, :, :])
        west = world.get_chunk(-1, 0)
        np.testing.assert_array_equal(padded[0, 1:-1, 1:-1], west.blocks[15, :, :])
        north = world.get_chunk(0, -1)
        np.testing.assert_array_equal(padded[1:-1, 1:-1, 0], north.blocks[:, :, 15])
        self.assertTrue(np.all(padded[:, -1, :] == B.AIR))     # open sky above

    def test_missing_neighbour_is_filled_with_stone(self):
        world = World(seed=8)
        world.generate_chunk(0, 0)
        padded = world.padded_blocks(world.get_chunk(0, 0))
        self.assertTrue(np.all(padded[0, 1:-1, 1:-1] == B.STONE))


class RaycastTests(unittest.TestCase):
    def setUp(self):
        self.world = flat_world(level=10)
        # a lone block to aim at
        self.world.set_block(2, 11, 0, B.BRICK)

    def test_hit_from_west(self):
        hit = self.world.raycast((0.5, 11.5, 0.5), (1.0, 0.0, 0.0), 6.0)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.block, (2, 11, 0))
        self.assertEqual(hit.normal, (-1, 0, 0))
        self.assertEqual(hit.placement, (1, 11, 0))
        self.assertAlmostEqual(hit.distance, 1.5, places=3)

    def test_hit_from_above(self):
        hit = self.world.raycast((2.5, 15.0, 0.5), (0.0, -1.0, 0.0), 6.0)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.block, (2, 11, 0))
        self.assertEqual(hit.normal, (0, 1, 0))
        self.assertEqual(hit.placement, (2, 12, 0))

    def test_miss_returns_none(self):
        self.assertIsNone(self.world.raycast((0.5, 30.0, 0.5), (0.0, 1.0, 0.0), 6.0))

    def test_respects_max_distance(self):
        self.assertIsNone(self.world.raycast((0.5, 11.5, 0.5), (1.0, 0.0, 0.0), 1.0))
        self.assertIsNotNone(self.world.raycast((0.5, 11.5, 0.5), (1.0, 0.0, 0.0), 3.0))

    def test_hits_ground_when_looking_down(self):
        hit = self.world.raycast((5.5, 11.0, 5.5), (0.0, -1.0, 0.0), 8.0)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.block[1], 9)          # top of the stone plateau
        self.assertEqual(hit.normal, (0, 1, 0))

    def test_all_six_faces(self):
        # an isolated floating block so every approach direction is unobstructed
        self.world.set_block(2, 20, 0, B.BRICK)
        origin = (2.5, 20.5, 0.5)
        cases = {
            (1.0, 0.0, 0.0): (-1, 0, 0),
            (-1.0, 0.0, 0.0): (1, 0, 0),
            (0.0, 1.0, 0.0): (0, -1, 0),
            (0.0, -1.0, 0.0): (0, 1, 0),
            (0.0, 0.0, 1.0): (0, 0, -1),
            (0.0, 0.0, -1.0): (0, 0, 1),
        }
        for direction, expected_normal in cases.items():
            start = (origin[0] - direction[0] * 3.0,
                     origin[1] - direction[1] * 3.0,
                     origin[2] - direction[2] * 3.0)
            hit = self.world.raycast(start, direction, 6.0)
            self.assertIsNotNone(hit, f"no hit for {direction}")
            self.assertEqual(hit.block, (2, 20, 0))
            self.assertEqual(hit.normal, expected_normal)

    def test_liquid_ignored_by_default(self):
        self.world.set_block(4, 11, 0, B.WATER)
        self.assertIsNone(self.world.raycast((0.5, 11.5, 0.5), (0, 0, 0), 6.0))
        hit = self.world.raycast((0.5, 11.5, 0.5), (1.0, 0.0, 0.0), 5.0)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.block_id, B.BRICK)     # still the brick behind it
        liquid = self.world.raycast((0.5, 11.5, 0.5), (1.0, 0.0, 0.0), 5.0, hit_liquid=True)
        self.assertEqual(liquid.block_id, B.BRICK)


class PersistenceTests(unittest.TestCase):
    def test_save_and_load_roundtrip(self):
        world = make_world(seed=31337, radius=1)
        world.set_block(1, 60, 1, B.BRICK)
        world.set_block(20, 61, -4, B.GLASS)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "world.npz")
            count = world.save(path)
            self.assertEqual(count, 2)

            loaded = World.load(path)
            self.assertEqual(loaded.seed, 31337)
            self.assertEqual(loaded.edit_count(), 2)
            loaded.generate_chunk(0, 0)
            self.assertEqual(loaded.get_block(1, 60, 1), B.BRICK)
            loaded.generate_chunk(1, -1)
            self.assertEqual(loaded.get_block(20, 61, -4), B.GLASS)

    def test_save_empty_world(self):
        world = World(seed=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "empty.npz")
            self.assertEqual(world.save(path), 0)
            loaded = World.load(path)
            self.assertEqual(loaded.edit_count(), 0)


class SeaLevelTests(unittest.TestCase):
    def test_ocean_chunks_have_water(self):
        world = make_world(seed=20260201, radius=4)
        water = 0
        for chunk in world.iter_chunks():
            water += int((chunk.blocks == B.WATER).sum())
        self.assertGreater(water, 0)
        # and no water above the sea level
        for chunk in world.iter_chunks():
            levels = np.nonzero(chunk.blocks == B.WATER)[1]
            if levels.size:
                self.assertLessEqual(int(levels.max()), SEA_LEVEL)


if __name__ == "__main__":
    unittest.main()
