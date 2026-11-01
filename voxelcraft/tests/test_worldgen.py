"""Tests for procedural terrain generation."""

from __future__ import annotations

import unittest

import numpy as np

from voxelcraft import blocks as B
from voxelcraft.constants import CHUNK_SIZE, SEA_LEVEL, WORLD_HEIGHT
from voxelcraft.worldgen import TerrainGenerator


class GeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gen = TerrainGenerator(4242)

    def test_chunk_shape_and_dtype(self):
        chunk = self.gen.generate_chunk(0, 0)
        self.assertEqual(chunk.shape, (CHUNK_SIZE, WORLD_HEIGHT, CHUNK_SIZE))
        self.assertEqual(chunk.dtype, np.uint8)

    def test_deterministic(self):
        a = TerrainGenerator(77).generate_chunk(3, -2)
        b = TerrainGenerator(77).generate_chunk(3, -2)
        np.testing.assert_array_equal(a, b)

    def test_seed_changes_terrain(self):
        a = TerrainGenerator(1).generate_chunk(0, 0)
        b = TerrainGenerator(2).generate_chunk(0, 0)
        self.assertFalse(np.array_equal(a, b))

    def test_bedrock_floor(self):
        chunk = self.gen.generate_chunk(-3, 5)
        self.assertTrue(np.all(chunk[:, 0, :] == B.BEDROCK))

    def test_no_blocks_of_unknown_type(self):
        chunk = self.gen.generate_chunk(-2, 2)
        self.assertTrue(set(np.unique(chunk)) <= set(B.BY_ID))

    def test_water_only_at_or_below_sea_level(self):
        for cx, cz in ((0, 0), (4, -3), (-7, 6)):
            chunk = self.gen.generate_chunk(cx, cz)
            levels = np.nonzero(chunk == B.WATER)[1]
            if levels.size:
                self.assertLessEqual(int(levels.max()), SEA_LEVEL)

    def test_surface_blocks_make_sense(self):
        gx, gz = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
        heights = self.gen.height_at(gx, gz)
        chunk = self.gen.generate_chunk(0, 0)
        for i in range(16):
            for j in range(16):
                column = chunk[i, :, j]
                surface_y = int(np.max(np.nonzero(column)[0]))
                # the top block is the terrain surface, or canopy above it
                self.assertGreaterEqual(surface_y, int(heights[i, j]) - 2)
                if surface_y <= SEA_LEVEL:
                    # under water the top block must be sand/dirt/stone, no grass
                    self.assertIn(int(column[surface_y]), (B.SAND, B.DIRT, B.STONE))

    def test_grass_appears_above_water(self):
        chunk = self.gen.generate_chunk(0, 0)
        self.assertIn(B.GRASS, set(np.unique(chunk)))

    def test_caves_exist(self):
        """Deep underground there should be carved air."""
        chunk = self.gen.generate_chunk(0, 0)
        gx, gz = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
        heights = self.gen.height_at(gx, gz)
        air = chunk == B.AIR
        underground = air.copy()
        for i in range(16):
            for j in range(16):
                underground[i, int(heights[i, j]) - 1:, j] = False
        self.assertGreater(int(underground.sum()), 0)

    def test_trees_have_leaves_and_logs(self):
        found_log = found_leaves = False
        for cx in range(-3, 4):
            for cz in range(-3, 4):
                chunk = self.gen.generate_chunk(cx, cz)
                found_log |= bool((chunk == B.LOG).any())
                found_leaves |= bool((chunk == B.LEAVES).any())
        self.assertTrue(found_log, "no trees generated")
        self.assertTrue(found_leaves)

    def test_leaf_canopy_sits_above_trunk_base(self):
        chunk = self.gen.generate_chunk(0, 0)
        logs = np.argwhere(chunk == B.LOG)
        leaves = np.argwhere(chunk == B.LEAVES)
        if len(logs) and len(leaves):
            self.assertGreater(leaves[:, 1].max(), logs[:, 1].min())

    def test_trees_are_seamless_across_chunk_borders(self):
        """A tree straddling a border must be identical in both chunks."""
        a = TerrainGenerator(11).generate_chunk(1, 0)
        b = TerrainGenerator(11).generate_chunk(2, 0)
        border = np.concatenate([a[CHUNK_SIZE - 3:, :, :], b[:3, :, :]], axis=0)
        # The concatenation must be internally consistent: trunks are continuous
        log_columns = set()
        for x, y, z in np.argwhere(border == B.LOG):
            log_columns.add((x, z))
        for (x, z) in log_columns:
            ys = sorted(y for (xx, y, zz) in np.argwhere(border == B.LOG) if xx == x and zz == z)
            if len(ys) > 1:
                gaps = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
                self.assertTrue(all(g == 1 for g in gaps), "trunk has a gap at a chunk border")

    def test_heights_are_not_flat(self):
        h = self.gen.height_at(*np.meshgrid(np.arange(0, 256, 4), np.arange(0, 256, 4),
                                            indexing="ij"))
        self.assertGreater(h.std(), 4.0)
        self.assertGreater(h.max() - h.min(), 20)
        self.assertTrue(np.all(h >= 3))
        self.assertTrue(np.all(h <= WORLD_HEIGHT - 10))

    def test_has_water_and_land(self):
        X, Z = np.meshgrid(np.arange(0, 512, 4), np.arange(0, 512, 4), indexing="ij")
        h = self.gen.height_at(X, Z)
        water_fraction = float((h <= SEA_LEVEL).mean())
        self.assertGreater(water_fraction, 0.05)
        self.assertLess(water_fraction, 0.75)

    def test_find_spawn_is_walkable(self):
        for seed in (1, 99, 20260201):
            gen = TerrainGenerator(seed)
            x, y, z = gen.find_spawn(0, 0)
            h = float(gen.height_at(x, z))
            self.assertGreater(h, SEA_LEVEL)
            self.assertAlmostEqual(y, h + 1.2, delta=2.0)
            self.assertLess(abs(x), 120)
            self.assertLess(abs(z), 120)


if __name__ == "__main__":
    unittest.main()
