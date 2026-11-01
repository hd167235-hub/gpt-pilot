"""Tests for the chunk mesher (face culling, UVs, ambient occlusion, lighting)."""

from __future__ import annotations

import unittest

import numpy as np

from voxelcraft import blocks as B
from voxelcraft import mesher
from voxelcraft.constants import CHUNK_SIZE, SEA_LEVEL, WORLD_HEIGHT
from voxelcraft.world import World


def padded_from(blocks: np.ndarray) -> np.ndarray:
    """Wrap a chunk array in an air border."""
    padded = np.full((CHUNK_SIZE + 2, WORLD_HEIGHT + 2, CHUNK_SIZE + 2), B.AIR, dtype=np.uint8)
    padded[1:-1, 1:-1, 1:-1] = blocks
    return padded


def quad_corners(vertices: np.ndarray) -> np.ndarray:
    """(n, 4, 6) view of a vertex array: each quad is two triangles v0v1v2/v0v2v3."""
    tris = vertices.reshape(-1, 3, vertices.shape[1])
    quads = np.concatenate([tris[0::2, 0:1], tris[0::2, 1:2], tris[0::2, 2:3],
                            tris[1::2, 2:3]], axis=1)
    return quads


def tile_indices(vertices: np.ndarray) -> set:
    """Atlas tile index of every face, derived from its UV centre."""
    from voxelcraft.textures import tile_uv

    centres = np.array([[(u0 + u1) / 2, (v0 + v1) / 2]
                        for (u0, v0, u1, v1) in (tile_uv(i) for i in range(16))])
    quads = quad_corners(vertices)
    uv_centre = quads[:, :, 3:5].mean(axis=1)
    distance = np.abs(centres[None, :, :] - uv_centre[:, None, :]).sum(axis=2)
    return {int(i) for i in np.argmin(distance, axis=1)}


def chunk_with(*placements) -> np.ndarray:
    """Build a chunk containing ``(x, y, z, block_id)`` placements."""
    blocks = np.zeros((CHUNK_SIZE, WORLD_HEIGHT, CHUNK_SIZE), dtype=np.uint8)
    for (x, y, z, block_id) in placements:
        blocks[x, y, z] = block_id
    return blocks


class FaceCullingTests(unittest.TestCase):
    def test_single_block_has_six_faces(self):
        geometry = mesher.mesh_chunk(padded_from(chunk_with((8, 20, 8, B.STONE))), 0, 0)
        self.assertEqual(mesher.face_count(geometry["opaque"]), 6)
        self.assertEqual(geometry["opaque"].shape, (36, mesher.VERTEX_FLOATS))

    def test_two_adjacent_blocks_hide_their_shared_face(self):
        geometry = mesher.mesh_chunk(
            padded_from(chunk_with((8, 20, 8, B.STONE), (9, 20, 8, B.STONE))), 0, 0)
        self.assertEqual(mesher.face_count(geometry["opaque"]), 10)

    def test_two_by_two_by_two_cube(self):
        placements = [(x, y, z, B.STONE)
                      for x in (4, 5) for y in (20, 21) for z in (4, 5)]
        geometry = mesher.mesh_chunk(padded_from(chunk_with(*placements)), 0, 0)
        self.assertEqual(mesher.face_count(geometry["opaque"]), 24)

    def test_face_hidden_by_neighbour_chunk(self):
        """A block at the chunk border must not emit the face touching the next chunk."""
        world = World(seed=6)
        for cx in (-1, 0, 1):
            for cz in (-1, 0, 1):
                world.generate_chunk(cx, cz)
        chunk = world.get_chunk(0, 0)
        # solid slab filling the eastern border and the chunk east of it
        chunk.blocks[:] = B.AIR
        world.get_chunk(1, 0).blocks[:] = B.AIR
        chunk.blocks[15, 30, 8] = B.STONE
        world.get_chunk(1, 0).blocks[0, 30, 8] = B.STONE
        geometry = mesher.mesh_chunk(world.padded_blocks(chunk), 0, 0)
        self.assertEqual(mesher.face_count(geometry["opaque"]), 5)

    def test_empty_chunk_has_no_faces(self):
        geometry = mesher.mesh_chunk(padded_from(chunk_with()), 0, 0)
        self.assertEqual(mesher.face_count(geometry["opaque"]), 0)
        self.assertEqual(geometry["opaque"].size, 0)

    def test_water_goes_to_transparent_pass(self):
        geometry = mesher.mesh_chunk(padded_from(chunk_with((8, 20, 8, B.WATER))), 0, 0)
        self.assertEqual(mesher.face_count(geometry["opaque"]), 0)
        # water/water faces are culled, so a lone water block yields 6
        self.assertEqual(mesher.face_count(geometry["transparent"]), 6)

    def test_glass_is_transparent_but_hides_neighbour_faces(self):
        geometry = mesher.mesh_chunk(
            padded_from(chunk_with((8, 20, 8, B.GLASS), (9, 20, 8, B.STONE))), 0, 0)
        self.assertEqual(mesher.face_count(geometry["transparent"]), 5)
        # the stone has no exposed face towards the glass but keeps the other 5
        self.assertEqual(mesher.face_count(geometry["opaque"]), 6)

    def test_adjacent_water_faces_are_culled(self):
        geometry = mesher.mesh_chunk(
            padded_from(chunk_with((8, 20, 8, B.WATER), (9, 20, 8, B.WATER))), 0, 0)
        self.assertEqual(mesher.face_count(geometry["transparent"]), 10)


class VertexDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        world = World(seed=8)
        for cx in (-1, 0, 1):
            for cz in (-1, 0, 1):
                world.generate_chunk(cx, cz)
        cls.world = world
        cls.chunk = world.get_chunk(0, 0)
        cls.geometry = mesher.mesh_chunk(world.padded_blocks(cls.chunk), 0, 0)

    def test_vertex_layout(self):
        verts = self.geometry["opaque"]
        self.assertEqual(verts.dtype, np.float32)
        self.assertEqual(verts.shape[1], 6)
        self.assertGreater(len(verts), 0)

    def test_indices_divisible_by_three(self):
        for key in ("opaque", "transparent"):
            verts = self.geometry[key]
            if verts.size:
                self.assertEqual(verts.shape[0] % 3, 0)

    def test_positions_are_inside_the_chunk(self):
        verts = self.geometry["opaque"]
        self.assertGreaterEqual(verts[:, 0].min(), 0.0)
        self.assertLessEqual(verts[:, 0].max(), CHUNK_SIZE)
        self.assertGreaterEqual(verts[:, 2].min(), 0.0)
        self.assertLessEqual(verts[:, 2].max(), CHUNK_SIZE)
        self.assertGreaterEqual(verts[:, 1].min(), 0.0)
        self.assertLessEqual(verts[:, 1].max(), WORLD_HEIGHT)

    def test_uvs_inside_atlas(self):
        verts = self.geometry["opaque"]
        self.assertGreaterEqual(verts[:, 3].min(), 0.0)
        self.assertLessEqual(verts[:, 3].max(), 1.0)
        self.assertGreaterEqual(verts[:, 4].min(), 0.0)
        self.assertLessEqual(verts[:, 4].max(), 1.0)
        # a face spans exactly one tile
        from voxelcraft.textures import tile_uv

        u0, v0, u1, v1 = tile_uv(0)
        quad = quad_corners(verts)
        np.testing.assert_allclose(np.abs(quad[:, 2, 3] - quad[:, 0, 3]), u1 - u0, atol=1e-6)
        np.testing.assert_allclose(np.abs(quad[:, 2, 4] - quad[:, 0, 4]), v1 - v0, atol=1e-6)

    def test_light_values_in_range(self):
        for key in ("opaque", "transparent"):
            verts = self.geometry[key]
            if verts.size:
                self.assertGreater(verts[:, 5].min(), 0.0)
                self.assertLessEqual(verts[:, 5].max(), 1.0)

    def test_top_faces_are_brighter_than_bottom_faces(self):
        geometry = mesher.mesh_chunk(padded_from(chunk_with((8, 20, 8, B.STONE))), 0, 0)
        quad = quad_corners(geometry["opaque"])
        horizontal = np.ptp(quad[:, :, 1], axis=1) < 1e-6
        tops = quad[horizontal & (quad[:, :, 1].mean(axis=1) > 20.5)]
        bottoms = quad[horizontal & (quad[:, :, 1].mean(axis=1) < 20.5)]
        self.assertEqual(len(tops), 1)
        self.assertEqual(len(bottoms), 1)
        self.assertGreater(float(tops[0, :, 5].mean()), float(bottoms[0, :, 5].mean()))

    def test_cube_faces_are_planar_quads(self):
        geometry = mesher.mesh_chunk(padded_from(chunk_with((8, 20, 8, B.STONE))), 0, 0)
        quad = quad_corners(geometry["opaque"])
        for face in quad:
            pos = face[:, :3]
            # every face is axis aligned: one coordinate is constant
            constant_axes = [i for i in range(3) if np.ptp(pos[:, i]) < 1e-6]
            self.assertEqual(len(constant_axes), 1)
            # and the other two span exactly one block
            for axis in range(3):
                if axis not in constant_axes:
                    self.assertAlmostEqual(float(np.ptp(pos[:, axis])), 1.0)

    def test_ambient_occlusion_darkens_corners(self):
        """Corners of a face get darker as neighbouring blocks occlude them."""

        def top_face_light(placements):
            geometry = mesher.mesh_chunk(padded_from(chunk_with(*placements)), 0, 0)
            quad = quad_corners(geometry["opaque"])
            horizontal = np.ptp(quad[:, :, 1], axis=1) < 1e-6
            top = quad[horizontal & (quad[:, :, 1].mean(axis=1) > 20.5)]
            # the top face of the block at (8, 20, 8)
            face = [f for f in top if f[:, 0].min() == 8 and f[:, 2].min() == 8]
            return face[0][:, 5].astype(np.float64)

        block = (8, 20, 8, B.STONE)
        lone = top_face_light([block])
        np.testing.assert_allclose(lone, 1.0, atol=1e-5)          # unobstructed
        self.assertEqual(int((lone < 1.0).sum()), 0)

        diagonal = top_face_light([block, (9, 21, 9, B.DIRT)])
        self.assertEqual(int((diagonal < 1.0).sum()), 1)          # one corner darker
        self.assertAlmostEqual(float(diagonal.min()), float(mesher.AO_LEVELS[2]), places=4)

        side = top_face_light([block, (9, 21, 8, B.DIRT)])
        self.assertEqual(int((side < 1.0).sum()), 2)              # an edge darker

        opposite = top_face_light([block, (9, 21, 8, B.DIRT), (7, 21, 8, B.DIRT)])
        self.assertEqual(int((opposite < 1.0).sum()), 4)

        # both sides plus the diagonal at one corner -> darkest AO level
        boxed = top_face_light([block, (9, 21, 8, B.DIRT), (8, 21, 9, B.DIRT),
                                (9, 21, 9, B.DIRT)])
        self.assertAlmostEqual(float(boxed.min()), float(mesher.AO_LEVELS[0]), places=4)

    def test_underground_faces_are_darker(self):
        world = World(seed=8)
        world.generate_chunk(0, 0)
        chunk = world.get_chunk(0, 0)
        chunk.blocks[:] = B.AIR
        chunk.blocks[:, 0, :] = B.BEDROCK
        chunk.blocks[8, 10, 8] = B.AIR
        # one block deep down, one high up
        chunk.blocks[4, 6, 4] = B.STONE
        chunk.blocks[4, 90, 4] = B.STONE
        geometry = mesher.mesh_chunk(world.padded_blocks(chunk), 0, 0)
        quad = quad_corners(geometry["opaque"])
        tops = quad[np.abs(quad[:, :, 1].mean(axis=1) - 7) < 0.5]
        highs = quad[np.abs(quad[:, :, 1].mean(axis=1) - 91) < 0.5]
        self.assertGreater(len(tops), 0)
        self.assertGreater(len(highs), 0)
        self.assertLess(float(tops[:, :, 5].mean()), float(highs[:, :, 5].mean()))

    def test_grass_uses_different_tiles_per_face(self):
        geometry = mesher.mesh_chunk(padded_from(chunk_with((8, 20, 8, B.GRASS))), 0, 0)
        self.assertEqual(tile_indices(geometry["opaque"]),
                         {B.T_GRASS_TOP, B.T_GRASS_SIDE, B.T_DIRT})

    def test_log_has_side_and_end_grain(self):
        geometry = mesher.mesh_chunk(padded_from(chunk_with((8, 20, 8, B.LOG))), 0, 0)
        self.assertEqual(tile_indices(geometry["opaque"]),
                         {B.T_LOG_SIDE, B.T_LOG_TOP})

    def test_meshing_is_deterministic(self):
        a = mesher.mesh_chunk(padded_from(chunk_with((8, 20, 8, B.BRICK))), 0, 0)
        b = mesher.mesh_chunk(padded_from(chunk_with((8, 20, 8, B.BRICK))), 0, 0)
        np.testing.assert_array_equal(a["opaque"], b["opaque"])

    def test_chunk_origin_is_applied(self):
        geometry = mesher.mesh_chunk(padded_from(chunk_with((8, 20, 8, B.STONE))), 3, -2)
        verts = geometry["opaque"]
        self.assertAlmostEqual(float(verts[:, 0].min()), 3 * CHUNK_SIZE + 8)
        self.assertAlmostEqual(float(verts[:, 2].min()), -2 * CHUNK_SIZE + 8)

    def test_real_chunk_is_reasonably_small(self):
        self.assertLess(mesher.face_count(self.geometry["opaque"]), 12000)


class ColumnTopTests(unittest.TestCase):
    def test_column_tops(self):
        blocks = chunk_with((3, 10, 4, B.STONE), (3, 30, 4, B.STONE), (5, 7, 5, B.DIRT))
        tops = mesher.column_tops(blocks)
        self.assertEqual(int(tops[3, 4]), 30)
        self.assertEqual(int(tops[5, 5]), 7)
        self.assertEqual(int(tops[0, 0]), -1)


class WaterLevelTests(unittest.TestCase):
    def test_ocean_surface_is_flat(self):
        world = World(seed=20260201)
        for cx in range(-3, 4):
            for cz in range(-3, 4):
                world.generate_chunk(cx, cz)
        water_top = -1
        for chunk in world.iter_chunks():
            levels = np.nonzero(chunk.blocks == B.WATER)[1]
            if levels.size:
                water_top = max(water_top, int(levels.max()))
        self.assertGreaterEqual(water_top, 0)
        self.assertEqual(water_top, SEA_LEVEL)


if __name__ == "__main__":
    unittest.main()
