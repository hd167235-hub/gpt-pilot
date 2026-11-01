"""Renderer, HUD, atlas and camera tests.

The GL calls are recorded by :mod:`voxelcraft.tests.glstub`, so these tests run
on machines without a GPU while still exercising the real renderer code path.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from voxelcraft.tests import glstub

GL = glstub.install()

from voxelcraft import blocks as B  # noqa: E402
from voxelcraft import camera as cam  # noqa: E402
from voxelcraft import font, hud, renderer, textures  # noqa: E402
from voxelcraft.player import Player  # noqa: E402
from voxelcraft.ui import UIBatch  # noqa: E402
from voxelcraft.world import World  # noqa: E402


class CameraTests(unittest.TestCase):
    def test_perspective_matrix(self):
        m = cam.perspective(90.0, 1.0, 0.1, 100.0)
        self.assertAlmostEqual(m[0, 0], m[1, 1])          # square aspect
        self.assertAlmostEqual(m[3, 2], -1.0)
        wide = cam.perspective(90.0, 2.0, 0.1, 100.0)
        self.assertAlmostEqual(wide[0, 0], wide[1, 1] / 2.0)

    def test_look_at_orthonormal(self):
        view = cam.look_at((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
        rotation = view[:3, :3]
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
        # the eye maps to the origin
        eye = np.array([1.0, 2.0, 3.0, 1.0])
        np.testing.assert_allclose((view @ eye)[:3], 0.0, atol=1e-9)

    def test_look_at_straight_down(self):
        view = cam.look_at((0.0, 5.0, 0.0), (0.0, 0.0, 0.0))
        self.assertFalse(np.any(np.isnan(view)))

    def test_direction_from_angles(self):
        np.testing.assert_allclose(cam.direction_from_angles(0.0, 0.0), [0, 0, -1], atol=1e-9)
        np.testing.assert_allclose(cam.direction_from_angles(math.pi / 2, 0.0),
                                   [1, 0, 0], atol=1e-9)
        np.testing.assert_allclose(cam.direction_from_angles(0.0, math.pi / 2),
                                   [0, 1, 0], atol=1e-9)

    def test_frustum_contains_forward_points(self):
        projection = cam.perspective(70.0, 1.6, 0.1, 300.0)
        view = cam.look_at((0.0, 0.0, 0.0), (0.0, 0.0, -1.0))
        frustum = cam.Frustum(projection @ view)
        self.assertTrue(frustum.intersects_aabb((-1, -1, -10), (1, 1, -8)))
        self.assertFalse(frustum.intersects_aabb((-1, -1, 10), (1, 1, 12)))    # behind
        self.assertFalse(frustum.intersects_aabb((200, -1, -10), (201, 1, -8)))

    def test_frustum_accepts_boxes_that_straddle_the_edge(self):
        projection = cam.perspective(70.0, 1.6, 0.1, 300.0)
        view = cam.look_at((0.0, 0.0, 0.0), (0.0, 0.0, -1.0))
        frustum = cam.Frustum(projection @ view)
        self.assertTrue(frustum.intersects_aabb((-100, -100, -100), (100, 100, 100)))


class TextureTests(unittest.TestCase):
    def test_atlas_dimensions(self):
        atlas = textures.build_atlas()
        self.assertEqual(atlas.shape, (textures.ATLAS_SIZE, textures.ATLAS_SIZE, 4))
        self.assertEqual(atlas.dtype, np.uint8)

    def test_atlas_is_deterministic(self):
        np.testing.assert_array_equal(textures.build_atlas(5), textures.build_atlas(5))

    def test_tiles_are_not_empty(self):
        atlas = textures.build_atlas()
        for index in range(B.T_WHITE + 1):
            u0, v0, u1, v1 = textures.tile_uv(index)
            x0, x1 = int(u0 * textures.ATLAS_SIZE), int(u1 * textures.ATLAS_SIZE)
            y0 = int((1 - v1) * textures.ATLAS_SIZE)
            y1 = int((1 - v0) * textures.ATLAS_SIZE)
            tile = atlas[y0:y1, x0:x1]
            self.assertEqual(tile.shape[:2], (textures.TILE_PX, textures.TILE_PX))
            self.assertGreater(tile[:, :, 3].max(), 0)

    def test_uv_rects_do_not_overlap(self):
        rects = [textures.tile_uv(i) for i in range(16)]
        for i, (u0, v0, u1, v1) in enumerate(rects):
            self.assertLess(u0, u1)
            self.assertLess(v0, v1)
            for j in range(i + 1, len(rects)):
                a0, b0, a1, b1 = rects[j]
                self.assertFalse(u0 < a1 and a0 < u1 and v0 < b1 and b0 < v1)

    def test_padding_protects_against_bleeding(self):
        """The padding repeats the edge pixel, so no filter can bleed a tile."""
        atlas = textures.build_atlas()
        pad = textures.PAD
        cell = textures.CELL
        for index in (0, 4, 9):
            cx = (index % textures.ATLAS_TILES) * cell
            cy = (index // textures.ATLAS_TILES) * cell
            core_x0, core_x1 = cx + pad, cx + pad + textures.TILE_PX
            core_y0, core_y1 = cy + pad, cy + pad + textures.TILE_PX
            # left padding column == first core column
            np.testing.assert_array_equal(atlas[core_y0:core_y1, core_x0 - 1],
                                          atlas[core_y0:core_y1, core_x0])
            # top padding row == first core row
            np.testing.assert_array_equal(atlas[core_y0 - 1, core_x0:core_x1],
                                          atlas[core_y0, core_x0:core_x1])

    def test_water_tile_is_translucent(self):
        atlas = textures.build_atlas()
        u0, v0, u1, v1 = textures.tile_uv(B.T_WATER)
        x0, x1 = int(u0 * textures.ATLAS_SIZE), int(u1 * textures.ATLAS_SIZE)
        y0, y1 = int((1 - v1) * textures.ATLAS_SIZE), int((1 - v0) * textures.ATLAS_SIZE)
        alpha = atlas[y0:y1, x0:x1, 3]
        self.assertLess(int(alpha.max()), 255)
        self.assertGreater(int(alpha.min()), 100)


class FontTests(unittest.TestCase):
    def test_coverage(self):
        text = ("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                "0123456789 .,:;!?()[]{}<>/\\|+-=_*&%$#@'\"~^`")
        for ch in text:
            self.assertIn(ch, font.INDEX)

    def test_atlas_size(self):
        atlas = font.build_atlas()
        self.assertEqual(atlas.shape, (font.ATLAS_H, font.ATLAS_W, 4))
        self.assertTrue((atlas[:, :, 3] > 0).any())

    def test_text_width(self):
        self.assertEqual(font.text_width("", 2), 0)
        self.assertGreater(font.text_width("HELLO", 2), font.text_width("HI", 2))
        self.assertGreater(font.text_width("HI", 4), font.text_width("HI", 2))

    def test_unknown_characters_do_not_crash(self):
        font.glyph_uv("\u2603")
        font.glyph_uv("é")


class UIBatchTests(unittest.TestCase):
    def test_quad_vertex_count(self):
        batch = UIBatch()
        batch.add_rect(0, 0, 10, 10, (1, 1, 1, 1))
        self.assertEqual(batch.to_array().shape, (6, 8))

    def test_text_generates_two_triangles_per_glyph(self):
        batch = UIBatch()
        batch.add_text("AB", 0, 0, 2.0, shadow=None)
        self.assertEqual(batch.to_array().shape[0], 12)      # 2 glyphs * 6
        batch.clear()
        batch.add_text("AB", 0, 0, 2.0)                      # with shadow
        self.assertEqual(batch.to_array().shape[0], 24)

    def test_spaces_are_skipped(self):
        batch = UIBatch()
        width = batch.add_text("A A", 0, 0, 2.0, shadow=None)
        self.assertEqual(batch.to_array().shape[0], 12)
        self.assertGreater(width, 0)

    def test_polygon(self):
        batch = UIBatch()
        batch.add_polygon([(0, 0, 0, 0), (1, 0, 1, 0), (1, 1, 1, 1), (0, 1, 0, 1)],
                          (1, 0, 0, 1))
        self.assertEqual(batch.to_array().shape, (6, 8))

    def test_data_is_float32_and_finite(self):
        batch = UIBatch()
        batch.add_border(0, 0, 100, 40, 2, (1, 1, 1, 1))
        batch.add_text("hello", 4, 4, 2.0)
        data = batch.to_array()
        self.assertEqual(data.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(data)))


class HudTests(unittest.TestCase):
    def build(self, width=1280, height=720):
        shapes, text = UIBatch(), UIBatch()
        hud.draw_crosshair(shapes, width, height)
        hud.draw_hotbar(shapes, text, width, height, 0)
        hud.draw_position_indicator(shapes, text, width, height, 1.2, -0.2)
        hud.draw_debug(shapes, text, width, height, ["*Voxelcraft", "FPS 60"])
        hud.draw_toast(shapes, text, width, height, "Hello", 1.0)
        return shapes, text

    def test_hud_produces_geometry(self):
        shapes, text = self.build()
        self.assertGreater(shapes.to_array().shape[0], 0)
        self.assertGreater(text.to_array().shape[0], 0)

    def test_all_quads_inside_the_screen(self):
        width, height = 1280, 720
        shapes, _ = self.build(width, height)
        data = shapes.to_array()
        self.assertGreaterEqual(data[:, 0].min(), -1.0)
        self.assertLessEqual(data[:, 0].max(), width + 1.0)
        self.assertGreaterEqual(data[:, 1].min(), -1.0)
        self.assertLessEqual(data[:, 1].max(), height + 1.0)

    def test_hotbar_lists_every_selectable_block(self):
        shapes, text = UIBatch(), UIBatch()
        hud.draw_hotbar(shapes, text, 1280, 720, 0)
        # 9 slots, each with a background, border (4) and 3 icon faces
        self.assertGreaterEqual(shapes.to_array().shape[0] // 6, 9 * 8)

    def test_block_icon_uses_block_tiles(self):
        shapes = UIBatch()
        hud.draw_block_icon(shapes, B.GRASS, 100, 100, 32)
        data = shapes.to_array().reshape(-1, 6, 8)
        self.assertEqual(len(data), 3)                   # three visible faces

        centres = np.array([[(uv[0] + uv[2]) / 2, (uv[1] + uv[3]) / 2]
                            for uv in (textures.tile_uv(i) for i in range(16))])
        used = set()
        for face in data:
            uv_centre = face[:, 2:4].mean(axis=0)
            used.add(int(np.argmin(np.abs(centres - uv_centre).sum(axis=1))))
        self.assertEqual(used, {B.T_GRASS_TOP, B.T_GRASS_SIDE})

    def test_loading_screen(self):
        shapes, text = UIBatch(), UIBatch()
        hud.draw_loading(shapes, text, 1280, 720, 0.42)
        self.assertGreater(shapes.to_array().shape[0], 0)
        self.assertGreater(text.to_array().shape[0], 0)

    def test_title_screen(self):
        shapes, text = UIBatch(), UIBatch()
        hud.draw_title_screen(shapes, text, 1280, 720, "subtitle",
                              ["##MOVE", "W A S D - walk"], "press enter")
        self.assertGreater(shapes.to_array().shape[0], 0)
        self.assertGreater(text.to_array().shape[0], 0)


class RendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = World(seed=11)
        cls.world.generate_chunk(0, 0)
        cls.world.generate_chunk(1, 0)
        cls.window = glstub.FakeWindow(width=640, height=360)
        cls.renderer = renderer.Renderer(cls.window, cls.world, render_distance=2)
        cls.player = Player(cls.world, position=(8.5, 60.0, 8.5))
        cls.player.place_on_surface()

    @classmethod
    def tearDownClass(cls):
        cls.renderer.release()

    def setUp(self):
        GL.reset()

    def test_shaders_compiled_and_linked(self):
        GL.reset()
        program = renderer.Program(renderer.CHUNK_VERT, renderer.CHUNK_FRAG, "test")
        self.assertGreater(len(GL.calls_of("glCompileShader")), 0)
        self.assertEqual(len(GL.calls_of("glLinkProgram")), 1)
        program.set_vec3("u_camera", (1, 2, 3))
        self.assertGreater(len(GL.calls_of("glUniform3f")), 0)

    def test_vertex_attributes_are_bound(self):
        """Every vertex shader input must have a matching glVertexAttribPointer."""
        GL.reset()
        temp = renderer.Renderer(glstub.FakeWindow(width=64, height=64), self.world,
                                 render_distance=1)
        temp.update_chunks(self.player, budget=10)
        pointers = GL.calls_of("glVertexAttribPointer")
        temp.release()
        self.assertGreaterEqual(len(pointers), 3)
        # chunk vertices: position(3), uv(2), light(1)
        strides = {int(call[4]) for call in pointers}
        self.assertIn(renderer.VERTEX_FLOATS * 4, strides)

    def test_chunk_meshes_are_created(self):
        built = self.renderer.update_chunks(self.player, budget=10)
        self.assertGreater(built, 0)
        self.assertGreater(len(self.renderer.meshes), 0)
        entry = next(iter(self.renderer.meshes.values()))
        self.assertGreater(entry["opaque"].count, 0)

    def test_mesh_is_rebuilt_after_an_edit(self):
        self.renderer.update_chunks(self.player, budget=10)
        cx, cz = 0, 0
        before = self.renderer.meshes[(cx, cz)]["signature"]
        world = self.world
        x, z = 8, 8
        y = world.top_solid_y(x, z)
        world.set_block(x, y, z, B.AIR)
        self.renderer.update_chunks(self.player, budget=10)
        self.assertNotEqual(self.renderer.meshes[(cx, cz)]["signature"], before)

    def test_draw_issues_world_and_hud_calls(self):
        self.renderer.update_chunks(self.player, budget=10)
        GL.reset()
        shapes, text = UIBatch(), UIBatch()
        hud.draw_crosshair(shapes, 640, 360)
        hud.draw_hotbar(shapes, text, 640, 360, 0)
        self.renderer.draw(self.player, shapes, text, selected_block=(8, 40, 8))
        draws = GL.calls_of("glDrawArrays")
        self.assertGreaterEqual(len(draws), 3)      # sky + chunks + hud
        self.assertGreater(len(GL.calls_of("glUniformMatrix4fv")), 0)
        self.assertGreater(self.renderer.stats["triangles"], 0)
        self.assertGreater(self.renderer.stats["chunks_drawn"], 0)

    def test_draw_without_hud(self):
        self.renderer.update_chunks(self.player, budget=10)
        GL.reset()
        self.renderer.draw(self.player)
        self.assertGreaterEqual(len(GL.calls_of("glDrawArrays")), 2)

    def test_underwater_state_is_handled(self):
        self.player.head_in_water = True
        try:
            self.renderer.update_chunks(self.player, budget=2)
            self.renderer.draw(self.player)
        finally:
            self.player.head_in_water = False

    def test_resize(self):
        self.renderer.on_resize(800, 500)
        self.assertEqual((self.renderer.width, self.renderer.height), (800, 500))
        self.assertGreater(len(GL.calls_of("glViewport")), 0)
        self.renderer.on_resize(640, 360)

    def test_render_distance_limits_meshes(self):
        self.renderer.render_distance = 0
        self.renderer.update_chunks(self.player, budget=100)
        far = [key for key in self.renderer.meshes
               if max(abs(key[0]), abs(key[1])) > 0]
        self.assertEqual(far, [])
        self.renderer.render_distance = 2

    def test_end_to_end_world_is_drawn(self):
        self.renderer.update_chunks(self.player, budget=100)
        GL.reset()
        self.renderer.draw(self.player)
        # one sky triangle + at least one chunk
        self.assertGreater(len(GL.calls_of("glDrawArrays")), 1)


class PreviewToolTests(unittest.TestCase):
    """The software rasterizer used for screenshots must stay in sync."""

    def test_small_frame(self):
        from tools.rasterizer import render_world

        world = World(seed=3)
        for cx in range(0, 2):
            for cz in range(0, 2):
                world.generate_chunk(cx, cz)
        player = Player(world, position=(8.5, 80.0, 8.5), yaw=0.7, pitch=-0.2)
        player.place_on_surface()
        canvas = render_world(world, player, width=64, height=40, radius=1)
        self.assertGreater(canvas.triangles_drawn, 0)
        self.assertEqual(canvas.color.shape, (40, 64, 3))
        self.assertGreater(float(canvas.color.std()), 0.01)   # not a flat image


if __name__ == "__main__":
    unittest.main()
