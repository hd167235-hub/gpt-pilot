"""End-to-end tests of the game window: loading, input, editing, saving.

Runs against the fake GL/window layer from :mod:`voxelcraft.tests.glstub`, so
the whole game loop executes on a headless machine.
"""

from __future__ import annotations

import math
import os
import tempfile
import time
import unittest

from voxelcraft.tests import glstub

GL = glstub.install()

from voxelcraft import blocks as B  # noqa: E402
from voxelcraft import game as game_module  # noqa: E402
from voxelcraft.game import VoxelGame  # noqa: E402

KEY = GL.key
MOUSE = GL.mouse


def wait_until_loaded(game: VoxelGame, timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while game.state == game_module.LOADING and time.time() < deadline:
        game._update(1 / 60)
        time.sleep(0.002)
    if game.state == game_module.LOADING:
        raise AssertionError("world did not finish loading")


class GameTestCase(unittest.TestCase):
    seed = 4242
    render_distance = 2

    def setUp(self):
        GL.reset()
        self.tmp = tempfile.TemporaryDirectory()
        self.save_path = os.path.join(self.tmp.name, "world.npz")
        self.game = VoxelGame(seed=self.seed, render_distance=self.render_distance,
                              width=480, height=270, save_path=self.save_path,
                              msaa=False)

    def tearDown(self):
        try:
            self.game._shutdown()
        finally:
            self.tmp.cleanup()

    # -- helpers ---------------------------------------------------------
    def start(self):
        wait_until_loaded(self.game)
        self.game.on_mouse_press(0, 0, MOUSE.LEFT, 0)      # click through the menu
        self.assertEqual(self.game.state, game_module.PLAYING)
        return self.game

    def step(self, frames: int = 1, dt: float = 1 / 60) -> None:
        for _ in range(frames):
            self.game._update(dt)

    def aim_at_ground(self):
        """Point the camera at the floor in front of the player and return the hit."""
        player = self.game.player
        player.pitch = -math.pi / 2 + 0.35
        return player.raycast()


class LoadingTests(GameTestCase):
    def test_initial_state(self):
        self.assertEqual(self.game.state, game_module.LOADING)
        self.assertEqual(self.game.world.seed, self.seed)
        self.assertGreater(len(self.game.streamer._pending), 0)

    def test_loading_finishes_and_meshes_the_world(self):
        wait_until_loaded(self.game)
        self.assertEqual(self.game.state, game_module.MENU)
        self.assertGreater(self.game.world.loaded_count(), 0)
        self.assertGreater(len(self.game.renderer.meshes), 0)

    def test_loading_screen_draws(self):
        self.game.on_draw()
        self.assertGreater(len(GL.calls_of("glDrawArrays")), 0)
        self.assertGreater(self.game._load_progress, -1.0)

    def test_spawn_is_on_solid_ground(self):
        wait_until_loaded(self.game)
        player = self.game.player
        x = int(math.floor(player.position[0]))
        z = int(math.floor(player.position[2]))
        ground = self.game.world.top_solid_y(x, z)
        self.assertGreater(ground, 0)
        self.assertGreater(player.position[1], ground)


class InputTests(GameTestCase):
    def test_clicking_the_menu_starts_the_game(self):
        self.start()
        self.assertTrue(self.game.exclusive_mouse)

    def test_escape_pauses_and_resumes(self):
        self.start()
        self.game.on_key_press(KEY.ESCAPE, 0)
        self.assertEqual(self.game.state, game_module.PAUSED)
        self.assertFalse(self.game.exclusive_mouse)
        self.game.on_key_release(KEY.ESCAPE, 0)
        self.game.on_key_press(KEY.ESCAPE, 0)
        self.assertEqual(self.game.state, game_module.PLAYING)

    def test_enter_resumes_from_the_menu(self):
        wait_until_loaded(self.game)
        self.game.on_key_press(KEY.RETURN, 0)
        self.assertEqual(self.game.state, game_module.PLAYING)
        self.game.on_key_press(KEY.ESCAPE, 0)
        self.game.on_key_press(KEY.RETURN, 0)
        self.assertEqual(self.game.state, game_module.PLAYING)

    def test_mouse_look_rotates_the_camera(self):
        self.start()
        before = self.game.player.yaw
        self.game.on_mouse_motion(100, 100, 50, 0)
        self.assertNotAlmostEqual(self.game.player.yaw, before)
        pitch_before = self.game.player.pitch
        self.game.on_mouse_motion(100, 100, 0, 50)
        self.assertLess(self.game.player.pitch, pitch_before)   # mouse up looks up

    def test_hotbar_selection_by_number_key(self):
        self.start()
        self.game.on_key_press(KEY._3, 0)
        self.assertEqual(self.game.selected_index, 2)
        self.assertEqual(B.HOTBAR_BLOCKS[2], B.STONE)

    def test_hotbar_selection_by_scroll(self):
        self.start()
        self.game.on_mouse_scroll(0, 0, 0, -1)
        self.assertEqual(self.game.selected_index, 1)
        self.game.on_mouse_scroll(0, 0, 0, 1)
        self.assertEqual(self.game.selected_index, 0)
        self.game.on_mouse_scroll(0, 0, 0, 1)
        self.assertEqual(self.game.selected_index, len(B.HOTBAR_BLOCKS) - 1)   # wraps

    def test_forward_key_moves_the_player(self):
        self.start()
        self.game.player.place_on_surface()
        start = self.game.player.position.copy()
        self.game.on_key_press(KEY.W, 0)
        self.step(45)
        self.assertGreater(float(((self.game.player.position - start) ** 2).sum()), 0.5)
        self.game.on_key_release(KEY.W, 0)

    def test_jump_key_leaves_the_ground(self):
        self.start()
        self.game.player.place_on_surface()
        self.step(10)                                   # settle
        ground = self.game.player.position[1]
        self.game.on_key_press(KEY.SPACE, 0)
        peak = ground
        for _ in range(30):
            self.step(1, 1 / 120)
            peak = max(peak, self.game.player.position[1])
        self.assertGreater(peak, ground + 0.4)
        self.game.on_key_release(KEY.SPACE, 0)

    def test_f_toggles_flying(self):
        self.start()
        self.assertFalse(self.game.player.flying)
        self.game.on_key_press(KEY.F, 0)
        self.assertTrue(self.game.player.flying)
        self.game.on_key_press(KEY.F, 0)
        self.assertFalse(self.game.player.flying)

    def test_double_tap_space_toggles_flying(self):
        self.start()
        self.game.on_key_press(KEY.SPACE, 0)
        self.game.on_key_release(KEY.SPACE, 0)
        self.game.on_key_press(KEY.SPACE, 0)          # immediately again
        self.assertTrue(self.game.player.flying)

    def test_debug_and_hud_toggles(self):
        self.start()
        self.game.on_key_press(KEY.F3, 0)
        self.assertTrue(self.game.show_debug)
        self.game.on_key_press(KEY.F1, 0)
        self.assertFalse(self.game.show_hud)

    def test_render_distance_keys(self):
        self.start()
        before = self.game.renderer.render_distance
        self.game.on_key_press(KEY.EQUAL, 0)
        self.assertEqual(self.game.renderer.render_distance, before + 1)
        self.game.on_key_press(KEY.MINUS, 0)
        self.assertEqual(self.game.renderer.render_distance, before)
        # it is clamped to a sane range
        for _ in range(20):
            self.game.on_key_press(KEY.MINUS, 0)
        self.assertEqual(self.game.renderer.render_distance, 2)
        for _ in range(30):
            self.game.on_key_press(KEY.EQUAL, 0)
        self.assertEqual(self.game.renderer.render_distance, 14)

    def test_respawn_key(self):
        self.start()
        self.game.player.teleport((0.5, 120.0, 0.5))
        self.game.on_key_press(KEY.R, 0)
        self.assertLess(self.game.player.position[1], 120.0)


class BlockEditingTests(GameTestCase):
    def test_left_click_destroys_a_block(self):
        self.start()
        hit = self.aim_at_ground()
        self.assertIsNotNone(hit)
        target, block_id = hit.block, hit.block_id
        self.assertNotEqual(block_id, B.AIR)
        self.game.on_mouse_press(0, 0, MOUSE.LEFT, 0)
        self.assertEqual(self.game.world.get_block(*target), B.AIR)

    def test_right_click_places_a_block(self):
        self.start()
        self.game.on_key_press(KEY._3, 0)                # stone in hand
        hit = self.aim_at_ground()
        self.assertIsNotNone(hit)
        spot = hit.placement
        self.game.on_mouse_press(0, 0, MOUSE.RIGHT, 0)
        self.assertEqual(self.game.world.get_block(*spot), B.STONE)

    def test_placing_stacks_while_held(self):
        self.start()
        hit = self.aim_at_ground()
        self.assertIsNotNone(hit)
        self.step(1)
        self.game.on_mouse_press(0, 0, MOUSE.RIGHT, 0)
        edits_after_press = self.game.world.edit_count()
        # holding the button for a while keeps placing (with a cooldown)
        self.step(60)
        self.assertGreaterEqual(self.game.world.edit_count(), edits_after_press)

    def test_edits_bump_the_chunk_version(self):
        self.start()
        hit = self.aim_at_ground()
        chunk = self.game.world.get_chunk(hit.block[0] >> 4, hit.block[2] >> 4)
        version = chunk.version
        self.game.on_mouse_press(0, 0, MOUSE.LEFT, 0)
        self.assertGreater(chunk.version, version)

    def test_middle_click_picks_the_block(self):
        self.start()
        self.aim_at_ground()
        self.game.on_mouse_press(0, 0, MOUSE.MIDDLE, 0)
        hit = self.game.player.raycast()
        if hit.block_id in B.HOTBAR_BLOCKS:
            self.assertEqual(B.HOTBAR_BLOCKS[self.game.selected_index], hit.block_id)

    def test_cannot_edit_while_paused(self):
        self.start()
        self.game.on_key_press(KEY.ESCAPE, 0)
        before = self.game.world.edit_count()
        self.game.on_mouse_press(0, 0, MOUSE.LEFT, 0)     # this resumes instead
        self.assertEqual(self.game.state, game_module.PLAYING)
        self.assertEqual(self.game.world.edit_count(), before)


class DrawTests(GameTestCase):
    def test_draw_every_state(self):
        self.game.on_draw()                      # loading
        wait_until_loaded(self.game)
        self.game.on_draw()                      # menu
        self.start()
        self.game.on_draw()                      # playing
        self.game.show_debug = True
        self.game.on_draw()
        self.game.on_key_press(KEY.ESCAPE, 0)
        self.game.on_draw()                      # paused
        self.assertGreater(len(GL.calls_of("glDrawArrays")), 10)

    def test_resize(self):
        self.start()
        self.game.on_resize(800, 480)
        self.assertEqual((self.game.renderer.width, self.game.renderer.height), (800, 480))
        self.game.on_draw()

    def test_debug_lines_are_strings(self):
        self.start()
        lines = self.game._debug_lines()
        self.assertTrue(all(isinstance(line, str) for line in lines))
        self.assertIn("seed", lines[0].replace("   seed", " seed"))


class PersistenceTests(GameTestCase):
    def test_save_on_pause_and_reload(self):
        self.start()
        hit = self.aim_at_ground()
        self.game.on_mouse_press(0, 0, MOUSE.LEFT, 0)
        self.assertEqual(self.game.world.get_block(*hit.block), B.AIR)
        self.game.on_key_press(KEY.ESCAPE, 0)             # pauses + saves
        self.assertTrue(os.path.exists(self.save_path))

        self.game._shutdown()
        GL.reset()
        reloaded = VoxelGame(seed=1, render_distance=1, width=320, height=200,
                             load_path=self.save_path, msaa=False)
        try:
            self.assertEqual(reloaded.world.seed, self.seed)
            reloaded.setup_done = True
            wait_until_loaded(reloaded)
            self.assertEqual(reloaded.world.get_block(*hit.block), B.AIR)
        finally:
            reloaded._shutdown()

    def test_autosave(self):
        self.start()
        self.aim_at_ground()
        self.game.on_mouse_press(0, 0, MOUSE.LEFT, 0)
        self.game._autosave_timer = 0.0
        self.step(2)
        self.assertTrue(os.path.exists(self.save_path))


class WorldStreamingTests(GameTestCase):
    def test_chunks_stream_in_around_the_player(self):
        self.start()
        before = self.game.world.loaded_count()
        self.step(120)
        self.assertGreaterEqual(self.game.world.loaded_count(), before)

    def test_unloading_far_chunks(self):
        self.start()
        self.step(30)
        loaded = self.game.world.loaded_count()
        self.assertLess(loaded, 400)

    def test_walking_into_new_territory_generates_chunks(self):
        self.start()
        player = self.game.player
        player.toggle_fly()
        self.step(5)
        x0 = player.position[0]
        self.game.on_key_press(KEY.W, 0)
        self.step(240)
        self.game.on_key_release(KEY.W, 0)
        self.assertGreater(abs(player.position[0] - x0), 1.0)
        self.assertGreater(self.game.world.loaded_count(), 0)
        # the player must never end up inside solid geometry
        self.assertFalse(player._collides(player.position))


if __name__ == "__main__":
    unittest.main()
