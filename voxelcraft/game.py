"""Window, input handling and the main loop of Voxelcraft."""

from __future__ import annotations

import logging
import math
import os
import time
from collections import deque
from typing import Optional

import pyglet

# GL error checking on every call is only useful while debugging the renderer,
# so turn it off before any display/window module is imported.
pyglet.options["debug_gl"] = False

from . import blocks as B  # noqa: E402
from . import hud  # noqa: E402
from .constants import DEFAULT_RENDER_DISTANCE, REACH  # noqa: E402
from .player import MoveInput, Player  # noqa: E402
from .renderer import Renderer  # noqa: E402
from .streaming import ChunkStreamer  # noqa: E402
from .ui import UIBatch  # noqa: E402
from .world import World  # noqa: E402

from pyglet.window import key, mouse  # noqa: E402

log = logging.getLogger(__name__)

DEFAULT_SEED = 20260201
SAVE_DIR = "saves"

LOADING, MENU, PLAYING, PAUSED = "loading", "menu", "playing", "paused"

CONTROL_LINES = [
    "##MOVE",
    "W A S D - walk          SPACE - jump / swim up",
    "SHIFT - sneak           CTRL - sprint",
    "F or double-tap SPACE - fly (SPACE up, SHIFT down)",
    "",
    "##BLOCKS",
    "LEFT CLICK - destroy block    RIGHT CLICK - place block",
    "MIDDLE CLICK - pick block     1-9 or wheel - select",
    "",
    "##VIEW",
    "F3 - debug overlay     F1 - hide HUD",
    "- / = - render distance        R - respawn",
    "ESC - pause",
]


class VoxelGame(pyglet.window.Window):
    """The playable window."""

    def __init__(self, seed: int = DEFAULT_SEED,
                 render_distance: int = DEFAULT_RENDER_DISTANCE,
                 width: int = 1280, height: int = 720,
                 fullscreen: bool = False,
                 save_path: Optional[str] = None,
                 load_path: Optional[str] = None,
                 msaa: bool = True):
        config = pyglet.gl.Config(
            double_buffer=True, depth_size=24,
            major_version=3, minor_version=3, forward_compatible=True,
            sample_buffers=1 if msaa else 0, samples=4 if msaa else 0,
        )
        super().__init__(width=width, height=height, caption="Voxelcraft",
                         resizable=True, config=config, vsync=True,
                         fullscreen=fullscreen)
        self.set_minimum_size(640, 400)

        self.state = LOADING
        self.mouse_sensitivity = 0.0022
        self.keys_down: set[int] = set()
        self.show_debug = False
        self.show_hud = True
        self.fps = 0.0
        self._frame_times: deque[float] = deque(maxlen=45)
        self._last_time = time.perf_counter()
        self._last_draw = time.perf_counter()
        self._last_space_press = 0.0
        self._action_cooldown = 0.0
        self._held_button: Optional[int] = None
        self._label_text = ""
        self._label_timer = 0.0
        self._toast_text = ""
        self._toast_timer = 0.0
        self._target = None
        self._selected_index = 0
        self._autosave_timer = 60.0

        # --- world ------------------------------------------------------
        loaded_world = None
        if load_path:
            try:
                loaded_world = World.load(load_path)
                seed = loaded_world.seed
                self._toast("Loaded " + os.path.basename(load_path))
            except Exception:
                log.exception("could not load %s", load_path)
        self.world = loaded_world or World(seed=seed)
        self.seed = self.world.seed
        self.save_path = save_path or os.path.join(SAVE_DIR, f"world_{self.seed}.npz")
        self.streamer = ChunkStreamer(self.world, workers=2, max_pending=10)

        spawn = self.world.generator.find_spawn(0.0, 0.0)
        self.player = Player(self.world, position=spawn, yaw=math.radians(200))
        self.player.spawn = self.player.position.copy()

        # --- rendering --------------------------------------------------
        self.renderer = Renderer(self, self.world, render_distance=render_distance)
        self.ui_shapes = UIBatch()
        self.ui_text = UIBatch()

        # --- loading progress ------------------------------------------
        self._load_total = ChunkStreamer._expected_count(render_distance)
        self._load_progress = 0.0
        self._load_stage = "Generating terrain"
        spawn_cx, spawn_cz = self._chunk_of(self.player.position)
        self.streamer.request_near(spawn_cx, spawn_cz, render_distance)

        pyglet.clock.schedule_interval(self._update, 1.0 / 120.0)
        log.info("Voxelcraft: seed=%s render_distance=%s", self.seed, render_distance)

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _chunk_of(position) -> tuple[int, int]:
        return (int(math.floor(position[0])) >> 4, int(math.floor(position[2])) >> 4)

    def _toast(self, message: str, duration: float = 2.5) -> None:
        self._toast_text = message
        self._toast_timer = duration

    def _label(self, message: str, duration: float = 1.6) -> None:
        self._label_text = message
        self._label_timer = duration

    @property
    def selected_index(self) -> int:
        return self._selected_index

    # -- input -----------------------------------------------------------
    def on_key_press(self, symbol, modifiers):
        if symbol in (key.W, key.A, key.S, key.D, key.SPACE, key.LSHIFT,
                      key.LCTRL, key.RCTRL, key.RSHIFT):
            self.keys_down.add(symbol)

        if symbol == key.ESCAPE:
            if self.state == PLAYING:
                self._pause()
            elif self.state in (PAUSED, MENU):
                self._resume()
        elif symbol in (key.RETURN, key.ENTER):
            if self.state == MENU:
                self._resume()
            elif self.state == PAUSED:
                self._resume()
        elif symbol == key.Q and self.state == PAUSED:
            self.close()
        elif self.state != PLAYING:
            return
        elif symbol == key.F3:
            self.show_debug = not self.show_debug
        elif symbol == key.F1:
            self.show_hud = not self.show_hud
        elif symbol == key.F:
            self._toggle_fly()
        elif symbol == key.R:
            self._respawn()
        elif symbol == key.SPACE:
            now = time.perf_counter()
            if now - self._last_space_press < 0.32:
                self._toggle_fly()
                self._last_space_press = 0.0
            else:
                self._last_space_press = now
        elif symbol in (key.MINUS, key.NUM_SUBTRACT):
            self._change_render_distance(-1)
        elif symbol in (key.EQUAL, key.NUM_ADD):
            self._change_render_distance(+1)
        elif key._1 <= symbol <= key._9:
            index = symbol - key._1
            if index < len(B.HOTBAR_BLOCKS):
                self._selected_index = index
                self._label(B.BY_ID[B.HOTBAR_BLOCKS[index]].name)

    def on_key_release(self, symbol, modifiers):
        self.keys_down.discard(symbol)

    def on_mouse_motion(self, x, y, dx, dy):
        if self.state == PLAYING and self.exclusive_mouse:
            self.player.rotate(dx * self.mouse_sensitivity,
                               -dy * self.mouse_sensitivity)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.on_mouse_motion(x, y, dx, dy)

    def on_mouse_press(self, x, y, button, modifiers):
        if self.state in (MENU, PAUSED):
            self._resume()
            return
        if self.state != PLAYING:
            return
        self._held_button = button
        self._action_cooldown = 0.0
        self._do_action(button)

    def on_mouse_release(self, x, y, button, modifiers):
        if self._held_button == button:
            self._held_button = None

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        if self.state != PLAYING or not scroll_y:
            return
        step = -1 if scroll_y > 0 else 1
        self._selected_index = (self._selected_index + step) % len(B.HOTBAR_BLOCKS)
        self._label(B.BY_ID[B.HOTBAR_BLOCKS[self._selected_index]].name)

    def on_resize(self, width, height):
        super().on_resize(width, height)
        if hasattr(self, "renderer"):
            self.renderer.on_resize(width, height)

    def on_close(self):
        # Release GL resources while the context is still alive, then let
        # pyglet perform its default close/exit handling.
        self._shutdown()
        super().on_close()

    # -- actions ---------------------------------------------------------
    def _do_action(self, button: int) -> None:
        if button == mouse.LEFT:
            hit = self.player.break_block()
            if hit is not None and hit.block_id != B.AIR:
                defn = B.BY_ID[hit.block_id]
                if defn.breakable:
                    self._toast(f"Broke {defn.name.lower()}", 0.8)
                else:
                    self._toast("Bedrock is unbreakable", 1.2)
        elif button == mouse.RIGHT:
            block_id = B.HOTBAR_BLOCKS[self._selected_index]
            placed = self.player.place_block(block_id)
            if placed is None:
                pass  # nothing to place against / occupied
        elif button == mouse.MIDDLE:
            hit = self.player.raycast()
            if hit is not None and hit.block_id in B.HOTBAR_BLOCKS:
                self._selected_index = B.HOTBAR_BLOCKS.index(hit.block_id)
                self._label(B.BY_ID[hit.block_id].name)

    def _toggle_fly(self) -> None:
        flying = self.player.toggle_fly()
        self._toast("Fly mode " + ("on" if flying else "off"), 1.2)

    def _respawn(self) -> None:
        self.player.respawn()
        self.player.place_on_surface()
        self._toast("Respawned", 1.2)

    def _change_render_distance(self, delta: int) -> None:
        self.renderer.render_distance = max(2, min(14, self.renderer.render_distance + delta))
        self._toast(f"Render distance: {self.renderer.render_distance} chunks", 1.4)

    def _pause(self) -> None:
        self.state = PAUSED
        self._held_button = None
        self.set_exclusive_mouse(False)
        self._save_world(quiet=True)

    def _resume(self) -> None:
        if self.state == LOADING:
            return
        self.state = PLAYING
        self.set_exclusive_mouse(True)
        self.keys_down.clear()

    # -- world / loop ----------------------------------------------------
    def _movement_input(self) -> MoveInput:
        keys = self.keys_down
        return MoveInput(
            forward=(1.0 if key.W in keys else 0.0) - (1.0 if key.S in keys else 0.0),
            strafe=(1.0 if key.D in keys else 0.0) - (1.0 if key.A in keys else 0.0),
            jump=key.SPACE in keys,
            sneak=key.LSHIFT in keys,
            sprint=(key.LCTRL in keys or key.RCTRL in keys),
            up=key.SPACE in keys,
            down=key.LSHIFT in keys,
        )

    def _update(self, dt: float) -> None:
        dt = min(dt, 0.1)
        if self.state == LOADING:
            self._update_loading()
            return

        self._decay_timers(dt)
        if self.state != PLAYING:
            return

        # keep the ground under the player's feet
        cx, cz = self._chunk_of(self.player.position)
        if self.world.get_chunk(cx, cz) is None:
            self.world.generate_chunk(cx, cz)
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if self.world.get_chunk(cx + dx, cz + dz) is None:
                    self.world.generate_chunk(cx + dx, cz + dz)

        move = self._movement_input()
        self.player.sneaking = move.sneak and not self.player.flying
        self.player.update(dt, move)

        # sprinting widens the field of view a little (juice)
        target_fov = 72.0 + (6.0 if self.player.sprinting else 0.0)
        self.renderer.fov += (target_fov - self.renderer.fov) * min(1.0, dt * 8.0)

        # repeat breaking / placing while the button is held
        self._action_cooldown -= dt
        if self._held_button is not None and self._action_cooldown <= 0.0:
            self._do_action(self._held_button)
            self._action_cooldown = 0.20 if self._held_button == mouse.LEFT else 0.24

        # streaming + meshing
        self.streamer.request_near(cx, cz, self.renderer.render_distance)
        self.streamer.poll(limit=2)
        self.renderer.update_chunks(self.player, budget=2)
        self.world.unload_far(cx, cz, self.renderer.render_distance + 3)
        self._target = self.player.raycast(REACH)

        self._autosave_timer -= dt
        if self._autosave_timer <= 0.0:
            self._autosave_timer = 60.0
            self._save_world(quiet=True)

    def _update_loading(self) -> None:
        cx, cz = self._chunk_of(self.player.position)
        radius = self.renderer.render_distance
        self.streamer.request_near(cx, cz, radius)
        self.streamer.poll(limit=3)
        missing = self.world.missing_chunks(cx, cz, radius)
        terrain_ready = 1.0 - len(missing) / max(1, self._load_total)
        # also mesh what is already there so the first frame is not a stutter
        self.renderer.update_chunks(self.player, budget=4)
        pending_meshes = sum(
            1 for (mcx, mcz) in self.world.chunks
            if self.renderer.meshes.get((mcx, mcz)) is None
        )
        mesh_ready = 1.0 - pending_meshes / max(1, len(self.world.chunks))
        self._load_progress = max(0.0, min(1.0, 0.7 * terrain_ready + 0.3 * mesh_ready))
        self._load_stage = "Generating terrain" if missing else "Building meshes"

        if not missing and pending_meshes == 0 and self.streamer.pending == 0:
            self.player.place_on_surface(self.player.position[0], self.player.position[2])
            self.player.spawn = self.player.position.copy()
            self.player.yaw = math.radians(200)
            self.player.pitch = math.radians(-8)
            self.state = MENU
            log.info("world ready: %s", self.world)

    def _decay_timers(self, dt: float) -> None:
        if self._label_timer > 0:
            self._label_timer = max(0.0, self._label_timer - dt)
        if self._toast_timer > 0:
            self._toast_timer = max(0.0, self._toast_timer - dt)

    def _measure_frame(self) -> None:
        now = time.perf_counter()
        self._frame_times.append(now - self._last_draw)
        self._last_draw = now
        if len(self._frame_times) >= 5:
            self.fps = len(self._frame_times) / max(1e-6, sum(self._frame_times))

    # -- drawing ---------------------------------------------------------
    def on_draw(self):
        if not hasattr(self, "renderer"):
            return
        self._measure_frame()
        self.ui_shapes.clear()
        self.ui_text.clear()

        if self.state == LOADING:
            hud.draw_loading(self.ui_shapes, self.ui_text, self.width, self.height,
                             self._load_progress, self._load_stage)
            self.renderer.draw_plain(self.ui_shapes, self.ui_text)
            return

        if self.show_hud and self.state == PLAYING:
            if self.player.head_in_water:
                hud.draw_underwater(self.ui_shapes, self.width, self.height)
            hud.draw_crosshair(self.ui_shapes, self.width, self.height)
            hud.draw_hotbar(self.ui_shapes, self.ui_text, self.width, self.height,
                            self._selected_index)
            hud.draw_position_indicator(self.ui_shapes, self.ui_text, self.width,
                                        self.height, self.player.yaw, self.player.pitch)
            if self._label_timer > 0:
                hud.draw_block_label(self.ui_text, self.ui_shapes, self.width,
                                     self.height, self._label_text)
            if self._toast_timer > 0:
                hud.draw_toast(self.ui_shapes, self.ui_text, self.width, self.height,
                               self._toast_text, self._toast_timer)
            if self.show_debug:
                hud.draw_debug(self.ui_shapes, self.ui_text, self.width, self.height,
                               self._debug_lines())

        if self.state == MENU:
            hud.draw_title_screen(
                self.ui_shapes, self.ui_text, self.width, self.height,
                subtitle="a tiny Minecraft in Python",
                lines=CONTROL_LINES,
                footer="CLICK or press ENTER to play",
            )
        elif self.state == PAUSED:
            hud.draw_title_screen(
                self.ui_shapes, self.ui_text, self.width, self.height,
                subtitle="paused",
                lines=CONTROL_LINES,
                footer="CLICK / ENTER to resume    Q to quit",
            )

        selected = None
        if self.state == PLAYING and self._target is not None:
            selected = self._target.block
        self.renderer.draw(self.player, self.ui_shapes, self.ui_text,
                           selected_block=selected)

    def _debug_lines(self) -> list[str]:
        p = self.player
        stats = self.renderer.stats
        target = self._target
        lines = [
            f"*Voxelcraft 1.0   seed {self.seed}",
            f"FPS {self.fps:5.1f}   frame {1000.0 / max(self.fps, 1e-3):4.1f} ms",
            f"XYZ {p.position[0]:7.2f} {p.position[1]:6.2f} {p.position[2]:7.2f}",
            f"Chunk {int(math.floor(p.position[0])) >> 4} "
            f"{int(math.floor(p.position[2])) >> 4}   "
            f"yaw {math.degrees(p.yaw) % 360:5.1f}  pitch {math.degrees(p.pitch):5.1f}",
            f"Chunks {self.world.loaded_count()} loaded, "
            f"{stats['chunks_drawn']} drawn, {self.streamer.pending} pending",
            f"Triangles {stats['triangles']}   edits {self.world.edit_count()}",
            f"Mode {'flying' if p.flying else 'swimming' if p.in_water else 'walking'}"
            f"{'  (on ground)' if p.on_ground else ''}",
            f"Render distance {self.renderer.render_distance} chunks",
        ]
        if target is not None:
            defn = B.BY_ID[target.block_id]
            lines.append(f"Looking at {defn.name} {target.block}")
        else:
            lines.append("Looking at nothing")
        return lines

    # -- persistence -----------------------------------------------------
    def _save_world(self, quiet: bool = False) -> None:
        try:
            count = self.world.save(self.save_path)
            if not quiet:
                self._toast(f"Saved {count} block changes", 2.0)
            log.info("saved %s block edits to %s", count, self.save_path)
        except Exception:
            log.exception("failed to save world to %s", self.save_path)

    def _shutdown(self) -> None:
        if getattr(self, "_shut_down", False):
            return
        self._shut_down = True
        try:
            pyglet.clock.unschedule(self._update)
        except Exception:
            pass
        try:
            self.streamer.shutdown()
        except Exception:
            pass
        try:
            if self.world.edit_count() > 0:
                self.world.save(self.save_path)
                log.info("world saved to %s", self.save_path)
        except Exception:
            log.exception("could not save the world on exit")
        try:
            self.renderer.release()
        except Exception:
            pass


def run(**kwargs) -> None:
    """Create the game, run pyglet and clean up afterwards."""
    try:
        game = VoxelGame(**kwargs)
    except Exception:
        if not kwargs.get("msaa"):
            raise
        # Some drivers refuse a multisampled config - fall back to a plain one.
        log.warning("could not create an MSAA window, retrying without AA",
                    exc_info=True)
        kwargs = dict(kwargs, msaa=False)
        game = VoxelGame(**kwargs)
    try:
        pyglet.app.run()
    finally:
        game._shutdown()
