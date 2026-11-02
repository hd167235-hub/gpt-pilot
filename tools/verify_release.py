#!/usr/bin/env python3
"""Verify the downloadable single-file build actually runs.

``download/Voxelcraft.py`` embeds every module as source and loads them through
an import hook, so it is worth checking that the *shipped artefact* - not just
the package - starts the game, generates a world, responds to input and edits
blocks.  This script drives it headlessly with the fake GL layer from
:mod:`voxelcraft.tests.glstub` (the same one the test-suite uses).

Usage::

    python -m tools.verify_release
"""

from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "download" / "Voxelcraft.py"


def main(argv=None) -> int:
    if not BUNDLE.exists():
        print(f"{BUNDLE} is missing - run `python -m tools.build_release` first",
              file=sys.stderr)
        return 2

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    # --- install the fake GL / window / clock layer ---------------------
    from voxelcraft.tests import glstub

    GL = glstub.install()
    fake_windows = GL.window_class.instances
    fake_windows.clear()

    # pyglet.app.run() -> tick the game loop a few hundred frames, then return
    app = types.ModuleType("pyglet.app")
    outcomes = {"frames": 0, "drew": 0}

    def run() -> None:
        import time

        import pyglet

        clock = pyglet.clock.clock
        game = None
        deadline = time.time() + 120
        while time.time() < deadline:
            clock.tick(1 / 60)
            for window in list(fake_windows):
                if window.closed:
                    continue
                window.dispatch_event("on_draw")
                outcomes["drew"] += 1
            outcomes["frames"] += 1
            if fake_windows:
                game = fake_windows[0]
            if game is not None and getattr(game, "state", None) not in (None, "loading"):
                break
            time.sleep(0.002)

        if game is None:
            raise RuntimeError("the game never created a window")
        outcomes["state"] = game.state
        outcomes["chunks"] = game.world.loaded_count()

        # play a little: start, aim at the ground, break a block, place one
        game.on_mouse_press(0, 0, GL.mouse.LEFT, 0)
        outcomes["playing"] = game.state
        player = game.player
        player.place_on_surface()
        player.pitch = -1.2
        clock.tick(1 / 60)
        hit = player.raycast()
        if hit is None:
            raise RuntimeError("nothing to aim at")
        before = game.world.get_block(*hit.block)
        game.on_mouse_press(0, 0, GL.mouse.LEFT, 0)
        outcomes["broke"] = (before, game.world.get_block(*hit.block))
        hit = player.raycast()
        if hit is not None:
            game.on_mouse_press(0, 0, GL.mouse.RIGHT, 0)
            outcomes["placed"] = game.world.get_block(*hit.placement)
        for _ in range(30):                       # a few frames of real play
            clock.tick(1 / 60)
            game.on_draw()
        outcomes["edits"] = game.world.edit_count()
        game.close()

    app.run = run
    app.windows = []
    app.event_loop = types.SimpleNamespace(is_running=False)
    sys.modules["pyglet.app"] = app
    import pyglet

    saved_app = pyglet.app
    pyglet.app = app

    sys.argv = ["Voxelcraft.py", "--seed", "4242", "--render-distance", "2",
                "--width", "480", "--height", "270", "--no-msaa",
                "--save", "/tmp/voxelcraft_release_check.npz"]
    GL.reset()
    try:
        runpy.run_path(str(BUNDLE), run_name="__main__")
    except SystemExit as exc:                      # the game returns an exit code
        code = exc.code or 0
        if code:
            print(f"bundle exited with {code}", file=sys.stderr)
            return int(code)
    finally:
        pyglet.app = saved_app

    draws = len(GL.calls_of("glDrawArrays"))
    buffers = len(GL.calls_of("glBufferData"))
    print(f"frames ticked     : {outcomes['frames']}")
    print(f"menu -> play      : {outcomes.get('state')} -> {outcomes.get('playing')}")
    print(f"chunks generated  : {outcomes.get('chunks')}")
    print(f"block broken      : {outcomes.get('broke')}")
    print(f"block placed      : {outcomes.get('placed')}")
    print(f"world edits       : {outcomes.get('edits')}")
    print(f"glDrawArrays calls: {draws}")
    print(f"glBufferData calls: {buffers}")

    problems = []
    if outcomes.get("playing") != "playing":
        problems.append("the game never entered play mode")
    if draws == 0 or buffers == 0:
        problems.append("the bundled game never drew anything")
    if outcomes.get("edits", 0) < 1:
        problems.append("breaking a block did not change the world")
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}", file=sys.stderr)
        return 1
    print("single-file build OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
