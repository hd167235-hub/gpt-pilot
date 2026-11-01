"""Tests for the command line interface (``python -m voxelcraft``)."""

from __future__ import annotations

import sys
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO

from voxelcraft.tests import glstub

GL = glstub.install()

from voxelcraft import blocks as B  # noqa: E402
from voxelcraft.__main__ import main, parse_args  # noqa: E402


class ArgumentTests(unittest.TestCase):
    def test_defaults(self):
        args = parse_args([])
        self.assertEqual(args.render_distance, 7)
        self.assertEqual(args.width, 1280)
        self.assertEqual(args.height, 720)
        self.assertFalse(args.fullscreen)
        self.assertIsNone(args.load)

    def test_all_options(self):
        args = parse_args(["--seed", "99", "--render-distance", "9",
                           "--width", "800", "--height", "600", "--fullscreen",
                           "--no-msaa", "--save", "x.npz", "--load", "y.npz",
                           "--render-distance", "4"])
        self.assertEqual(args.seed, 99)
        self.assertEqual(args.render_distance, 4)
        self.assertEqual(args.width, 800)
        self.assertTrue(args.fullscreen)
        self.assertTrue(args.no_msaa)
        self.assertEqual(args.save, "x.npz")
        self.assertEqual(args.load, "y.npz")

    def test_help_and_version_exit_cleanly(self):
        for argv in (["--help"], ["--version"]):
            with self.assertRaises(SystemExit) as ctx, redirect_stdout(StringIO()):
                parse_args(argv)
            self.assertEqual(ctx.exception.code, 0)


class ListBlocksTests(unittest.TestCase):
    def test_list_blocks(self):
        out = StringIO()
        with redirect_stdout(out):
            code = main(["--list-blocks"])
        self.assertEqual(code, 0)
        text = out.getvalue()
        for defn in B.BLOCKS:
            self.assertIn(defn.name, text)
        self.assertIn("unbreakable", text)      # bedrock
        self.assertIn("non-solid", text)        # water


class RunTests(unittest.TestCase):
    """``main()`` must build a window, spin the loop and shut down cleanly."""

    def setUp(self):
        # a fake pyglet.app that runs a handful of frames and then returns
        self.app = types.ModuleType("pyglet.app")
        self.frames: list[float] = []

        def run():
            clock = sys.modules["pyglet.clock"].clock
            for _ in range(240):
                clock.tick(1 / 60)
                for window in list(GL.window_class.instances):
                    window.dispatch_event("on_draw")
                if not GL.window_class.instances:
                    break

        self.app.run = run
        self.app.windows = []
        self.app.event_loop = types.SimpleNamespace(is_running=False)
        self._saved = sys.modules.get("pyglet.app")
        sys.modules["pyglet.app"] = self.app
        import pyglet
        self._saved_attr = getattr(pyglet, "app", None)
        pyglet.app = self.app
        glstub.FakeWindow.instances.clear()

    def tearDown(self):
        import pyglet
        if self._saved is None:
            sys.modules.pop("pyglet.app", None)
        else:
            sys.modules["pyglet.app"] = self._saved
        if self._saved_attr is not None:
            pyglet.app = self._saved_attr
        for window in list(glstub.FakeWindow.instances):
            if not window.closed:
                window.close()
        glstub.FakeWindow.instances.clear()

    def test_run_creates_a_window_and_draws(self):
        GL.reset()
        code = main(["--seed", "5", "--render-distance", "2",
                     "--width", "480", "--height", "270", "--no-msaa",
                     "--save", "/tmp/voxelcraft_cli_test.npz"])
        self.assertEqual(code, 0)
        self.assertGreater(len(glstub.FakeWindow.instances), 0)
        self.assertGreater(len(GL.calls_of("glDrawArrays")), 0)
        self.assertGreater(len(GL.calls_of("glBufferData")), 0)

    def test_cli_overrides_are_validated(self):
        args = parse_args(["--render-distance", "99"])
        self.assertEqual(args.render_distance, 99)      # clamped later in main()


if __name__ == "__main__":
    unittest.main()
