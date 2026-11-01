"""Fake ``pyglet.gl`` / ``pyglet.window`` / ``pyglet.clock`` for headless tests.

The game needs an OpenGL context to run for real.  CI machines (and the
sandbox this project was developed in) have neither a GPU nor the GL libraries,
so :func:`install` replaces pyglet's GL layer with a recorder.  This lets the
test-suite execute the *entire* renderer and game loop - shader compilation,
buffer uploads, draws, input handling - and assert that they issue sane GL
calls, without touching a real driver.

Usage::

    from voxelcraft.tests import glstub
    gl = glstub.install()
    from voxelcraft.game import VoxelGame
    game = VoxelGame(seed=1, render_distance=2)
    ...
    assert gl.calls_of("glDrawArrays")
"""

from __future__ import annotations

import ctypes
import importlib.util
import pathlib
import sys
import types

import pyglet

#: GL functions that must report success through an ``out`` parameter.
_STATUS_FUNCS = {"glGetShaderiv", "glGetProgramiv"}
#: GL functions whose return value the renderer uses.
_RETURNS = {"glGetUniformLocation": 0, "glGetAttribLocation": 0,
            "glGetShaderInfoLog": b"", "glGetProgramInfoLog": b""}

_GL_STUB = None


def _load_standalone(module_name: str, filename: str):
    """Import a pyglet submodule directly from its file (skips package __init__)."""
    path = pathlib.Path(pyglet.__file__).parent / "window" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class _Device:
    """Stands in for the window's GL context."""

    def __init__(self):
        self.get_state = lambda: "fake"
        self.window = None


class FakeWindow:
    """Minimal stand-in for :class:`pyglet.window.Window`."""

    instances: list["FakeWindow"] = []

    def __init__(self, width: int = 800, height: int = 600, caption: str = "",
                 resizable: bool = False, config=None, vsync: bool = False,
                 fullscreen: bool = False, **kwargs):
        self.width = width
        self.height = height
        self.size = (width, height)
        self.caption = caption
        self.resizable = resizable
        self.config = config
        self.vsync = vsync
        self.fullscreen = fullscreen
        self.exclusive_mouse = False
        self.minimum_size = None
        self.has_exit = False
        self.closed = False
        self.calls: list[str] = []
        FakeWindow.instances.append(self)

    # -- pyglet API used by the game ------------------------------------
    def set_minimum_size(self, width, height):
        self.minimum_size = (width, height)

    def set_exclusive_mouse(self, exclusive=True):
        self.exclusive_mouse = exclusive
        self.calls.append(f"set_exclusive_mouse({exclusive})")

    def set_mouse_visible(self, visible=True):
        pass

    def on_resize(self, width, height):
        self.width, self.height = width, height
        self.size = (width, height)
        return True

    def on_close(self):
        self.has_exit = True
        self.close()

    def close(self):
        self.closed = True

    def dispatch_event(self, name, *args):
        handler = getattr(self, name, None)
        if handler is not None:
            return handler(*args)
        return None


class _Config:
    """Stand-in for :class:`pyglet.gl.Config`."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __repr__(self) -> str:
        return f"FakeConfig({self.kwargs})"


class _Clock:
    def __init__(self):
        self.scheduled: list[tuple[float, object]] = []
        self.unscheduled: list[object] = []

    def schedule_interval(self, func, interval, *args, **kwargs):
        self.scheduled.append((interval, func))
        return func

    def schedule(self, func, *args, **kwargs):
        self.scheduled.append((None, func))
        return func

    def unschedule(self, func, *args, **kwargs):
        self.unscheduled.append(func)

    def tick(self, dt: float = 1 / 60.0) -> None:
        for _interval, func in list(self.scheduled):
            func(dt)


def install() -> types.ModuleType:
    """Install the stubs and return the fake ``pyglet.gl`` module."""
    global _GL_STUB
    if _GL_STUB is not None:
        return _GL_STUB

    fake_gl = types.ModuleType("pyglet.gl")
    fake_gl.__fake__ = True
    fake_gl.calls: list[tuple[str, tuple]] = []
    fake_gl.constants: dict[str, int] = {}

    def _function(name):
        def call(*args, **kwargs):
            fake_gl.calls.append((name, args))
            if name in _STATUS_FUNCS:
                for arg in args:
                    obj = getattr(arg, "_obj", None)
                    if isinstance(obj, (ctypes.c_int, ctypes.c_uint)):
                        obj.value = 1
            return _RETURNS.get(name, 1)
        call.__name__ = name
        return call

    def _constant(name):
        if name not in fake_gl.constants:
            fake_gl.constants[name] = len(fake_gl.constants) + 1
        return fake_gl.constants[name]

    def module_getattr(name):
        if name in ("GLuint", "GLint", "GLenum", "GLsizei", "GLfloat", "GLboolean"):
            return {"GLuint": ctypes.c_uint, "GLint": ctypes.c_int,
                    "GLenum": ctypes.c_uint, "GLsizei": ctypes.c_int,
                    "GLfloat": ctypes.c_float, "GLboolean": ctypes.c_ubyte}[name]
        if name == "Config":
            return _Config
        if name in ("gl_info", "Context"):
            return _Device() if name == "gl_info" else _Device
        if name.startswith("GL_"):
            return _constant(name)
        if name.startswith("gl"):
            return _function(name)
        raise AttributeError(name)

    fake_gl.__getattr__ = module_getattr

    # helper for assertions
    def calls_of(name):
        return [args for fname, args in fake_gl.calls if fname == name]

    def reset():
        fake_gl.calls.clear()

    fake_gl.calls_of = calls_of
    fake_gl.reset = reset

    # --- wire the stubs into the pyglet package -------------------------
    sys.modules["pyglet.gl"] = fake_gl
    setattr(pyglet, "gl", fake_gl)

    key_module = _load_standalone("pyglet.window.key", "key.py")
    mouse_module = _load_standalone("pyglet.window.mouse", "mouse.py")

    fake_window = types.ModuleType("pyglet.window")
    fake_window.__fake__ = True
    fake_window.Window = FakeWindow
    fake_window.key = key_module
    fake_window.mouse = mouse_module
    fake_window.BaseWindow = FakeWindow
    sys.modules["pyglet.window"] = fake_window
    sys.modules["pyglet.window.key"] = key_module
    sys.modules["pyglet.window.mouse"] = mouse_module
    setattr(pyglet, "window", fake_window)

    clock = _Clock()
    fake_clock = types.ModuleType("pyglet.clock")
    fake_clock.__fake__ = True
    fake_clock.schedule_interval = clock.schedule_interval
    fake_clock.schedule = clock.schedule
    fake_clock.unschedule = clock.unschedule
    fake_clock.clock = clock
    sys.modules["pyglet.clock"] = fake_clock
    setattr(pyglet, "clock", fake_clock)
    fake_gl.clock = clock
    fake_gl.window_class = FakeWindow
    fake_gl.key = key_module
    fake_gl.mouse = mouse_module

    _GL_STUB = fake_gl
    return fake_gl


def uninstall() -> None:
    """Undo :func:`install` (mostly useful between test modules)."""
    global _GL_STUB
    for name in ("pyglet.gl", "pyglet.window", "pyglet.window.key",
                 "pyglet.window.mouse", "pyglet.clock"):
        sys.modules.pop(name, None)
    _GL_STUB = None
