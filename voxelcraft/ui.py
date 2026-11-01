"""Backend independent 2D interface batching.

The HUD is drawn with textured quads in screen space (origin top-left, y down).
Each :class:`UIBatch` collects vertices for **one** texture; the renderer binds
the block atlas for shapes and the font atlas for text.

A vertex is 8 floats: ``x, y, u, v, r, g, b, a``.
"""

from __future__ import annotations

import numpy as np

from . import font

FLOATS_PER_VERTEX = 8
VERTS_PER_QUAD = 6

WHITE = (1.0, 1.0, 1.0, 1.0)
BLACK = (0.0, 0.0, 0.0, 1.0)


class UIBatch:
    """Accumulates screen-space quads for a single texture."""

    def __init__(self):
        self.vertices: list[float] = []

    def clear(self) -> None:
        self.vertices.clear()

    @property
    def empty(self) -> bool:
        return not self.vertices

    def add_quad(self, x: float, y: float, w: float, h: float,
                 uv: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
                 color: tuple[float, float, float, float] = WHITE) -> None:
        u0, v0, u1, v1 = uv
        r, g, b, a = color
        x1, y1 = x + w, y + h
        verts = self.vertices
        # two triangles: (x,y) (x1,y) (x1,y1)  and  (x,y) (x1,y1) (x,y1)
        for (px, py, pu, pv) in (
            (x, y, u0, v0),
            (x1, y, u1, v0),
            (x1, y1, u1, v1),
            (x, y, u0, v0),
            (x1, y1, u1, v1),
            (x, y1, u0, v1),
        ):
            verts.extend((px, py, pu, pv, r, g, b, a))

    def add_polygon(self, points, color: tuple[float, float, float, float]) -> None:
        """Add a convex polygon given as ``[(x, y, u, v), ...]`` (triangle fan)."""
        if len(points) < 3:
            return
        r, g, b, a = color
        p0 = points[0]
        for i in range(1, len(points) - 1):
            for (px, py, pu, pv) in (p0, points[i], points[i + 1]):
                self.vertices.extend((px, py, pu, pv, r, g, b, a))

    def add_rect(self, x: float, y: float, w: float, h: float,
                 color: tuple[float, float, float, float],
                 uv: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)) -> None:
        self.add_quad(x, y, w, h, uv, color)

    def add_border(self, x: float, y: float, w: float, h: float, thickness: float,
                   color: tuple[float, float, float, float],
                   uv: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)) -> None:
        self.add_quad(x, y, w, thickness, uv, color)                       # top
        self.add_quad(x, y + h - thickness, w, thickness, uv, color)       # bottom
        self.add_quad(x, y + thickness, thickness, h - 2 * thickness, uv, color)
        self.add_quad(x + w - thickness, y + thickness, thickness, h - 2 * thickness, uv, color)

    def add_text(self, text: str, x: float, y: float, scale: float = 2.0,
                 color: tuple[float, float, float, float] = WHITE,
                 shadow: tuple[float, float, float, float] | None = (0.0, 0.0, 0.0, 0.6),
                 spacing: int = 1) -> float:
        """Draw ``text`` and return the advance width in pixels."""
        step = (font.GLYPH_W + spacing) * scale
        glyph_w = font.GLYPH_W * scale
        glyph_h = font.GLYPH_H * scale
        if shadow is not None:
            off = max(1.0, round(scale * 0.5))
            shadow_color = (0.0, 0.0, 0.0, shadow[3] if len(shadow) > 3 else 0.6)
            for i, ch in enumerate(text):
                if ch == " ":
                    continue
                self.add_quad(x + i * step + off, y + off, glyph_w, glyph_h,
                              font.glyph_uv(ch), shadow_color)
        for i, ch in enumerate(text):
            if ch == " ":
                continue
            self.add_quad(x + i * step, y, glyph_w, glyph_h, font.glyph_uv(ch), color)
        return len(text) * step - spacing * scale if text else 0.0

    def add_text_shadowed(self, text: str, x: float, y: float, scale: float = 2.0,
                          color: tuple[float, float, float, float] = WHITE) -> float:
        """Text with the classic Minecraft drop shadow."""
        return self.add_text(text, x, y, scale, color, shadow=(0.0, 0.0, 0.0, 0.6))

    def to_array(self) -> np.ndarray:
        """Vertices as a contiguous ``(count, 8)`` float32 array."""
        if not self.vertices:
            return np.zeros((0, FLOATS_PER_VERTEX), dtype=np.float32)
        return np.ascontiguousarray(
            np.asarray(self.vertices, dtype=np.float32).reshape(-1, FLOATS_PER_VERTEX)
        )


def color(rgb: tuple[float, float, float], alpha: float = 1.0):
    """Convenience: ``color((0.5, 0.2, 0.1), 0.8)`` -> RGBA tuple."""
    return (rgb[0], rgb[1], rgb[2], alpha)
