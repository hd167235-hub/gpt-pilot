"""Heads-up display: crosshair, hotbar, debug overlay, menus and toasts.

Everything is drawn with two :class:`voxelcraft.ui.UIBatch` objects - one using
the block atlas (rectangles and block icons) and one using the font atlas
(text) - so the HUD needs nothing but textured quads.
"""

from __future__ import annotations

import math

from . import blocks as B
from . import font
from .textures import tile_uv

WHITE_TILE = tile_uv(B.T_WHITE)

SLOT_SIZE = 46
SLOT_GAP = 4
HOTBAR_SLOTS = len(B.HOTBAR_BLOCKS)

PANEL = (0.0, 0.0, 0.0, 0.45)
PANEL_BORDER = (1.0, 1.0, 1.0, 0.30)
TEXT = (1.0, 1.0, 1.0, 1.0)
TEXT_DIM = (0.72, 0.76, 0.80, 1.0)
ACCENT = (0.55, 0.85, 0.35, 1.0)


def _hotbar_width() -> int:
    return HOTBAR_SLOTS * SLOT_SIZE + (HOTBAR_SLOTS - 1) * SLOT_GAP


def draw_rect(shapes, x, y, w, h, color):
    shapes.add_quad(x, y, w, h, WHITE_TILE, color)


def draw_panel(shapes, x, y, w, h, color=PANEL, border=PANEL_BORDER, thickness=2):
    draw_rect(shapes, x, y, w, h, color)
    if border is not None:
        shapes.add_border(x, y, w, h, thickness, border, WHITE_TILE)


def draw_crosshair(shapes, width, height, color=(1.0, 1.0, 1.0, 0.85)):
    cx, cy = width / 2.0, height / 2.0
    thickness = 2
    arm = 9
    gap = 3
    outline = (0.0, 0.0, 0.0, 0.45)
    for dx, dy, w, h in (
        (-gap - arm, -thickness / 2, arm, thickness),
        (gap, -thickness / 2, arm, thickness),
        (-thickness / 2, -gap - arm, thickness, arm),
        (-thickness / 2, gap, thickness, arm),
    ):
        draw_rect(shapes, cx + dx - 1, cy + dy - 1, w + 2, h + 2, outline)
        draw_rect(shapes, cx + dx, cy + dy, w, h, color)


def draw_block_icon(shapes, block_id: int, cx: float, cy: float, size: float):
    """Draw an isometric cube using the block's top and side textures.

    ``cx, cy`` is the centre of the icon and ``size`` its full width.
    """
    defn = B.BY_ID[block_id]
    top_tile = defn.top if defn.top is not None else B.T_WHITE
    side_tile = defn.side if defn.side is not None else B.T_WHITE
    tu0, tv0, tu1, tv1 = tile_uv(top_tile)
    su0, sv0, su1, sv1 = tile_uv(side_tile)

    half = size / 2.0
    qh = size * 0.27            # half height of the top diamond
    drop = size * 0.42          # length of the vertical cube edge
    top_y = cy - (2.0 * qh + drop) / 2.0

    top = (cx, top_y)
    right = (cx + half, top_y + qh)
    bottom = (cx, top_y + 2.0 * qh)
    left = (cx - half, top_y + qh)

    # top face (brightest)
    shapes.add_polygon([
        (top[0], top[1], tu0, tv0),
        (right[0], right[1], tu1, tv0),
        (bottom[0], bottom[1], tu1, tv1),
        (left[0], left[1], tu0, tv1),
    ], (1.0, 1.0, 1.0, 1.0))

    # front-right face
    shapes.add_polygon([
        (bottom[0], bottom[1], su0 + (su1 - su0) * 0.15, sv1),
        (right[0], right[1], su1, sv1),
        (right[0], right[1] + drop, su1, sv0),
        (bottom[0], bottom[1] + drop, su0 + (su1 - su0) * 0.15, sv0),
    ], (0.76, 0.76, 0.76, 1.0))

    # front-left face (in shadow)
    shapes.add_polygon([
        (left[0], left[1], su0, sv1),
        (bottom[0], bottom[1], su1 - (su1 - su0) * 0.15, sv1),
        (bottom[0], bottom[1] + drop, su1 - (su1 - su0) * 0.15, sv0),
        (left[0], left[1] + drop, su0, sv0),
    ], (0.55, 0.55, 0.55, 1.0))


def draw_hotbar(shapes, text, width, height, selected: int, block_names=None):
    total = _hotbar_width()
    x0 = (width - total) / 2.0
    y0 = height - SLOT_SIZE - 12
    for i, block_id in enumerate(B.HOTBAR_BLOCKS):
        x = x0 + i * (SLOT_SIZE + SLOT_GAP)
        active = i == selected
        draw_rect(shapes, x - 2, y0 - 2, SLOT_SIZE + 4, SLOT_SIZE + 4,
                  (1.0, 1.0, 1.0, 0.95) if active else (0.0, 0.0, 0.0, 0.40))
        draw_rect(shapes, x, y0, SLOT_SIZE, SLOT_SIZE,
                  (0.55, 0.55, 0.55, 0.55) if active else (0.15, 0.15, 0.18, 0.35))
        shapes.add_border(x, y0, SLOT_SIZE, SLOT_SIZE, 2,
                          (0.1, 0.1, 0.1, 0.7), WHITE_TILE)

        draw_block_icon(shapes, block_id, x + SLOT_SIZE / 2.0, y0 + SLOT_SIZE / 2.0 - 2,
                        SLOT_SIZE - 18)

        # slot number
        text.add_text(str(i + 1), x + 3, y0 + 3, 1.0, (1, 1, 1, 0.65))
        if active:
            label = B.BY_ID[block_id].name
            lw = font.text_width(label, 2.0, 1)
            text.add_text(label, x + SLOT_SIZE / 2.0 - lw / 2.0, y0 - 22, 2.0,
                          (1, 1, 1, 0.95), shadow=(0, 0, 0, 0.7))

    return x0, y0


def draw_block_label(text, shapes, width, height, label):
    if not label:
        return
    scale = 2.0
    w = font.text_width(label, 2, 1) + 16
    x = (width - w) / 2.0
    y = height - SLOT_SIZE - 46
    draw_rect(shapes, x, y, w, 26, (0.0, 0.0, 0.0, 0.45))
    text.add_text(label, x + 8, y + 6, scale, TEXT)


def draw_debug(shapes, text, width, height, lines, title="Voxelcraft"):
    scale = 2.0
    pad = 8
    line_h = 18
    widest = max([font.text_width(line, 2, 1) for line in lines] + [font.text_width(title, 2, 1)])
    panel_w = widest + pad * 2
    panel_h = len(lines) * line_h + pad * 2
    draw_panel(shapes, 8, 8, panel_w, panel_h, (0.0, 0.0, 0.0, 0.55))
    for i, line in enumerate(lines):
        color = ACCENT if line.startswith("*") else TEXT_DIM
        text.add_text(line.lstrip("*"), 8 + pad, 8 + pad + i * line_h, scale, color)


def draw_toast(shapes, text, width, height, message, timer=1.0):
    if not message:
        return
    scale = 2.0
    w = font.text_width(message, 2, 1) + 20
    x = (width - w) / 2.0
    y = 60
    draw_rect(shapes, x, y, w, 28, (0.0, 0.0, 0.0, 0.5 * min(1.0, timer * 2)))
    text.add_text(message, x + 10, y + 7, scale, TEXT)


def draw_underwater(shapes, width, height):
    draw_rect(shapes, 0, 0, width, height, (0.12, 0.34, 0.55, 0.35))


def draw_hurt_flash(shapes, width, height, alpha):
    if alpha > 0.01:
        draw_rect(shapes, 0, 0, width, height, (0.7, 0.05, 0.05, alpha))


def draw_title_screen(shapes, text, width, height, subtitle=None, lines=(), footer=None):
    panel_w = min(width - 80, 760)
    panel_h = 120 + len(lines) * 22 + (36 if subtitle else 0)
    x = (width - panel_w) / 2.0
    y = (height - panel_h) / 2.0
    draw_panel(shapes, x, y, panel_w, panel_h, (0.03, 0.05, 0.08, 0.82),
               (1.0, 1.0, 1.0, 0.25), 2)

    title = "VOXELCRAFT"
    tw = font.text_width(title, 6, 2)
    text.add_text(title, (width - tw) / 2.0 + 3, y + 26 + 3, 6.0, (0.0, 0.0, 0.0, 0.5), shadow=None)
    text.add_text(title, (width - tw) / 2.0, y + 26, 6.0, (0.62, 0.88, 0.35, 1.0), shadow=None)

    cursor = y + 26 + 52
    if subtitle:
        sw = font.text_width(subtitle, 2, 1)
        text.add_text(subtitle, (width - sw) / 2.0, cursor, 2.0, (0.85, 0.9, 0.95, 1.0))
        cursor += 34

    for i, line in enumerate(lines):
        if line.startswith("##"):
            lw = font.text_width(line[2:].strip(), 2, 1)
            text.add_text(line[2:].strip(), (width - lw) / 2.0, cursor, 2.0, ACCENT)
        else:
            text.add_text(line, x + 44, cursor, 2.0, TEXT_DIM)
        cursor += 22

    if footer:
        fw = font.text_width(footer, 2, 1)
        text.add_text(footer, (width - fw) / 2.0, y + panel_h - 34, 2.0,
                      (1.0, 0.9, 0.4, 0.85))


def draw_loading(shapes, text, width, height, progress: float, message: str = "Generating terrain"):
    draw_rect(shapes, 0, 0, width, height, (0.06, 0.09, 0.13, 1.0))
    title = "VOXELCRAFT"
    tw = font.text_width(title, 6, 2)
    text.add_text(title, (width - tw) / 2.0, height / 2 - 80, 6.0, (0.62, 0.88, 0.35, 1.0))
    bar_w, bar_h = min(width - 160, 520), 22
    x = (width - bar_w) / 2.0
    y = height / 2 + 10
    draw_rect(shapes, x, y, bar_w, bar_h, (0.0, 0.0, 0.0, 0.5))
    draw_rect(shapes, x + 2, y + 2, max(0.0, (bar_w - 4) * progress), bar_h - 4,
              (0.55, 0.85, 0.35, 1.0))
    shapes.add_border(x, y, bar_w, bar_h, 2, (1, 1, 1, 0.35), WHITE_TILE)
    msg = f"{message}... {int(progress * 100)}%"
    mw = font.text_width(msg, 2, 1)
    text.add_text(msg, (width - mw) / 2.0, y + bar_h + 14, 2.0, TEXT_DIM)


def draw_position_indicator(shapes, text, width, height, yaw: float, pitch: float):
    """Compass strip with a tick every 15 degrees and the four cardinal points."""
    strip_w = 300
    x = (width - strip_w) / 2.0
    y = 10
    draw_rect(shapes, x, y, strip_w, 22, (0.0, 0.0, 0.0, 0.30))
    shapes.add_border(x, y, strip_w, 22, 1, (1, 1, 1, 0.18), WHITE_TILE)
    degrees = math.degrees(yaw) % 360.0
    span = 90.0   # degrees visible across the strip

    for tick in range(0, 360, 15):
        delta = ((tick - degrees + 180) % 360) - 180
        if abs(delta) > span / 2:
            continue
        px = x + strip_w / 2.0 + (delta / (span / 2.0)) * (strip_w / 2.0)
        major = tick % 90 == 0
        h = 8 if major else 4
        draw_rect(shapes, px - 0.5, y + 22 - h, 1.2, h, (1, 1, 1, 0.55 if major else 0.3))

    for label, deg in (("N", 0), ("NE", 45), ("E", 90), ("SE", 135),
                       ("S", 180), ("SW", 225), ("W", 270), ("NW", 315)):
        delta = ((deg - degrees + 180) % 360) - 180
        if abs(delta) > span / 2 - 6:
            continue
        px = x + strip_w / 2.0 + (delta / (span / 2.0)) * (strip_w / 2.0)
        color = (1.0, 0.55, 0.45, 0.95) if label == "N" else (1, 1, 1, 0.85)
        lw = font.text_width(label, 1.5, 1)
        text.add_text(label, px - lw / 2.0, y + 3, 1.5, color)

    # centre marker
    draw_rect(shapes, x + strip_w / 2.0 - 1, y + 18, 2, 6, (0.55, 0.85, 0.35, 0.95))
