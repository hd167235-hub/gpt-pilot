"""Procedural texture atlas.

All block textures are generated at start-up with NumPy - the project ships no
binary assets.  Tiles are laid out in a 4x4 grid; every tile is padded by
``PAD`` pixels (the edge pixel is repeated) so that mipmapping and bilinear
filtering cannot bleed one tile into its neighbour.

Tile indices are defined in :mod:`voxelcraft.blocks` (``T_*``).
"""

from __future__ import annotations

import numpy as np

from .constants import ATLAS_TILES, TILE_PX

PAD = 2
CELL = TILE_PX + 2 * PAD
ATLAS_SIZE = ATLAS_TILES * CELL

# --- palette --------------------------------------------------------------
GRASS = (94, 157, 52)
GRASS_DARK = (72, 128, 40)
GRASS_LIGHT = (116, 178, 66)
DIRT = (134, 96, 67)
DIRT_DARK = (110, 78, 54)
DIRT_LIGHT = (152, 112, 80)
STONE = (128, 128, 132)
STONE_DARK = (104, 104, 110)
STONE_LIGHT = (150, 150, 155)
SAND = (219, 207, 160)
SAND_DARK = (198, 186, 140)
SAND_LIGHT = (233, 224, 182)
WOOD = (109, 82, 50)
WOOD_DARK = (86, 63, 38)
WOOD_LIGHT = (128, 98, 61)
LEAF = (58, 110, 40)
LEAF_DARK = (40, 84, 28)
LEAF_LIGHT = (78, 136, 52)
BRICK = (150, 68, 54)
BRICK_DARK = (124, 54, 43)
MORTAR = (176, 168, 160)
SNOW = (238, 244, 250)
SNOW_DARK = (216, 226, 238)
BEDROCK = (52, 52, 56)
WATER = (56, 110, 190)


def _blank(color=(0, 0, 0, 0)) -> np.ndarray:
    """A transparent 16x16 RGBA tile."""
    tile = np.zeros((TILE_PX, TILE_PX, 4), dtype=np.uint8)
    tile[:, :, :3] = color[:3]
    tile[:, :, 3] = color[3] if len(color) > 3 else 255
    return tile


def _fill(tile, color):
    tile[:, :, :3] = color[:3]
    return tile


def _speckle(tile, rng, colors, count, size=1):
    """Scatter small random blobs of ``colors`` over the tile (seamless)."""
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    for _ in range(count):
        x = rng.integers(0, TILE_PX)
        y = rng.integers(0, TILE_PX)
        c = colors[rng.integers(0, len(colors))]
        w = size if rng.random() > 0.25 else size + 1
        for dy in range(w):
            for dx in range(w):
                tile[(y + dy) % TILE_PX, (x + dx) % TILE_PX, :3] = c


def _noise_shade(tile, rng, amount=8):
    """Add fine per-pixel brightness noise (quantised, pixel-art friendly)."""
    n = rng.integers(-amount, amount + 1, size=(TILE_PX, TILE_PX, 1))
    rgb = tile[:, :, :3].astype(np.int16) + n
    tile[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)


# --- individual tiles -----------------------------------------------------
def _grass_top(rng):
    t = _blank((*GRASS, 255))
    _speckle(t, rng, [GRASS_DARK, GRASS_LIGHT, (86, 146, 46)], 40, 1)
    _noise_shade(t, rng, 10)
    return t


def _grass_side(rng):
    t = _blank((*DIRT, 255))
    _speckle(t, rng, [DIRT_DARK, DIRT_LIGHT], 26)
    _noise_shade(t, rng, 7)
    # A ragged band of grass along the top edge.
    for x in range(TILE_PX):
        depth = 3 + int(rng.integers(0, 3))
        if rng.random() < 0.18:
            depth -= 2
        for y in range(max(1, depth)):
            t[y, x, :3] = GRASS if (x + y) % 7 else GRASS_DARK
    t[0, :, :3] = GRASS_LIGHT
    return t


def _dirt_tile(rng):
    t = _blank((*DIRT, 255))
    _speckle(t, rng, [DIRT_DARK, DIRT_LIGHT], 30)
    _noise_shade(t, rng, 9)
    return t


def _stone_tile(rng):
    t = _blank((*STONE, 255))
    _speckle(t, rng, [STONE_DARK, STONE_LIGHT], 34)
    # a couple of hairline cracks
    for _ in range(2):
        x, y = int(rng.integers(0, TILE_PX)), int(rng.integers(0, TILE_PX))
        for _ in range(6):
            t[y % TILE_PX, x % TILE_PX, :3] = STONE_DARK
            x += int(rng.integers(-1, 2))
            y += 1
    _noise_shade(t, rng, 6)
    return t


def _cobble_tile(rng):
    """Irregular rounded stones separated by dark mortar."""
    t = _blank((*STONE_DARK, 255))
    # jittered grid of stone centres -> pseudo-Voronoi cobbles
    centres = []
    for gy in range(3):
        for gx in range(3):
            cx = (gx + 0.5) * TILE_PX / 3.0 + rng.uniform(-1.6, 1.6)
            cy = (gy + 0.5) * TILE_PX / 3.0 + rng.uniform(-1.6, 1.6)
            shade = int(rng.integers(-18, 20))
            centres.append((cx, cy, shade))
    ys, xs = np.mgrid[0:TILE_PX, 0:TILE_PX]
    best = np.full((TILE_PX, TILE_PX), 1e9)
    owner = np.zeros((TILE_PX, TILE_PX), dtype=np.int32)
    for i, (cx, cy, _shade) in enumerate(centres):
        # wrap distances so the tile stays seamless
        dx = np.abs(xs + 0.5 - cx)
        dy = np.abs(ys + 0.5 - cy)
        dx = np.minimum(dx, TILE_PX - dx)
        dy = np.minimum(dy, TILE_PX - dy)
        d = np.sqrt(dx * dx + dy * dy)
        closer = d < best
        best = np.where(closer, d, best)
        owner = np.where(closer, i, owner)

    base = np.array(STONE, dtype=np.float64)
    for i, (_cx, _cy, shade) in enumerate(centres):
        mask = owner == i
        if not mask.any():
            continue
        col = np.clip(base + shade, 0, 255)
        t[mask, :3] = col.astype(np.uint8)
    # stones bulge: brighten the top-left, darken the bottom-right
    gy, gx = np.gradient(best)
    relief = (gx + gy) * 26.0
    rgb = t[:, :, :3].astype(np.float64) + relief[:, :, None]
    t[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    # mortar in the gaps between stones
    mortar = best > 1.55
    t[mortar, :3] = STONE_DARK
    _speckle(t, rng, [STONE_LIGHT, STONE], 14)
    _noise_shade(t, rng, 5)
    return t


def _sand_tile(rng):
    t = _blank((*SAND, 255))
    _speckle(t, rng, [SAND_DARK, SAND_LIGHT], 34)
    _noise_shade(t, rng, 8)
    return t


def _log_side(rng):
    t = _blank((*WOOD, 255))
    _noise_shade(t, rng, 6)
    for x in range(TILE_PX):
        if x % 4 == 0:
            t[:, x, :3] = WOOD_DARK
        elif x % 4 == 2:
            t[:, x, :3] = WOOD_LIGHT
    for _ in range(14):
        x = int(rng.integers(0, TILE_PX))
        y = int(rng.integers(0, TILE_PX))
        t[y, x, :3] = WOOD_DARK
    return t


def _log_top(rng):
    t = _blank((*WOOD_LIGHT, 255))
    _noise_shade(t, rng, 5)
    c = (TILE_PX - 1) / 2
    for y in range(TILE_PX):
        for x in range(TILE_PX):
            d = max(abs(x - c), abs(y - c))
            ring = int(d) % 3
            if ring == 0:
                t[y, x, :3] = WOOD
            elif ring == 1:
                t[y, x, :3] = WOOD_DARK
    # bark outline
    t[0, :, :3] = WOOD_DARK
    t[-1, :, :3] = WOOD_DARK
    t[:, 0, :3] = WOOD_DARK
    t[:, -1, :3] = WOOD_DARK
    return t


def _leaves_tile(rng):
    t = _blank((*LEAF, 255))
    _speckle(t, rng, [LEAF_DARK, LEAF_LIGHT], 60)
    _noise_shade(t, rng, 14)
    # a few "holes" rendered dark instead of transparent (fast leaves)
    for _ in range(5):
        x, y = int(rng.integers(0, TILE_PX)), int(rng.integers(0, TILE_PX))
        t[y, x, :3] = LEAF_DARK
    return t


def _planks_tile(rng):
    t = _blank((*WOOD_LIGHT, 255))
    for row in range(0, TILE_PX, 4):
        shade = int(rng.integers(-10, 11))
        col = tuple(np.clip(np.array(WOOD_LIGHT) + shade, 0, 255).astype(int))
        t[row:row + 4, :, :3] = col
        t[row, :, :3] = WOOD_DARK               # seam
        seam_x = int(rng.integers(2, TILE_PX - 2))
        t[row:row + 4, seam_x, :3] = WOOD_DARK  # staggered plank join
    # grain
    for _ in range(22):
        y = int(rng.integers(0, TILE_PX))
        x = int(rng.integers(0, TILE_PX))
        if t[y, x, 0] != WOOD_DARK[0]:
            t[y, x, :3] = WOOD
    _noise_shade(t, rng, 4)
    return t


def _brick_tile(rng):
    t = _blank((*MORTAR, 255))
    row_h = 4
    for row in range(0, TILE_PX, row_h):
        offset = 0 if (row // row_h) % 2 == 0 else 4
        x = -offset
        while x < TILE_PX:
            bw = 7
            shade = int(rng.integers(-12, 13))
            col = tuple(np.clip(np.array(BRICK) + shade, 0, 255).astype(int))
            y0 = row
            y1 = min(row + row_h - 1, TILE_PX)
            x0 = max(x, 0)
            x1 = min(x + bw, TILE_PX)
            if x1 > x0 and y1 > y0:
                t[y0:y1, x0:x1, :3] = col
            x += bw + 1
        t[row + row_h - 1, :, :3] = MORTAR
    _noise_shade(t, rng, 5)
    return t


def _glass_tile(rng):
    t = _blank((205, 232, 246, 0))
    t[:, :, 3] = 0
    # frame
    t[0, :, 3] = 210
    t[-1, :, 3] = 210
    t[:, 0, 3] = 210
    t[:, -1, 3] = 210
    t[0, :, :3] = (235, 245, 250)
    t[-1, :, :3] = (235, 245, 250)
    t[:, 0, :3] = (235, 245, 250)
    t[:, -1, :3] = (235, 245, 250)
    # diagonal highlight
    for i in range(3, 9):
        if 0 <= i < TILE_PX and 0 <= i - 3 < TILE_PX:
            t[i - 3, TILE_PX - 1 - i, :3] = (240, 250, 255)
            t[i - 3, TILE_PX - 1 - i, 3] = 150
    # faint body tint
    body = t[:, :, 3] == 0
    t[body, :3] = (196, 224, 240)
    t[body, 3] = 28
    return t


def _water_tile(rng):
    t = _blank((*WATER, 255))
    t[:, :, 3] = 165
    _noise_shade(t, rng, 12)
    for y in range(TILE_PX):
        for x in range(TILE_PX):
            wave = (x * 2 + y * 3) % 11
            if wave == 0:
                t[y, x, :3] = (92, 146, 214)
            elif wave == 5:
                t[y, x, :3] = (44, 92, 168)
    return t


def _snow_tile(rng):
    t = _blank((*SNOW, 255))
    _speckle(t, rng, [SNOW_DARK, (255, 255, 255)], 30)
    _noise_shade(t, rng, 5)
    return t


def _bedrock_tile(rng):
    t = _blank((*BEDROCK, 255))
    shades = [(30, 30, 34), (70, 70, 76), (92, 92, 98), (44, 44, 48)]
    for _ in range(70):
        x, y = int(rng.integers(0, TILE_PX)), int(rng.integers(0, TILE_PX))
        c = shades[int(rng.integers(0, len(shades)))]
        w = 1 + int(rng.random() < 0.3)
        t[y:y + w, x:x + w, :3] = c
    _noise_shade(t, rng, 8)
    return t


def _white_tile():
    t = _blank((255, 255, 255, 255))
    return t


_BUILDERS = {
    0: _grass_top,
    1: _grass_side,
    2: _dirt_tile,
    3: _stone_tile,
    4: _cobble_tile,
    5: _sand_tile,
    6: _log_side,
    7: _log_top,
    8: _leaves_tile,
    9: _planks_tile,
    10: _brick_tile,
    11: _glass_tile,
    12: _water_tile,
    13: _snow_tile,
    14: _bedrock_tile,
    15: lambda rng: _white_tile(),
}


def build_atlas(seed: int = 1337) -> np.ndarray:
    """Generate the padded RGBA atlas.

    Returns an ``(ATLAS_SIZE, ATLAS_SIZE, 4)`` uint8 array ordered row-major so
    that tile index ``i`` sits at column ``i % 4``, row ``i // 4``.
    """
    atlas = np.zeros((ATLAS_SIZE, ATLAS_SIZE, 4), dtype=np.uint8)
    for index in range(ATLAS_TILES * ATLAS_TILES):
        rng = np.random.default_rng(seed + index * 7919)
        builder = _BUILDERS.get(index)
        tile = builder(rng) if builder is not None else _blank((255, 0, 255, 255))
        cx = (index % ATLAS_TILES) * CELL
        cy = (index // ATLAS_TILES) * CELL
        # paste with clamped (edge repeating) padding
        padded = np.pad(tile, ((PAD, PAD), (PAD, PAD), (0, 0)), mode="edge")
        atlas[cy:cy + CELL, cx:cx + CELL] = padded
    return atlas


def tile_uv(index: int, size: int = ATLAS_SIZE) -> tuple[float, float, float, float]:
    """UV rectangle ``(u0, v0, u1, v1)`` of the *core* of a tile.

    ``v`` is flipped because OpenGL textures start at the bottom left while the
    atlas array is stored top-down.
    """
    cx = (index % ATLAS_TILES) * CELL + PAD
    cy = (index // ATLAS_TILES) * CELL + PAD
    u0 = cx / size
    u1 = (cx + TILE_PX) / size
    # flip vertically
    v0 = 1.0 - (cy + TILE_PX) / size
    v1 = 1.0 - cy / size
    return u0, v0, u1, v1


def save_png(path: str, seed: int = 1337) -> None:
    """Debug helper: write the atlas to a PNG file."""
    from PIL import Image

    Image.fromarray(build_atlas(seed), "RGBA").resize(
        (ATLAS_SIZE * 8, ATLAS_SIZE * 8), Image.NEAREST
    ).save(path)
