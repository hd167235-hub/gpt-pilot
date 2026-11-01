"""Render Voxelcraft screenshots without a GPU.

The tool builds a real world with the real generator, meshes it with the real
mesher and rasterizes the result with :mod:`tools.rasterizer`, so the output is
pixel-comparable to what the OpenGL renderer draws (minus shader details).

Usage::

    python -m tools.render_preview --scene vista --out docs/vista.png
    python -m tools.render_preview --scene all --width 1280 --height 720
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from voxelcraft import blocks as B
from voxelcraft import hud
from voxelcraft.constants import CHUNK_SIZE, SEA_LEVEL, WORLD_HEIGHT
from voxelcraft.player import Player
from voxelcraft.ui import UIBatch
from voxelcraft.world import SOLID_LOOKUP, World
from voxelcraft.worldgen import TerrainGenerator

from .rasterizer import render_world


# ---------------------------------------------------------------------------
# world helpers
# ---------------------------------------------------------------------------
def build_world(seed: int, center: tuple[float, float], radius: int = 6) -> World:
    world = World(seed=seed)
    ccx, ccz = int(center[0]) >> 4, int(center[1]) >> 4
    for cz in range(ccz - radius, ccz + radius + 1):
        for cx in range(ccx - radius, ccx + radius + 1):
            world.generate_chunk(cx, cz)
    return world


class HeightField:
    """Cached heightmap lookup so viewpoint scoring is cheap."""

    def __init__(self, generator: TerrainGenerator, radius: int = 200, step: int = 2):
        self.generator = generator
        self.radius = radius
        self.step = step
        self.size = (2 * radius) // step
        xs = np.arange(-radius, radius, step, dtype=np.float64)
        X, Z = np.meshgrid(xs, xs, indexing="ij")
        self.H = generator.height_at(X, Z)
        self.xs = xs

    def at(self, x, z):
        i = np.clip(((np.asarray(x) + self.radius) / self.step).astype(np.int32),
                    0, self.size - 1)
        j = np.clip(((np.asarray(z) + self.radius) / self.step).astype(np.int32),
                    0, self.size - 1)
        return self.H[i, j]


def column_tops(world: World, x0: int, z0: int, width: int,
                solid_only: bool = False) -> np.ndarray:
    """Highest block per column for a square region (trees included).

    ``solid_only`` ignores water, which gives the seabed under an ocean.
    """
    tops = np.full((width, width), -1, dtype=np.int32)
    for cx in range((x0) >> 4, (x0 + width - 1) >> 4):
        pass
    for (cx, cz), chunk in world.chunks.items():
        bx = cx * CHUNK_SIZE - x0
        bz = cz * CHUNK_SIZE - z0
        if bx + CHUNK_SIZE <= 0 or bz + CHUNK_SIZE <= 0:
            continue
        if bx >= width or bz >= width:
            continue
        if solid_only:
            mask = SOLID_LOOKUP[chunk.blocks]
        else:
            mask = chunk.blocks != B.AIR
        local = np.argmax(mask[:, ::-1, :], axis=1)
        nonair = mask.any(axis=1)
        top = np.where(nonair, WORLD_HEIGHT - 1 - local, -1)
        xs0, xs1 = max(0, bx), min(width, bx + CHUNK_SIZE)
        zs0, zs1 = max(0, bz), min(width, bz + CHUNK_SIZE)
        if xs1 <= xs0 or zs1 <= zs0:
            continue
        tops[xs0:xs1, zs0:zs1] = top[xs0 - bx:xs1 - bx, zs0 - bz:zs1 - bz]
    return tops


def _openness(tops: np.ndarray, x0: int, z0: int, x: float, z: float, yaw: float,
              eye: float) -> tuple[float, float, float]:
    """How open the view is along ``yaw``: (openness, relief, water-ish)."""
    import numpy as np

    fov = math.radians(66)
    size = tops.shape[0]
    angles = yaw + np.linspace(-fov / 2, fov / 2, 11)
    dists = np.array([12.0, 20.0, 30.0, 42.0, 56.0, 72.0, 92.0])
    dirs_x = np.sin(angles)[:, None]
    dirs_z = -np.cos(angles)[:, None]
    px = x + dirs_x * dists[None, :]
    pz = z + dirs_z * dists[None, :]
    i = np.clip((px - x0).astype(np.int32), 0, size - 1)
    j = np.clip((pz - z0).astype(np.int32), 0, size - 1)
    h = tops[i, j].astype(np.float64)
    h[h < 0] = SEA_LEVEL

    above = np.clip(h - (eye - 1.5), 0.0, None)          # things blocking the view
    open_score = float(np.mean(np.clip(1.0 - above / 20.0, 0.0, 1.0)))
    relief = float(np.abs(np.diff(h, axis=1)).sum()) / 100.0
    low = float((h <= SEA_LEVEL).mean())
    return open_score, relief, low


def find_vista(world: World, x0: int, z0: int, width: int, margin: int = 96):
    """Pick the most scenic elevated viewpoint inside a square region."""
    tops = column_tops(world, x0, z0, width)
    flat = tops.copy()
    flat[flat < 0] = 0
    # never place the camera close to the edge of the loaded world
    flat[:margin, :] = 0
    flat[-margin:, :] = 0
    flat[:, :margin] = 0
    flat[:, -margin:] = 0
    order = np.argsort(flat.ravel())[::-1]
    candidates = []
    seen = []
    for idx in order:
        i, j = np.unravel_index(idx, flat.shape)
        if flat[i, j] < SEA_LEVEL + 3:
            break
        x, z = x0 + float(i), z0 + float(j)
        if any(abs(x - sx) < 24 and abs(z - sz) < 24 for sx, sz in seen):
            continue
        seen.append((x, z))
        candidates.append((x, z, float(flat[i, j])))
        if len(candidates) >= 12:
            break

    best = None
    for (x, z, h) in candidates:
        eye = h + 2.2
        for deg in range(0, 360, 10):
            yaw = math.radians(deg)
            open_score, relief, low = _openness(tops, x0, z0, x, z, yaw, eye)
            score = open_score * 100.0 + relief * 12.0 + low * 25.0 + h * 0.35
            if best is None or score > best[0]:
                best = (score, x, h, z, yaw, open_score, relief)

    score, x, h, z, yaw, open_score, relief = best
    print(f"    vista ({x:.0f},{h:.0f},{z:.0f}) yaw {math.degrees(yaw):.0f} "
          f"open {open_score:.2f} relief {relief:.2f} score {score:.0f}")
    return (x, h, z), yaw


def find_cave(world: World, max_y: int = 40, prefer_opening: bool = True):
    """Find a roomy underground air pocket, ideally with a shaft to the surface."""
    best = None
    for (cx, cz), chunk in world.chunks.items():
        blocks = chunk.blocks
        if not (blocks == B.AIR).any():
            continue
        ys, xs, zs = np.nonzero(blocks[:, 3:max_y, :] == B.AIR)
        ys = ys + 3
        keep = (xs > 4) & (xs < 12) & (zs > 4) & (zs < 12)
        xs, ys, zs = xs[keep], ys[keep], zs[keep]
        if len(xs) == 0:
            continue
        roof = blocks[xs, ys + 2, zs] != B.AIR
        floor = blocks[xs, ys - 1, zs] != B.AIR
        xs, ys, zs = xs[roof & floor], ys[roof & floor], zs[roof & floor]
        if len(xs) == 0:
            continue
        step = max(1, len(xs) // 32)
        for x, y, z in zip(xs[::step], ys[::step], zs[::step]):
            box = blocks[max(0, x - 4):x + 5, y - 1:y + 3, max(0, z - 4):z + 5]
            room = int((box == B.AIR).sum())
            sky = int((blocks[x, y + 3:, z] == B.AIR).sum())
            ground = world.top_solid_y(cx * CHUNK_SIZE + int(x), cz * CHUNK_SIZE + int(z),
                                       default=y + 20)
            height = ground - int(y)
            opening = sky >= max(4, height - 3)
            score = room + (60 if (prefer_opening and opening) else 0) + sky * 0.5
            candidate = (score, cx * CHUNK_SIZE + int(x), int(y), cz * CHUNK_SIZE + int(z),
                         opening, room)
            if best is None or candidate[0] > best[0]:
                best = candidate
    if best is None:
        return None
    return best[1], best[2], best[3]


def build_house(world: World, x: int, z: int) -> tuple[float, float, float]:
    """Place a small house so the screenshots show block placement."""
    ground = world.top_solid_y(x, z, default=SEA_LEVEL)
    y0 = ground + 1
    w, d, hgt = 7, 5, 4
    for dx in range(w):
        for dz in range(d):
            for dy in range(hgt):
                edge = dx in (0, w - 1) or dz in (0, d - 1)
                if dy == 0:
                    world.set_block(x + dx, y0, z + dz, B.COBBLE)
                elif edge:
                    world.set_block(x + dx, y0 + dy, z + dz, B.PLANKS)
            if not (dx in (0, w - 1) or dz in (0, d - 1)):
                world.set_block(x + dx, y0 + hgt, z + dz, B.BRICK)
                world.set_block(x + dx, y0, z + dz, B.PLANKS)
    # windows and a door
    for dx in (2, 4):
        world.set_block(x + dx, y0 + 2, z, B.GLASS)
        world.set_block(x + dx, y0 + 2, z + d - 1, B.GLASS)
    world.set_block(x + 3, y0 + 1, z, B.AIR)
    world.set_block(x + 3, y0 + 2, z, B.AIR)
    # roof
    for dx in range(-1, w + 1):
        for dz in range(-1, d + 1):
            world.set_block(x + dx, y0 + hgt + 1, z + dz, B.BRICK)
    return (x + w / 2, y0 + 2.6, z + d / 2)


# ---------------------------------------------------------------------------
# scenes
# ---------------------------------------------------------------------------
def scene_vista(seed: int, width: int, height: int, with_hud: bool):
    world = build_world(seed, (0, 0), radius=10)
    (x, h, z), yaw = find_vista(world, -160, -160, 320, margin=110)
    player = Player(world, position=(x + 0.5, h + 2.0, z + 0.5), yaw=yaw,
                    pitch=math.radians(-7))
    player.flying = True
    return world, player, 9, dict(with_hud=with_hud)


def scene_beach(seed: int, width: int, height: int, with_hud: bool):
    """A shoreline: sand underfoot, the sea and its horizon ahead."""
    generator = TerrainGenerator(seed)
    world = build_world(seed, (0, 0), radius=5)
    xs = np.arange(-76, 76, 2, dtype=np.float64)
    X, Z = np.meshgrid(xs, xs, indexing="ij")
    H = generator.height_at(X, Z)

    best = None
    for i in range(0, X.shape[0], 2):
        for j in range(0, X.shape[1], 2):
            h = float(H[i, j])
            if not (SEA_LEVEL <= h <= SEA_LEVEL + 3):
                continue
            for deg in range(0, 360, 15):
                rad = math.radians(deg)
                dirx, dirz = math.sin(rad), -math.cos(rad)
                d = np.array([7.0, 11.0, 16.0, 22.0, 30.0, 40.0])
                probe = generator.height_at(X[i, j] + dirx * d, Z[i, j] + dirz * d)
                water = int((probe < SEA_LEVEL).sum())
                if water < 5:
                    continue
                score = water * 10.0 - abs(h - (SEA_LEVEL + 1)) * 4.0
                if best is None or score > best[0]:
                    best = (score, float(X[i, j]), float(Z[i, j]), h, rad, water)
    if best is None:
        raise SystemExit("no beach found for this seed")
    _, x, z, h, yaw, water = best
    print(f"    beach ({x:.0f},{h:.0f},{z:.0f}) yaw {math.degrees(yaw):.0f} water {water}")
    player = Player(world, position=(x + 0.5, h + 1.0, z + 0.5), yaw=yaw,
                    pitch=math.radians(-5))
    player.flying = True
    return world, player, 5, dict(with_hud=with_hud)


def _ray_clear(world: World, origin, direction, distance: float) -> bool:
    """True when nothing solid is between ``origin`` and ``origin + dir*distance``."""
    hit = world.raycast(origin, direction, distance)
    return hit is None


#: the default seed spawns on a sandy island, so the forest shot uses a seed
#: with dense woodland right next to the origin.
FOREST_SEED = 2024


def scene_forest(seed: int, width: int, height: int, with_hud: bool):
    """Eye level in the woods, framed so several trunks are actually visible."""
    seed = FOREST_SEED
    radius = 6
    world = build_world(seed, (0, 0), radius=radius)
    limit = radius * CHUNK_SIZE - 26

    trunks = []
    for (cx, cz), chunk in world.chunks.items():
        for (lx, _ly, lz) in np.argwhere(chunk.blocks == B.LOG)[::2]:
            trunks.append((cx * CHUNK_SIZE + int(lx), cz * CHUNK_SIZE + int(lz)))
    if not trunks:
        raise SystemExit("no trees found for this seed")
    trunks = np.array(trunks, dtype=np.float64)
    inner = trunks[(np.abs(trunks[:, 0]) <= limit) & (np.abs(trunks[:, 1]) <= limit)]
    if len(inner) == 0:
        raise SystemExit("no trees inside the loaded area")

    best = None
    # sample open, dry ground and score every direction by the trunks in view
    for cx in range(-limit, limit, 3):
        for cz in range(-limit, limit, 3):
            ground = world.top_solid_y(cx, cz, default=SEA_LEVEL)
            if ground <= SEA_LEVEL + 1:
                continue
            above = world.get_block(cx, ground + 2, cz)
            if above != B.AIR:
                continue                                   # standing in a tree
            eye = np.array([cx + 0.5, ground + 1.7, cz + 0.5])
            for deg in range(0, 360, 20):
                rad = math.radians(deg)
                direction = np.array([math.sin(rad), 0.0, -math.cos(rad)])
                if not _ray_clear(world, eye, direction, 16.0):
                    continue                               # a wall blocks the view
                dx = inner[:, 0] + 0.5 - eye[0]
                dz = inner[:, 1] + 0.5 - eye[2]
                distance = np.hypot(dx, dz)
                forward = dx * direction[0] + dz * direction[2]
                side = np.abs(dx * direction[2] - dz * direction[0])
                in_cone = (forward > 3.0) & (forward < 22.0) & (side / np.maximum(distance, 1e-6) < 0.75)
                visible = int(in_cone.sum())
                if visible < 3:
                    continue
                mean_distance = float(distance[in_cone].mean())
                score = visible * 4.0 - abs(mean_distance - 8.0)
                if best is None or score > best[0]:
                    best = (score, cx, cz, ground, rad, visible)
    if best is None:
        raise SystemExit("could not frame the forest")
    _, cx, cz, ground, yaw, visible = best
    player = Player(world, position=(cx + 0.5, ground + 1.02, cz + 0.5), yaw=yaw,
                    pitch=math.radians(2))
    player.flying = True
    print(f"    forest ({cx},{ground},{cz}) yaw {math.degrees(yaw):.0f} "
          f"trunks in view {visible}")
    return world, player, 5, dict(with_hud=with_hud)


def scene_cave(seed: int, width: int, height: int, with_hud: bool):
    """Underground: the largest air pocket found in the generated area."""
    world = build_world(seed, (0, 0), radius=4)
    spot = find_cave(world)
    if spot is None:
        raise SystemExit("no cave found in this seed")
    x, y, z = spot
    # look towards the nearest opening, or straight ahead when there is none
    best = None
    for deg in range(0, 360, 15):
        rad = math.radians(deg)
        direction = np.array([math.sin(rad), 0.0, -math.cos(rad)])
        if not _ray_clear(world, (x + 0.5, y + 1.6, z + 0.5), direction, 7.0):
            continue
        probe = (x + 0.5 + direction[0] * 6, y + 1.6, z + 0.5 + direction[2] * 6)
        sky = 0
        for dy in range(1, 40):
            if world.get_block(int(probe[0]), int(probe[1]) + dy, int(probe[2])) == B.AIR:
                sky += 1
            else:
                break
        if best is None or sky > best[0]:
            best = (sky, rad)
    yaw = best[1] if best else math.radians(35)
    player = Player(world, position=(x + 0.5, y, z + 0.5), yaw=yaw,
                    pitch=math.radians(8))
    player.flying = True
    print(f"    cave ({x},{y},{z}) yaw {math.degrees(yaw):.0f}"
          f" sky above exit {best[0] if best else 0}")
    return world, player, 3, dict(with_hud=with_hud, cave=True)


def scene_house(seed: int, width: int, height: int, with_hud: bool):
    """A player-built house with a cobblestone path - shows off block editing."""
    world = build_world(seed, (0, 0), radius=6)
    tops = column_tops(world, -96, -96, 192)

    # pick the flattest buildable spot that is not underwater
    best = None
    for x in range(-80, 80, 2):
        for z in range(-80, 80, 2):
            h = int(tops[x + 96, z + 96])
            if not (SEA_LEVEL + 2 <= h <= SEA_LEVEL + 26):
                continue
            window = tops[x + 96 - 7:x + 96 + 8, z + 96 - 7:z + 96 + 8]
            if window.size == 0:
                continue
            flatness = int(window.max()) - int(window.min())
            clearance = tops[x + 96 - 7:x + 96 - 3, z + 96 - 14:z + 96 - 6]
            if clearance.size and clearance.max() > h + 3:
                continue                     # no room to stand in front
            if best is None or flatness < best[0]:
                best = (flatness, x, z, h)
    if best is None:
        raise SystemExit("no flat spot for the house")
    _, hx, hz, h = best
    hx, hz = hx - 3, hz - 2
    build_house(world, hx, hz)
    for i in range(1, 8):
        world.set_block(hx + 3, h, hz - i, B.COBBLE)

    centre = np.array([hx + 3.5, h + 3.0, hz + 2.5])
    camera = None
    for deg in range(0, 360, 10):
        rad = math.radians(deg)
        for distance in (15.0, 13.0, 17.0, 11.0):
            x = centre[0] + math.sin(rad) * distance
            z = centre[2] - math.cos(rad) * distance
            ground = world.top_solid_y(int(x), int(z), default=SEA_LEVEL)
            if abs(ground - h) > 3:
                continue
            eye = np.array([x, ground + 1.7, z])
            if world.get_block(int(x), int(eye[1]), int(z)) != B.AIR:
                continue
            if not _ray_clear(world, eye, centre - eye, float(np.linalg.norm(centre - eye))):
                continue
            camera = (x, ground, z)
            break
        if camera:
            break
    if camera is None:
        raise SystemExit("no camera position with a clear view of the house")
    x, ground, z = camera
    player = Player(world, position=(x, ground + 1.02, z), yaw=0.0)
    player.look_at(centre)
    player.flying = True
    print(f"    house at ({hx},{h},{hz}) flatness {best[0]} camera ({x:.0f},{ground},{z:.0f})")
    return world, player, 6, dict(with_hud=with_hud)


def scene_underwater(seed: int, width: int, height: int, with_hud: bool):
    """Dive into the shallows: the water surface above, the seabed below."""
    world = build_world(seed, (0, 0), radius=5)
    half = 80
    tops = column_tops(world, -half, -half, half * 2, solid_only=True)

    # prefer a shallow shelf next to a drop-off - lots of depth variation in view
    best = None
    for x in range(-half + 6, half - 6, 2):
        for z in range(-half + 6, half - 6, 2):
            window = tops[x + half - 5: x + half + 6, z + half - 5: z + half + 6]
            if window.size == 0:
                continue
            deepest = int(window.max())
            depth = SEA_LEVEL - deepest
            if not (4 <= depth <= 12):
                continue
            variation = int(window.max() - window.min())
            score = variation * 2.0 - abs(depth - 6)
            if best is None or score > best[0]:
                best = (score, x, z, depth)
    if best is None:
        raise SystemExit("no shallow reef for this seed")
    _, x, z, depth = best
    player = Player(world, position=(x + 0.5, SEA_LEVEL - 2.2, z + 0.5),
                    yaw=math.radians(150), pitch=math.radians(24))
    player.flying = True
    player.head_in_water = True
    print(f"    underwater ({x},{z}) depth {depth}")
    return world, player, 5, dict(with_hud=with_hud, underwater=True)


SCENES = {
    "vista": scene_vista,
    "beach": scene_beach,
    "forest": scene_forest,
    "cave": scene_cave,
    "house": scene_house,
    "underwater": scene_underwater,
}


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------
def build_hud(width: int, height: int, player, world, with_debug: bool = False):
    shapes, text = UIBatch(), UIBatch()
    hud.draw_crosshair(shapes, width, height)
    hud.draw_hotbar(shapes, text, width, height, 0)
    hud.draw_position_indicator(shapes, text, width, height, player.yaw, player.pitch)
    if with_debug:
        lines = [
            "*Voxelcraft 1.0   seed %d" % world.seed,
            "FPS  60.0   frame 16.7 ms",
            "XYZ %7.2f %6.2f %7.2f" % tuple(player.position),
            "Chunks %d loaded" % world.loaded_count(),
        ]
        hud.draw_debug(shapes, text, width, height, lines)
    return shapes, text


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def render_scene(name: str, seed: int, width: int, height: int,
                 with_hud: bool = True, with_debug: bool = False) -> str:
    started = time.time()
    world, player, radius, opts = SCENES[name](seed, width, height, with_hud)
    shapes = text = None
    if with_hud:
        shapes, text = build_hud(width, height, player, world, with_debug)
    renderer = render_world(world, player, width=width, height=height,
                            radius=radius, ui_shapes=shapes, ui_text=text,
                            underwater=opts.get("underwater"))
    if opts.get("cave"):
        renderer.fog_color = np.array([0.03, 0.035, 0.045], dtype=np.float32)
        renderer.fog_end = 34.0
        renderer.fog_start = 10.0
    path = f"docs/shots/{name}.png"
    import os

    os.makedirs("docs/shots", exist_ok=True)
    renderer.save(path)
    print(f"{name:8s} -> {path}  ({renderer.triangles_drawn} triangles, "
          f"{time.time() - started:.1f}s)")
    return path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Render Voxelcraft screenshots")
    parser.add_argument("--scene", default="vista",
                        help=", ".join(SCENES) + " or 'all'")
    parser.add_argument("--seed", type=int, default=20260201)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-hud", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    names = list(SCENES) if args.scene == "all" else [args.scene]
    for name in names:
        render_scene(name, args.seed, args.width, args.height,
                     with_hud=not args.no_hud, with_debug=args.debug)


if __name__ == "__main__":  # pragma: no cover
    main()
