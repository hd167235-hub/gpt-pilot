"""Procedural terrain generation.

The world is generated chunk by chunk from layered Perlin noise:

``continent``  very slow fBm deciding ocean from land (~±20 blocks)
``hills``      medium fBm giving rolling hills (~±16 blocks)
``micro``      fast fBm adding block-scale bumpiness (~±5 blocks)
``mountains``  masked ridged noise for sharp 80-120 block peaks
``caves``      two 3D fBm fields intersected to carve winding tunnels
``trees``      deterministic hashing on a 6 block grid

Everything is a pure function of ``(seed, x, z)`` so chunks can be generated in
any order (and regenerated later) without seams.  Trees use a hash of their
grid cell, therefore a tree that straddles a chunk border is grown identically
in both chunks.
"""

from __future__ import annotations

import numpy as np

from . import blocks as B
from .constants import CHUNK_SIZE, SEA_LEVEL, WORLD_HEIGHT
from .noise import PerlinNoise, hash01, smoothstep

#: coarse grid (in blocks) used to decide where trees grow
TREE_GRID = 6
TREE_CHANCE = 0.46
SNOW_LINE = SEA_LEVEL + 56


class TerrainGenerator:
    """Deterministic, seedable chunk generator."""

    def __init__(self, seed: int = 20260201):
        self.seed = int(seed)
        self.noise = PerlinNoise(self.seed)
        self.cave_noise_a = PerlinNoise(self.seed * 7 + 1)
        self.cave_noise_b = PerlinNoise(self.seed * 7 + 2)

    # -- heights ---------------------------------------------------------
    def height_at(self, x, z) -> np.ndarray:
        """Terrain surface height (the y of the topmost solid block)."""
        x = np.asarray(x, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        n = self.noise

        continent = n.fbm2(x / 700.0 + 0.5, z / 700.0 - 0.5, octaves=4) * 46.0
        hills = n.fbm2(x / 120.0 + 13.5, z / 120.0 - 7.25, octaves=4) * 16.0
        micro = n.fbm2(x / 38.0 - 3.7, z / 38.0 + 5.1, octaves=3) * 5.0
        mountain_mask = smoothstep(
            0.0, 0.36,
            n.fbm2(x / 190.0 + 19.7, z / 190.0 + 71.3, octaves=3) * 2.7,
        )
        ridges = n.ridged2(x / 190.0 - 3.1, z / 190.0 + 8.7, octaves=5,
                           sharpness=2.0)

        h = (
            SEA_LEVEL
            + 2.0
            + continent
            + hills
            + micro
            + mountain_mask ** 1.15 * ridges ** 2 * 78.0
        )
        return np.clip(np.floor(h), 3, WORLD_HEIGHT - 10).astype(np.int32)

    def _temperature(self, x, z) -> np.ndarray:
        """Slow 'how cold is it here' field, in blocks of altitude."""
        return np.asarray(
            self.noise.fbm2(np.asarray(x) / 900.0 + 4.2, np.asarray(z) / 900.0 - 2.6,
                            octaves=2), dtype=np.float64
        ) * 9.0

    # -- chunk generation ------------------------------------------------
    def generate_chunk(self, cx: int, cz: int) -> np.ndarray:
        """Build one chunk.

        Returns a ``(CHUNK_SIZE, WORLD_HEIGHT, CHUNK_SIZE)`` uint8 array indexed
        ``[x, y, z]`` with local coordinates ``0 <= x,z < 16``.
        """
        ox, oz = cx * CHUNK_SIZE, cz * CHUNK_SIZE
        size = CHUNK_SIZE
        out = np.zeros((size, WORLD_HEIGHT, size), dtype=np.uint8)

        xs = np.arange(size, dtype=np.float64) + ox
        zs = np.arange(size, dtype=np.float64) + oz
        wx, wz = np.meshgrid(xs, zs, indexing="ij")     # (x, z)
        heights = self.height_at(wx, wz)                # (x, z)
        h = heights[:, None, :]                         # (x, 1, z)
        y = np.arange(WORLD_HEIGHT, dtype=np.int32)[None, :, None]
        depth = h - y

        solid = y <= h
        out[solid] = B.STONE

        # sand near the water line (beaches, lake beds, sea floor)
        beach_band = h <= SEA_LEVEL + 2
        skin = solid & (depth < 3)
        out[skin & ~beach_band] = B.DIRT
        out[solid & (depth < 4) & beach_band] = B.SAND

        # surface block
        surface = solid & (depth == 0)
        out[surface] = B.GRASS
        out[surface & beach_band] = B.SAND
        out[surface & (h <= SEA_LEVEL - 6)] = B.SAND     # deep sea floor

        # snow on high or cold ground (snowy peaks look great from a distance)
        cold = surface & ((h > SNOW_LINE) | ((h > SEA_LEVEL + 20)
                                            & (self._temperature(wx, wz)[:, None, :] > 6.0)))
        out[cold] = B.SNOW

        # bedrock floor
        out[:, 0, :] = B.BEDROCK
        out[:, 1, :] = np.where(out[:, 1, :] != B.AIR, B.BEDROCK, out[:, 1, :])

        self._carve_caves(out, ox, oz, heights)

        # Oceans: fill empty cells that sit above the terrain surface and at or
        # below sea level.  Caves below sea level stay dry, which is both faster
        # and avoids water flooding the whole underground.
        water = (out == B.AIR) & (y <= SEA_LEVEL) & (y > h)
        out[water] = B.WATER

        self._plant_trees(out, ox, oz)
        return out

    def _carve_caves(self, out: np.ndarray, ox: int, oz: int,
                     heights: np.ndarray) -> None:
        """Carve winding tunnels and a few deep caverns.

        Only the band between the bedrock and the highest column of this chunk is
        sampled - caves above the surface are invisible anyway and skipping them
        roughly halves the cost of generating a chunk.
        """
        size = CHUNK_SIZE
        y_lo = 2
        y_hi = int(heights.max())
        if y_hi <= y_lo + 4:
            return
        ys = np.arange(y_lo, y_hi)

        wx, wy, wz = np.meshgrid(
            np.arange(size) + ox,
            ys,
            np.arange(size) + oz,
            indexing="ij",
        )
        n1 = self.cave_noise_a.fbm3(wx / 42.0, wy / 30.0, wz / 42.0, octaves=2)
        n2 = self.cave_noise_b.fbm3(wx / 42.0 + 50.0, wy / 30.0 - 30.0,
                                    wz / 42.0 + 90.0, octaves=2)
        carve = (np.abs(n1) < 0.070) & (np.abs(n2) < 0.070)

        # a few big caverns deep down (single octave: they are low frequency)
        cavern_band = ys < 40
        if cavern_band.any():
            cavern = self.cave_noise_a.fbm3(wx[:, cavern_band, :] / 70.0 + 11.0,
                                            wy[:, cavern_band, :] / 40.0,
                                            wz[:, cavern_band, :] / 70.0 - 23.0,
                                            octaves=1)
            carve[:, cavern_band, :] |= cavern > 0.32

        # keep the surface intact: never carve the top two blocks of a column
        h = heights[:, None, :]
        y_grid = ys[None, :, None]
        carve &= y_grid < h - 2

        region = out[:, y_lo:y_hi, :]
        region[carve] = B.AIR

    def _plant_trees(self, out: np.ndarray, ox: int, oz: int) -> None:
        """Grow trees whose trunk may live in a neighbouring chunk."""
        size = CHUNK_SIZE
        g0x = (ox - 3) // TREE_GRID - 1
        g1x = (ox + size + 3) // TREE_GRID + 1
        g0z = (oz - 3) // TREE_GRID - 1
        g1z = (oz + size + 3) // TREE_GRID + 1

        GX, GZ = np.meshgrid(np.arange(g0x, g1x + 1), np.arange(g0z, g1z + 1),
                             indexing="ij")
        seed = self.seed
        jitter_x = hash01(GX, GZ, seed, seed=11)
        jitter_z = hash01(GX, GZ, seed, seed=23)
        roll = hash01(GX, GZ, seed, seed=37)
        kind = hash01(GX, GZ, seed, seed=53)

        root_x = (GX * TREE_GRID + (jitter_x * TREE_GRID).astype(np.int64)) - ox
        root_z = (GZ * TREE_GRID + (jitter_z * TREE_GRID).astype(np.int64)) - oz
        ground = self.height_at(root_x + ox, root_z + oz).astype(np.int64)
        wants_tree = (roll < TREE_CHANCE) & (ground > SEA_LEVEL + 1) & (ground < SNOW_LINE - 4)

        for i in range(root_x.size):
            if not wants_tree.flat[i]:
                continue
            lx = int(root_x.flat[i])
            lz = int(root_z.flat[i])
            if not (-3 <= lx <= size + 2 and -3 <= lz <= size + 2):
                continue
            gy = int(ground.flat[i])
            if not (1 <= gy < WORLD_HEIGHT - 10):
                continue
            if 0 <= lx < size and 0 <= lz < size and out[lx, gy, lz] != B.GRASS:
                continue          # no trees on sand, snow or stone
            trunk = 4 + int(kind.flat[i] * 4)
            self._grow_tree(out, lx, gy, lz, trunk)

    @staticmethod
    def _grow_tree(out: np.ndarray, lx: int, gy: int, lz: int, trunk: int) -> None:
        size = CHUNK_SIZE
        top = gy + trunk

        def put(x, y, z, block):
            if 0 <= x < size and 0 <= z < size and 0 <= y < WORLD_HEIGHT:
                current = out[x, y, z]
                if current == B.AIR or (block == B.LOG and current in (B.AIR, B.LEAVES)):
                    out[x, y, z] = block

        for dy in range(1, trunk + 1):
            put(lx, gy + dy, lz, B.LOG)

        # classic oak canopy: two wide rings around the top, then a cap
        for dy in range(-2, 2):
            radius = 2 if dy < 1 else 1
            for dx in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if abs(dx) == radius and abs(dz) == radius:
                        if (dx * 3 + dz * 5 + dy) % 3 == 0:
                            continue          # ragged corners
                    if dx == 0 and dz == 0 and dy < 1:
                        continue              # keep the trunk
                    put(lx + dx, top + dy, lz + dz, B.LEAVES)
        put(lx, top + 1, lz, B.LEAVES)

    # -- helpers ---------------------------------------------------------
    def height_map(self, x0: int, z0: int, width: int, depth: int,
                   step: int = 1) -> np.ndarray:
        """Height array for a region (used by the preview tools and tests)."""
        xs = np.arange(x0, x0 + width, step, dtype=np.float64)
        zs = np.arange(z0, z0 + depth, step, dtype=np.float64)
        X, Z = np.meshgrid(xs, zs, indexing="ij")
        return self.height_at(X, Z)

    def find_spawn(self, prefer_x: float = 0.0, prefer_z: float = 0.0,
                   radius: int = 96) -> tuple[float, float, float]:
        """Find dry, reasonably flat land near ``(prefer_x, prefer_z)``."""
        best = None
        for r in range(4, radius, 8):
            for dx in range(-r, r + 1, max(4, r // 2)):
                for dz in range(-r, r + 1, max(4, r // 2)):
                    x = int(prefer_x + dx)
                    z = int(prefer_z + dz)
                    h = int(self.height_at(x, z))
                    if not (SEA_LEVEL + 2 <= h <= SEA_LEVEL + 30):
                        continue
                    neighbourhood = self.height_at(
                        x + np.array([-2, 2, 0, 0]), z + np.array([0, 0, -2, 2])
                    )
                    flatness = float(np.abs(neighbourhood - h).max())
                    score = flatness - 0.02 * r
                    if best is None or score < best[0]:
                        best = (score, x, z, h)
            if best is not None and r >= 24:
                break
        if best is None:
            h = int(self.height_at(prefer_x, prefer_z))
            return (prefer_x + 0.5, float(max(h, SEA_LEVEL) + 2), prefer_z + 0.5)
        _, x, z, h = best
        return (float(x) + 0.5, float(h) + 1.2, float(z) + 0.5)
