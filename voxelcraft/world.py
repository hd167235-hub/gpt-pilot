"""Chunk storage, block access, streaming, player edits and voxel raycasting."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from . import blocks as B
from .constants import CHUNK_SIZE, REACH, WORLD_HEIGHT
from .worldgen import TerrainGenerator

_PAD = 1  # meshing padding (see mesher.py)
PADDED_X = CHUNK_SIZE + 2 * _PAD
PADDED_Y = WORLD_HEIGHT + 2 * _PAD
PADDED_Z = CHUNK_SIZE + 2 * _PAD

# Lookup tables for fast per-voxel queries (run-length independent of the registry).
_OPAQUE = np.zeros(B.NUM_BLOCKS, dtype=bool)
_SOLID = np.zeros(B.NUM_BLOCKS, dtype=bool)
_LIQUID = np.zeros(B.NUM_BLOCKS, dtype=bool)
_TRANSPARENT = np.zeros(B.NUM_BLOCKS, dtype=bool)
for _b in B.BLOCKS:
    _OPAQUE[_b.id] = bool(_b.opaque)
    _SOLID[_b.id] = bool(_b.solid)
    _LIQUID[_b.id] = bool(_b.liquid)
    _TRANSPARENT[_b.id] = bool(_b.transparent)

OPAQUE_LOOKUP = _OPAQUE
SOLID_LOOKUP = _SOLID
LIQUID_LOOKUP = _LIQUID
TRANSPARENT_LOOKUP = _TRANSPARENT


@dataclass
class RaycastHit:
    """Result of :meth:`World.raycast`."""

    block: tuple[int, int, int]      # the block that was hit
    normal: tuple[int, int, int]     # face normal pointing back towards the ray
    block_id: int
    distance: float

    @property
    def placement(self) -> tuple[int, int, int]:
        """Where a new block would go when right-clicking."""
        return (
            self.block[0] + self.normal[0],
            self.block[1] + self.normal[1],
            self.block[2] + self.normal[2],
        )


class Chunk:
    """A ``CHUNK_SIZE`` x ``WORLD_HEIGHT`` x ``CHUNK_SIZE`` column of blocks."""

    __slots__ = ("cx", "cz", "blocks", "version", "has_edits")

    def __init__(self, cx: int, cz: int, blocks: np.ndarray):
        self.cx = cx
        self.cz = cz
        self.blocks = blocks
        self.version = 0          # bumped whenever a block changes
        self.has_edits = False

    @property
    def origin(self) -> tuple[int, int, int]:
        return self.cx * CHUNK_SIZE, 0, self.cz * CHUNK_SIZE

    def top_solid_y(self, lx: int, lz: int) -> int:
        """Highest non-air block in a column (-1 if the column is empty)."""
        column = self.blocks[lx, :, lz]
        nz = np.flatnonzero(column)
        return int(nz[-1]) if nz.size else -1

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Chunk {self.cx},{self.cz} v{self.version}>"


class World:
    """A dictionary of chunks plus the (sparse) log of player edits."""

    def __init__(self, seed: int = 20260201,
                 generator: Optional[TerrainGenerator] = None):
        self.seed = int(seed)
        self.generator = generator or TerrainGenerator(self.seed)
        self.chunks: dict[tuple[int, int], Chunk] = {}
        # edits[chunk] = {(lx, y, lz): block_id}
        self.edits: dict[tuple[int, int], dict[tuple[int, int, int], int]] = {}

    # -- chunk access ----------------------------------------------------
    def get_chunk(self, cx: int, cz: int) -> Optional[Chunk]:
        return self.chunks.get((cx, cz))

    def chunk_at_block(self, x: int, z: int) -> Optional[Chunk]:
        return self.chunks.get((x >> 4, z >> 4))

    def generate_chunk(self, cx: int, cz: int) -> Chunk:
        """Generate (or return the existing) chunk at ``cx, cz``."""
        existing = self.chunks.get((cx, cz))
        if existing is not None:
            return existing
        return self.add_generated_chunk(cx, cz, self.generator.generate_chunk(cx, cz))

    def add_generated_chunk(self, cx: int, cz: int, data: np.ndarray) -> Chunk:
        """Install freshly generated blocks, re-applying any stored player edits.

        Called on the main thread after a worker finished (see streaming.py).
        """
        chunk = Chunk(cx, cz, data)
        stored = self.edits.get((cx, cz))
        if stored:
            for (lx, y, lz), block_id in stored.items():
                if 0 <= y < WORLD_HEIGHT:
                    data[lx, y, lz] = block_id
            chunk.has_edits = True
        self.chunks[(cx, cz)] = chunk
        return chunk

    # -- blocks ----------------------------------------------------------
    def get_block(self, x: int, y: int, z: int) -> int:
        """Block id at world coordinates (AIR outside the loaded world)."""
        if y < 0 or y >= WORLD_HEIGHT:
            return B.AIR
        chunk = self.chunks.get((x >> 4, z >> 4))
        if chunk is None:
            return B.AIR
        return int(chunk.blocks[x & 15, y, z & 15])

    def is_solid(self, x: int, y: int, z: int) -> bool:
        return bool(_SOLID[self.get_block(x, y, z)])

    def is_opaque(self, x: int, y: int, z: int) -> bool:
        return bool(_OPAQUE[self.get_block(x, y, z)])

    def is_liquid(self, x: int, y: int, z: int) -> bool:
        return bool(_LIQUID[self.get_block(x, y, z)])

    def set_block(self, x: int, y: int, z: int, block_id: int,
                  record: bool = True) -> bool:
        """Write a block, bumping chunk versions. Returns True if it changed."""
        if y < 0 or y >= WORLD_HEIGHT:
            return False
        cx, cz = x >> 4, z >> 4
        chunk = self.chunks.get((cx, cz))
        if chunk is None:
            return False
        lx, lz = x & 15, z & 15
        if chunk.blocks[lx, y, lz] == block_id:
            return False
        chunk.blocks[lx, y, lz] = block_id
        chunk.version += 1
        chunk.has_edits = True
        # Neighbouring chunks need a remesh when we touch a border column so that
        # ambient occlusion and hidden faces stay correct.
        if lx == 0:
            self._touch(cx - 1, cz)
        elif lx == CHUNK_SIZE - 1:
            self._touch(cx + 1, cz)
        if lz == 0:
            self._touch(cx, cz - 1)
        elif lz == CHUNK_SIZE - 1:
            self._touch(cx, cz + 1)
        if record:
            self.edits.setdefault((cx, cz), {})[(lx, y, lz)] = block_id
        return True

    def _touch(self, cx: int, cz: int) -> None:
        chunk = self.chunks.get((cx, cz))
        if chunk is not None:
            chunk.version += 1

    def height_at(self, x: float, z: float) -> int:
        """Terrain generator height (ignores player edits)."""
        return int(self.generator.height_at(x, z))

    def top_solid_y(self, x: int, z: int, default: int = 0) -> int:
        """Highest non-air block at ``x, z``; falls back to the noise height."""
        chunk = self.chunks.get((x >> 4, z >> 4))
        if chunk is None:
            return default
        y = chunk.top_solid_y(x & 15, z & 15)
        return y if y >= 0 else default

    # -- streaming -------------------------------------------------------
    def missing_chunks(self, center_cx: int, center_cz: int,
                       radius: int) -> list[tuple[int, int, int]]:
        """(distance^2, cx, cz) for every chunk inside ``radius`` not yet built."""
        out = []
        for dz in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                d2 = dx * dx + dz * dz
                if d2 > (radius + 0.5) ** 2:
                    continue
                cx, cz = center_cx + dx, center_cz + dz
                if (cx, cz) not in self.chunks:
                    out.append((d2, cx, cz))
        out.sort()
        return out

    def stream(self, center_cx: int, center_cz: int, radius: int,
               budget: int = 2) -> int:
        """Generate at most ``budget`` of the closest missing chunks."""
        created = 0
        for _, cx, cz in self.missing_chunks(center_cx, center_cz, radius):
            if created >= budget:
                break
            self.generate_chunk(cx, cz)
            created += 1
        return created

    def unload_far(self, center_cx: int, center_cz: int, keep_radius: int) -> list[tuple[int, int]]:
        """Drop chunks outside ``keep_radius`` and return their coordinates."""
        limit = (keep_radius + 0.5) ** 2
        removed = []
        for (cx, cz) in list(self.chunks):
            dx, dz = cx - center_cx, cz - center_cz
            if dx * dx + dz * dz > limit:
                del self.chunks[(cx, cz)]
                removed.append((cx, cz))
        return removed

    def loaded_count(self) -> int:
        return len(self.chunks)

    def edit_count(self) -> int:
        return sum(len(v) for v in self.edits.values())

    # -- meshing helper --------------------------------------------------
    def padded_blocks(self, chunk: Chunk) -> np.ndarray:
        """Chunk blocks surrounded by a one-block border from the neighbours.

        Missing neighbours are filled with stone so the outer shell of unloaded
        chunks does not show a wall of faces (it gets remeshed once the
        neighbour arrives - see ``Chunk.version``).
        """
        out = np.empty((PADDED_X, PADDED_Y, PADDED_Z), dtype=np.uint8)
        out[:] = B.STONE
        out[:, -1, :] = B.AIR          # open sky above the world
        out[:, 0, :] = B.STONE
        out[1:-1, 1:-1, 1:-1] = chunk.blocks

        cx, cz = chunk.cx, chunk.cz
        chunks = self.chunks
        # Edge columns (padded index 0 -> neighbour local 15, padded -1 -> local 0).
        west = chunks.get((cx - 1, cz))
        east = chunks.get((cx + 1, cz))
        north = chunks.get((cx, cz - 1))
        south = chunks.get((cx, cz + 1))
        if west is not None:
            out[0, 1:-1, 1:-1] = west.blocks[15, :, :]
        if east is not None:
            out[-1, 1:-1, 1:-1] = east.blocks[0, :, :]
        if north is not None:
            out[1:-1, 1:-1, 0] = north.blocks[:, :, 15]
        if south is not None:
            out[1:-1, 1:-1, -1] = south.blocks[:, :, 0]
        # Diagonals only matter for ambient occlusion at chunk corners.
        for (ncx, ncz, col, ncol) in (
            (cx - 1, cz - 1, 0, 0), (cx - 1, cz + 1, 0, -1),
            (cx + 1, cz - 1, -1, 0), (cx + 1, cz + 1, -1, -1),
        ):
            diagonal = chunks.get((ncx, ncz))
            if diagonal is None:
                continue
            nx = 15 if col == 0 else 0
            nz = 15 if ncol == 0 else 0
            out[col, 1:-1, ncol] = diagonal.blocks[nx, :, nz]
        return out

    # -- raycasting ------------------------------------------------------
    def raycast(self, origin, direction, max_distance: float = REACH,
                hit_liquid: bool = False) -> Optional[RaycastHit]:
        """Voxel traversal (Amanatides & Woo). Returns the first *solid* block."""
        o = np.asarray(origin, dtype=np.float64)
        d = np.asarray(direction, dtype=np.float64)
        norm = np.linalg.norm(d)
        if norm < 1e-9:
            return None
        d = d / norm

        cell = [int(math.floor(o[0])), int(math.floor(o[1])), int(math.floor(o[2]))]
        step = [0 if abs(d[i]) < 1e-9 else (1 if d[i] > 0 else -1) for i in range(3)]
        t_max = [float("inf")] * 3
        t_delta = [float("inf")] * 3
        for i in range(3):
            if step[i] > 0:
                t_max[i] = (cell[i] + 1 - o[i]) / d[i]
                t_delta[i] = 1.0 / d[i]
            elif step[i] < 0:
                t_max[i] = (cell[i] - o[i]) / d[i]
                t_delta[i] = -1.0 / d[i]

        normal = [0, 0, 0]
        distance = 0.0
        # Guard against pathological loops.
        for _ in range(int(max_distance * 4) + 8):
            block_id = self.get_block(cell[0], cell[1], cell[2])
            defn = B.BY_ID.get(block_id)
            if block_id != B.AIR and defn is not None:
                if defn.solid or (hit_liquid and defn.liquid):
                    return RaycastHit((cell[0], cell[1], cell[2]),
                                      (normal[0], normal[1], normal[2]),
                                      block_id, distance)
            if min(t_max) > max_distance:
                return None
            axis = int(np.argmin(t_max))
            distance = t_max[axis]
            if distance > max_distance:
                return None
            cell[axis] += step[axis]
            t_max[axis] += t_delta[axis]
            normal = [0, 0, 0]
            normal[axis] = -step[axis]
        return None

    # -- persistence -----------------------------------------------------
    def save(self, path: str) -> int:
        """Write seed + edits to a ``.npz`` file. Returns the edit count."""
        entries = []
        for (cx, cz), chunk_edits in self.edits.items():
            for (lx, y, lz), block_id in chunk_edits.items():
                entries.append((cx * CHUNK_SIZE + lx, y, cz * CHUNK_SIZE + lz, block_id))
        edits = np.asarray(entries, dtype=np.int32).reshape(-1, 4)
        meta = {"version": 1, "seed": self.seed, "edits": int(edits.shape[0])}
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        np.savez_compressed(path, edits=edits,
                            meta=np.array(json.dumps(meta), dtype=np.str_))
        return len(entries)

    @classmethod
    def load(cls, path: str) -> "World":
        """Load a world saved with :meth:`save`."""
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            edits = data["edits"].reshape(-1, 4)
        world = cls(seed=int(meta.get("seed", 20260201)))
        for x, y, z, block_id in edits:
            x, y, z, block_id = int(x), int(y), int(z), int(block_id)
            cx, cz = x >> 4, z >> 4
            world.edits.setdefault((cx, cz), {})[(x & 15, y, z & 15)] = block_id
        return world

    # -- misc ------------------------------------------------------------
    def iter_chunks(self) -> Iterator[Chunk]:
        return iter(self.chunks.values())

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<World seed={self.seed} chunks={len(self.chunks)} edits={self.edit_count()}>"
