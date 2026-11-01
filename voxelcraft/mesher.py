"""Vectorised chunk meshing.

For every block face that is exposed we emit two triangles.  The work is done
with NumPy so a chunk is built in a few milliseconds:

1.  Compute the six "is this face visible" masks for the whole chunk at once
    (comparing each block against its neighbour in that direction).
2.  ``np.nonzero`` turns each mask into a list of faces; only a few thousand
    exist even for a busy chunk, so the per-face work (ambient occlusion, light)
    is cheap.
3.  Expand each face's four corners into six vertices.

Vertex layout (6 floats, 24 bytes)::

    x, y, z      position in world space
    u, v         atlas texture coordinates
    light        baked brightness (face shading * sky light * ambient occlusion)

Two passes are produced: opaque geometry and alpha blended geometry (water,
glass).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import blocks as B
from .constants import CHUNK_SIZE, WORLD_HEIGHT
from .textures import tile_uv
from .world import OPAQUE_LOOKUP, TRANSPARENT_LOOKUP

VERTEX_FLOATS = 6
FLOATS_PER_VERTEX = VERTEX_FLOATS

# padded chunk strides (see world.PADDED_*)
_STRIDE_X = (WORLD_HEIGHT + 2) * (CHUNK_SIZE + 2)
_STRIDE_Y = CHUNK_SIZE + 2


@dataclass(frozen=True)
class Face:
    """Static description of one cube face direction."""

    index: int
    axis: int
    sign: int
    normal: tuple[int, int, int]
    tile_kind: str                     # "top" | "side" | "bottom"
    corners: np.ndarray                # (4, 3) unit-cube corner offsets
    uvs: np.ndarray                    # (4, 2) texture coordinates in [0, 1]
    ao_offsets: np.ndarray             # (4, 3, 3) [[side1], [side2], [corner]]
    brightness: float


def _make_face(index, axis, sign, p, t1, t2, uvs, tile_kind, brightness) -> Face:
    p = np.asarray(p, dtype=np.float64)
    t1 = np.asarray(t1, dtype=np.float64)
    t2 = np.asarray(t2, dtype=np.float64)
    corners = np.array([p, p + t1, p + t1 + t2, p + t2], dtype=np.float64)
    normal = np.zeros(3)
    normal[axis] = sign
    # tangent signs per corner, derived from the corner position along t1/t2
    ao = np.zeros((4, 3, 3), dtype=np.int32)
    for i in range(4):
        a1 = 1 if corners[i].dot(t1) > 0.5 else -1
        a2 = 1 if corners[i].dot(t2) > 0.5 else -1
        d1 = (t1 * a1).astype(np.int32)
        d2 = (t2 * a2).astype(np.int32)
        ao[i, 0] = d1
        ao[i, 1] = d2
        ao[i, 2] = d1 + d2
    return Face(index, axis, sign, (int(normal[0]), int(normal[1]), int(normal[2])),
                tile_kind, corners.astype(np.float32),
                np.asarray(uvs, dtype=np.float32), ao, brightness)


# Face order: +X, -X, +Y, -Y, +Z, -Z
FACES: tuple[Face, ...] = (
    _make_face(0, 0, +1, (1, 0, 0), (0, 1, 0), (0, 0, 1),
               [(1, 1), (1, 0), (0, 0), (0, 1)], "side", 0.74),
    _make_face(1, 0, -1, (0, 0, 0), (0, 0, 1), (0, 1, 0),
               [(0, 1), (1, 1), (1, 0), (0, 0)], "side", 0.74),
    _make_face(2, 1, +1, (0, 1, 0), (0, 0, 1), (1, 0, 0),
               [(0, 0), (0, 1), (1, 1), (1, 0)], "top", 1.0),
    _make_face(3, 1, -1, (0, 0, 0), (1, 0, 0), (0, 0, 1),
               [(0, 1), (1, 1), (1, 0), (0, 0)], "bottom", 0.50),
    _make_face(4, 2, +1, (0, 0, 1), (1, 0, 0), (0, 1, 0),
               [(0, 1), (1, 1), (1, 0), (0, 0)], "side", 0.88),
    _make_face(5, 2, -1, (0, 0, 0), (0, 1, 0), (1, 0, 0),
               [(1, 1), (1, 0), (0, 0), (0, 1)], "side", 0.88),
)

#: brightness multiplier for each of the four ambient occlusion levels
AO_LEVELS = np.array([0.55, 0.72, 0.86, 1.0], dtype=np.float32)
AO_QUAD_INDEX = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)


def _tile_table(kind: str) -> np.ndarray:
    table = np.full(B.NUM_BLOCKS, B.T_WHITE, dtype=np.int32)
    for defn in B.BLOCKS:
        tile = getattr(defn, kind, None)
        if tile is not None:
            table[defn.id] = tile
    return table


_TILE_TOP = _tile_table("top")
_TILE_SIDE = _tile_table("side")
_TILE_BOTTOM = _tile_table("bottom")
_TILE_BY_KIND = {"top": _TILE_TOP, "side": _TILE_SIDE, "bottom": _TILE_BOTTOM}

# UV rectangles for every atlas tile, (16, 4)
_TILE_UV = np.array([tile_uv(i) for i in range(16)], dtype=np.float32)


def column_tops(blocks: np.ndarray) -> np.ndarray:
    """Highest non-air y per column, ``-1`` for empty columns. Shape (x, z)."""
    nonair = blocks != B.AIR
    any_block = nonair.any(axis=1)
    reversed_first = np.argmax(nonair[:, ::-1, :], axis=1)
    tops = (WORLD_HEIGHT - 1 - reversed_first).astype(np.int32)
    return np.where(any_block, tops, -1)


def _build_pass(padded: np.ndarray, cx: int, cz: int, transparent: bool,
                tops: np.ndarray) -> np.ndarray:
    """Build the vertex array for one render pass."""
    cur = padded[1:-1, 1:-1, 1:-1]
    origin_x, origin_z = cx * CHUNK_SIZE, cz * CHUNK_SIZE

    if transparent:
        cur_flag = TRANSPARENT_LOOKUP[cur]
    else:
        cur_flag = OPAQUE_LOOKUP[cur]

    chunk_vertices: list[np.ndarray] = []
    occ = OPAQUE_LOOKUP[padded]  # occluders for ambient occlusion

    for face in FACES:
        nx, ny, nz = face.normal
        nb = padded[1 + nx:1 + nx + CHUNK_SIZE,
                    1 + ny:1 + ny + WORLD_HEIGHT,
                    1 + nz:1 + nz + CHUNK_SIZE]
        nb_opaque = OPAQUE_LOOKUP[nb]
        if transparent:
            mask = cur_flag & ~nb_opaque & (nb != cur)
        else:
            mask = cur_flag & ~nb_opaque

        idx = np.nonzero(mask)
        count = idx[0].size
        if count == 0:
            continue

        fx = idx[0].astype(np.int32)
        fy = idx[1].astype(np.int32)
        fz = idx[2].astype(np.int32)
        block_ids = cur[fx, fy, fz].astype(np.int32)

        # --- ambient occlusion ------------------------------------------
        # Flat indices into the padded volume for the neighbour layer.
        base_flat = (fx + 1 + nx) * _STRIDE_X + (fy + 1 + ny) * _STRIDE_Y + (fz + 1 + nz)
        occ_flat = occ.ravel()
        ao = np.empty((count, 4), dtype=np.float32)
        for c in range(4):
            s1 = occ_flat[base_flat + (face.ao_offsets[c, 0, 0] * _STRIDE_X
                                       + face.ao_offsets[c, 0, 1] * _STRIDE_Y
                                       + face.ao_offsets[c, 0, 2])]
            s2 = occ_flat[base_flat + (face.ao_offsets[c, 1, 0] * _STRIDE_X
                                       + face.ao_offsets[c, 1, 1] * _STRIDE_Y
                                       + face.ao_offsets[c, 1, 2])]
            cr = occ_flat[base_flat + (face.ao_offsets[c, 2, 0] * _STRIDE_X
                                       + face.ao_offsets[c, 2, 1] * _STRIDE_Y
                                       + face.ao_offsets[c, 2, 2])]
            # classic voxel AO: two sides -> fully dark, else 3 - (s1+s2+corner)
            sides = s1.astype(np.int8) + s2.astype(np.int8) + cr.astype(np.int8)
            level = np.where(s1 & s2, 0, 3 - sides)
            ao[:, c] = AO_LEVELS[level]

        # --- sky light: darken as we go deeper below the surface ---------
        # 0.48 keeps caves readable (a real game would need torches here).
        column_top = tops[fx, fz]
        depth = np.maximum(column_top - fy - 3, 0)
        sky = np.clip(1.0 - 0.045 * depth, 0.48, 1.0).astype(np.float32)
        if transparent:
            sky = np.maximum(sky, 0.55)

        # --- texture coordinates ---------------------------------------
        tiles = _TILE_BY_KIND[face.tile_kind][block_ids]
        uv_rect = _TILE_UV[tiles]                       # (count, 4)
        u = uv_rect[:, 0:1] + face.uvs[:, 0][None, :] * (uv_rect[:, 2:3] - uv_rect[:, 0:1])
        v = uv_rect[:, 1:2] + face.uvs[:, 1][None, :] * (uv_rect[:, 3:4] - uv_rect[:, 1:2])
        uv = np.stack([u, v], axis=2)                    # (count, 4, 2)

        # --- positions --------------------------------------------------
        base = np.empty((count, 3), dtype=np.float32)
        base[:, 0] = fx + origin_x
        base[:, 1] = fy
        base[:, 2] = fz + origin_z
        positions = base[:, None, :] + face.corners[None, :, :]   # (count, 4, 3)

        light = (face.brightness * sky[:, None] * ao).astype(np.float32)
        if transparent:
            light *= 0.92   # blended blocks sit slightly back

        quads = np.concatenate(
            [positions, uv, light[:, :, None]], axis=2
        ).astype(np.float32)                             # (count, 4, 6)
        triangles = quads[:, AO_QUAD_INDEX, :].reshape(-1, VERTEX_FLOATS)
        chunk_vertices.append(triangles)

    if not chunk_vertices:
        return np.zeros((0, VERTEX_FLOATS), dtype=np.float32)
    return np.concatenate(chunk_vertices, axis=0)


def mesh_chunk(padded: np.ndarray, cx: int, cz: int) -> dict[str, np.ndarray]:
    """Mesh a padded chunk.

    ``padded`` must be a ``(18, 98, 18)`` block array (see
    :meth:`voxelcraft.world.World.padded_blocks`).

    Returns ``{"opaque": vertices, "transparent": vertices}``.
    """
    blocks = padded[1:-1, 1:-1, 1:-1]
    tops = column_tops(blocks)
    return {
        "opaque": _build_pass(padded, cx, cz, transparent=False, tops=tops),
        "transparent": _build_pass(padded, cx, cz, transparent=True, tops=tops),
    }


def face_count(vertices: np.ndarray) -> int:
    """Number of quads in a vertex array."""
    return 0 if vertices.size == 0 else int(vertices.shape[0] // 6)
