"""Vectorised Perlin noise and helpers.

Everything here works on NumPy arrays (scalars are accepted too) so a whole
chunk can be generated with a handful of array operations.  The implementation
is classic Perlin noise (Ken Perlin, 1985/2002) with a seeded permutation
table; it is fully deterministic for a given seed.

``noise2`` / ``noise3`` return values in roughly ``[-1, 1]``, ``fbm2`` / ``fbm3``
return fractal Brownian motion normalised to the same range and ``ridged2``
returns ``[0, 1]`` mountain ridges.
"""

from __future__ import annotations

import numpy as np

# 8 unit-ish gradients used by the 2D variant and 12 for the 3D variant.
_G2 = np.array(
    [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, 1], [1, -1], [-1, -1]],
    dtype=np.float64,
)
_G2X, _G2Y = _G2[:, 0].copy(), _G2[:, 1].copy()

_G3 = np.array(
    [
        [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
        [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
        [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
    ],
    dtype=np.float64,
)
_G3X, _G3Y, _G3Z = _G3[:, 0].copy(), _G3[:, 1].copy(), _G3[:, 2].copy()


def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    return a + t * (b - a)


def smoothstep(edge0: float, edge1: float, x):
    """Hermite interpolation between two edges, clamped to ``[0, 1]``."""
    x = np.asarray(x, dtype=np.float64)
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class PerlinNoise:
    """Seeded Perlin noise generator (2D and 3D) with fractal helpers."""

    def __init__(self, seed: int = 0):
        seed = int(seed) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        perm = rng.permutation(256).astype(np.int32)
        # Doubled table removes the need for modulo in the hot loops.
        self.perm = np.concatenate([perm, perm]).astype(np.int32)
        self.seed = seed
        # A second, differently seeded permutation decorrelates the 3D lattice
        # from the 2D one (otherwise caves would line up with hills).
        rng2 = np.random.default_rng(seed ^ 0x9E3779B9)
        perm2 = rng2.permutation(256).astype(np.int32)
        self.perm3 = np.concatenate([perm2, perm2]).astype(np.int32)

    # -- raw noise --------------------------------------------------------
    def noise2(self, x, y) -> np.ndarray:
        """2D Perlin noise, approximately ``[-1, 1]``."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        xi = np.floor(x)
        yi = np.floor(y)
        xf = x - xi
        yf = y - yi
        xi = xi.astype(np.int32) & 255
        yi = yi.astype(np.int32) & 255
        u = _fade(xf)
        v = _fade(yf)

        p = self.perm
        a = p[xi] + yi
        b = p[xi + 1] + yi
        h00 = p[a]
        h10 = p[b]
        h01 = p[a + 1]
        h11 = p[b + 1]

        n00 = _G2X[h00 & 7] * xf + _G2Y[h00 & 7] * yf
        n10 = _G2X[h10 & 7] * (xf - 1.0) + _G2Y[h10 & 7] * yf
        n01 = _G2X[h01 & 7] * xf + _G2Y[h01 & 7] * (yf - 1.0)
        n11 = _G2X[h11 & 7] * (xf - 1.0) + _G2Y[h11 & 7] * (yf - 1.0)

        nx0 = _lerp(n00, n10, u)
        nx1 = _lerp(n01, n11, u)
        return _lerp(nx0, nx1, v) * 0.78

    def noise3(self, x, y, z) -> np.ndarray:
        """3D Perlin noise, approximately ``[-1, 1]``."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        xi = np.floor(x)
        yi = np.floor(y)
        zi = np.floor(z)
        xf = x - xi
        yf = y - yi
        zf = z - zi
        xi = xi.astype(np.int32) & 255
        yi = yi.astype(np.int32) & 255
        zi = zi.astype(np.int32) & 255
        u, v, w = _fade(xf), _fade(yf), _fade(zf)

        p = self.perm3
        a = p[xi] + yi
        aa = p[a] + zi
        ab = p[a + 1] + zi
        b = p[xi + 1] + yi
        ba = p[b] + zi
        bb = p[b + 1] + zi

        h000, h100 = p[aa], p[ba]
        h010, h110 = p[ab], p[bb]
        h001, h101 = p[aa + 1], p[ba + 1]
        h011, h111 = p[ab + 1], p[bb + 1]

        def dot(h, dx, dy, dz):
            i = h % 12
            return _G3X[i] * dx + _G3Y[i] * dy + _G3Z[i] * dz

        n000 = dot(h000, xf, yf, zf)
        n100 = dot(h100, xf - 1.0, yf, zf)
        n010 = dot(h010, xf, yf - 1.0, zf)
        n110 = dot(h110, xf - 1.0, yf - 1.0, zf)
        n001 = dot(h001, xf, yf, zf - 1.0)
        n101 = dot(h101, xf - 1.0, yf, zf - 1.0)
        n011 = dot(h011, xf, yf - 1.0, zf - 1.0)
        n111 = dot(h111, xf - 1.0, yf - 1.0, zf - 1.0)

        nx00 = _lerp(n000, n100, u)
        nx10 = _lerp(n010, n110, u)
        nx01 = _lerp(n001, n101, u)
        nx11 = _lerp(n011, n111, u)
        nxy0 = _lerp(nx00, nx10, v)
        nxy1 = _lerp(nx01, nx11, v)
        return _lerp(nxy0, nxy1, w) * 0.82

    # -- fractal helpers --------------------------------------------------
    def fbm2(self, x, y, octaves: int = 4, lacunarity: float = 2.0,
             gain: float = 0.5) -> np.ndarray:
        """Fractal Brownian motion in 2D, normalised to about ``[-1, 1]``."""
        total = np.zeros(np.broadcast(np.asarray(x), np.asarray(y)).shape)
        amp = 1.0
        freq = 1.0
        norm = 0.0
        for _ in range(octaves):
            total = total + self.noise2(np.asarray(x) * freq, np.asarray(y) * freq) * amp
            norm += amp
            amp *= gain
            freq *= lacunarity
        return total / norm

    def fbm3(self, x, y, z, octaves: int = 3, lacunarity: float = 2.0,
             gain: float = 0.5) -> np.ndarray:
        """Fractal Brownian motion in 3D (used for caves)."""
        shape = np.broadcast(np.asarray(x), np.asarray(y), np.asarray(z)).shape
        total = np.zeros(shape)
        amp = 1.0
        freq = 1.0
        norm = 0.0
        for _ in range(octaves):
            total = total + self.noise3(np.asarray(x) * freq, np.asarray(y) * freq,
                                        np.asarray(z) * freq) * amp
            norm += amp
            amp *= gain
            freq *= lacunarity
        return total / norm

    def ridged2(self, x, y, octaves: int = 4, lacunarity: float = 2.0,
                gain: float = 0.5, sharpness: float = 2.2) -> np.ndarray:
        """Ridged multifractal noise in ``[0, 1]`` - gives sharp mountain crests.

        Each octave contributes ``1 - sharpness * |noise|`` clamped to ``[0, 1]``,
        so crests form along the zero crossings of the Perlin field.
        """
        total = np.zeros(np.broadcast(np.asarray(x), np.asarray(y)).shape)
        amp = 1.0
        freq = 1.0
        norm = 0.0
        for _ in range(octaves):
            n = self.noise2(np.asarray(x) * freq, np.asarray(y) * freq)
            total = total + np.clip(1.0 - sharpness * np.abs(n), 0.0, 1.0) * amp
            norm += amp
            amp *= gain
            freq *= lacunarity
        return np.clip(total / norm, 0.0, 1.0)


# --- deterministic hashing (tree placement etc.) --------------------------

_M64 = np.uint64(0xFFFFFFFFFFFFFFFF)
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)


def hash01(*values, seed: int = 0) -> np.ndarray:
    """Deterministic pseudo random floats in ``[0, 1)`` from integer inputs.

    Unlike :class:`PerlinNoise` this is stateless and cheap, which makes it a
    good fit for "should there be a tree at this column" style decisions that
    must agree between neighbouring chunks.
    """
    with np.errstate(over="ignore"):
        h = np.uint64(int(seed) & 0xFFFFFFFF) * _GOLDEN + np.uint64(0x1234567)
        for v in values:
            arr = np.asarray(v, dtype=np.int64).astype(np.uint64)
            h = h ^ (arr * _GOLDEN)
            h = h ^ (h >> np.uint64(30))
            h = h * np.uint64(0xBF58476D1CE4E5B9)
            h = h ^ (h >> np.uint64(27))
            h = h * np.uint64(0x94D049BB133111EB)
            h = h ^ (h >> np.uint64(31))
    return (h & _M64).astype(np.float64) / float(2 ** 64)
