"""A tiny pure-NumPy software rasterizer (no OpenGL required).

It consumes exactly the data the OpenGL renderer consumes - the interleaved
vertex arrays produced by :func:`voxelcraft.mesher.mesh_chunk` and the 2D quads
produced by :mod:`voxelcraft.hud` - and writes a PNG.  This makes it possible to
preview and regression-test the game's visuals on machines without a GPU (CI),
and to generate documentation screenshots.

It is *not* used by the game itself.
"""

from __future__ import annotations

import math

import numpy as np

from voxelcraft import camera as cam
from voxelcraft import font, textures
from voxelcraft.constants import CHUNK_SIZE, WORLD_HEIGHT

SKY_ZENITH = np.array([0.20, 0.45, 0.86], dtype=np.float32)
SKY_HORIZON = np.array([0.68, 0.85, 0.99], dtype=np.float32)
SUN_DIR = np.array([0.35, 0.72, 0.60], dtype=np.float32)
FOG_START = 0.55
FOG_END = 0.95


def _edge(ax, ay, bx, by, px, py):
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def _clip_polygon(poly: np.ndarray) -> np.ndarray:
    """Clip a polygon (N, 7) against the near plane ``z + w >= 0``."""
    out = []
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        da = a[2] + a[3]
        db = b[2] + b[3]
        a_in = da >= 0.0
        b_in = db >= 0.0
        if a_in:
            out.append(a)
        if a_in != b_in:
            t = da / (da - db)
            out.append(a + (b - a) * t)
    return np.array(out) if out else np.zeros((0, 7))


class SoftwareRenderer:
    """Z-buffered triangles + textured screen-space quads."""

    def __init__(self, width: int, height: int, fog_color=None):
        self.width = int(width)
        self.height = int(height)
        self.color = np.zeros((self.height, self.width, 3), dtype=np.float32)
        self.depth = np.full((self.height, self.width), -np.inf, dtype=np.float32)
        self.atlas = textures.build_atlas().astype(np.float32) / 255.0
        self.font_atlas = font.build_atlas().astype(np.float32) / 255.0
        # flattened copies make per-pixel texture lookups a single gather
        self.atlas_flat = self.atlas.reshape(-1, 4)
        self.font_flat = self.font_atlas.reshape(-1, 4)
        self.fog_color = (SKY_HORIZON.copy() if fog_color is None
                          else np.asarray(fog_color, dtype=np.float32))
        self.fog_start = 0.0
        self.fog_end = 1e9
        self.triangles_drawn = 0

    # -- sky -------------------------------------------------------------
    def draw_sky(self, view_proj: np.ndarray, eye, sun_dir=SUN_DIR,
                 zenith=SKY_ZENITH, horizon=SKY_HORIZON, underwater=0.0) -> None:
        """Evaluate the sky shader at half resolution and upscale."""
        w, h = self.width, self.height
        inv = np.linalg.inv(view_proj).astype(np.float32)
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        ndc_x = (xs + 0.5) / w * 2.0 - 1.0
        ndc_y = 1.0 - (ys + 0.5) / h * 2.0

        px = inv[0, 0] * ndc_x + inv[0, 1] * ndc_y + inv[0, 2] + inv[0, 3]
        py = inv[1, 0] * ndc_x + inv[1, 1] * ndc_y + inv[1, 2] + inv[1, 3]
        pz = inv[2, 0] * ndc_x + inv[2, 1] * ndc_y + inv[2, 2] + inv[2, 3]
        pw = inv[3, 0] * ndc_x + inv[3, 1] * ndc_y + inv[3, 2] + inv[3, 3]
        pw = np.where(np.abs(pw) < 1e-6, 1e-6, pw)
        dirx = px / pw - float(eye[0])
        diry = py / pw - float(eye[1])
        dirz = pz / pw - float(eye[2])
        length = np.sqrt(dirx * dirx + diry * diry + dirz * dirz) + 1e-9
        dirx, diry, dirz = dirx / length, diry / length, dirz / length

        t = np.power(np.clip(diry, 0.0, 1.0), 0.75)[:, :, None]
        sky = horizon[None, None, :] * (1.0 - t) + zenith[None, None, :] * t
        below = np.clip(-diry * 7.0, 0.0, 1.0)[:, :, None]
        sky = sky * (1.0 - below) + (horizon * 0.90)[None, None, :] * below

        sun = np.clip(dirx * sun_dir[0] + diry * sun_dir[1] + dirz * sun_dir[2], 0.0, 1.0)
        sky = sky + (np.power(sun, 200.0) * 1.4)[:, :, None] * np.array([1.0, 0.96, 0.84])
        sky = sky + (np.power(sun, 8.0) * 0.10)[:, :, None] * np.array([1.0, 0.92, 0.72])
        sky = sky * (1.0 - underwater) + np.array([0.10, 0.30, 0.48]) * underwater

        self.color[:] = np.clip(sky, 0.0, 1.0)

    # -- triangles -------------------------------------------------------
    def draw_vertices(self, vertices: np.ndarray, view_proj: np.ndarray,
                      transparent: bool = False) -> None:
        """Rasterize an interleaved vertex buffer (x,y,z,u,v,light)."""
        if vertices is None or len(vertices) == 0:
            return
        vp = view_proj.astype(np.float32)
        pos = np.concatenate(
            [vertices[:, 0:3], np.ones((len(vertices), 1), dtype=np.float32)], axis=1
        )
        clip = pos @ vp.T
        poly = np.concatenate([clip, vertices[:, 3:6]], axis=1)

        tris = poly.reshape(-1, 3, 7)
        if transparent:
            order = np.argsort(-tris[:, :, 3].mean(axis=1))
            tris = tris[order]
        for tri in tris:
            for piece in self._clip_triangle(tri):
                self._rasterize(piece, transparent)

    @staticmethod
    def _clip_triangle(tri: np.ndarray):
        if np.all(tri[:, 2] + tri[:, 3] >= 0.0):
            yield tri
            return
        if np.all(tri[:, 2] + tri[:, 3] < 0.0):
            return
        poly = _clip_polygon(tri)
        for i in range(1, len(poly) - 1):
            yield poly[[0, i, i + 1]]

    def _rasterize(self, tri: np.ndarray, transparent: bool) -> None:
        w = tri[:, 3]
        ndc = tri[:, 0:2] / w[:, None]
        sx = (ndc[:, 0] * 0.5 + 0.5) * self.width
        sy = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * self.height

        x0 = max(0, int(math.floor(sx.min())))
        x1 = min(self.width, int(math.ceil(sx.max())) + 1)
        y0 = max(0, int(math.floor(sy.min())))
        y1 = min(self.height, int(math.ceil(sy.max())) + 1)
        if x1 <= x0 or y1 <= y0:
            return

        area = _edge(sx[0], sy[0], sx[1], sy[1], sx[2], sy[2])
        if abs(area) < 1e-9:
            return
        if area < 0:
            sx = sx[[0, 2, 1]]
            sy = sy[[0, 2, 1]]
            tri = tri[[0, 2, 1]]
            w = w[[0, 2, 1]]
            area = -area

        xs = np.arange(x0, x1, dtype=np.float32) + 0.5
        ys = np.arange(y0, y1, dtype=np.float32) + 0.5
        px, py = np.meshgrid(xs, ys)

        e0 = _edge(sx[1], sy[1], sx[2], sy[2], px, py)
        e1 = _edge(sx[2], sy[2], sx[0], sy[0], px, py)
        e2 = _edge(sx[0], sy[0], sx[1], sy[1], px, py)
        inside = (e0 >= 0) & (e1 >= 0) & (e2 >= 0)
        if not inside.any():
            return

        b0 = e0 / area
        b1 = e1 / area
        b2 = e2 / area
        inv_w = b0 / w[0] + b1 / w[1] + b2 / w[2]
        inv_w = np.where(np.abs(inv_w) < 1e-12, 1e-12, inv_w)

        sub = self.depth[y0:y1, x0:x1]
        test = inside & (inv_w > sub)
        if not test.any():
            return

        u = ((b0 * tri[0, 4] / w[0] + b1 * tri[1, 4] / w[1] + b2 * tri[2, 4] / w[2]) / inv_w)
        v = ((b0 * tri[0, 5] / w[0] + b1 * tri[1, 5] / w[1] + b2 * tri[2, 5] / w[2]) / inv_w)
        light = (b0 * tri[0, 6] / w[0] + b1 * tri[1, 6] / w[1] + b2 * tri[2, 6] / w[2]) / inv_w

        atlas_h, atlas_w = self.atlas.shape[0], self.atlas.shape[1]
        np.clip(u * atlas_w, 0, atlas_w - 1, out=u)
        np.clip((1.0 - v) * atlas_h, 0, atlas_h - 1, out=v)
        index = v.astype(np.int32) * atlas_w + u.astype(np.int32)
        texel = self.atlas_flat.reshape(-1)[
            (index * 4)[:, :, None] + np.arange(4, dtype=np.int32)
        ].reshape(index.shape[0], index.shape[1], 4)

        rgb = texel[:, :, :3] * light[:, :, None]
        alpha = texel[:, :, 3]
        valid = test & (alpha >= 0.02)
        if not valid.any():
            return

        distance = 1.0 / inv_w
        fog = np.clip((distance - self.fog_start) / max(self.fog_end - self.fog_start, 1e-6),
                      0.0, 1.0)
        rgb = rgb * (1.0 - fog[:, :, None]) + self.fog_color[None, None, :] * fog[:, :, None]

        target = self.color[y0:y1, x0:x1]
        if transparent:
            a = alpha[:, :, None] * valid[:, :, None]
            blended = target * (1.0 - a) + rgb * a
            target[valid] = blended[valid]
        else:
            target[valid] = rgb[valid]
            sub[valid] = inv_w[valid]
        self.triangles_drawn += 1

    # -- 2D UI -----------------------------------------------------------
    def draw_ui(self, batch, texture: str = "block") -> None:
        """Rasterize a 2D textured triangle batch (HUD) with alpha blending."""
        data = batch.to_array()
        if data.size == 0:
            return
        verts = np.asarray(data, dtype=np.float32).reshape(-1, 8)
        atlas = self.atlas if texture == "block" else self.font_atlas
        for tri in verts.reshape(-1, 3, 8):
            self._rasterize_screen_triangle(tri.astype(np.float32), atlas)

    def _rasterize_screen_triangle(self, tri: np.ndarray, atlas: np.ndarray) -> None:
        xs = tri[:, 0]
        ys = tri[:, 1]
        x0 = max(0, int(math.floor(xs.min())))
        x1 = min(self.width, int(math.ceil(xs.max())) + 1)
        y0 = max(0, int(math.floor(ys.min())))
        y1 = min(self.height, int(math.ceil(ys.max())) + 1)
        if x1 <= x0 or y1 <= y0:
            return
        area = _edge(xs[0], ys[0], xs[1], ys[1], xs[2], ys[2])
        if abs(area) < 1e-9:
            return
        if area < 0:
            tri = tri[[0, 2, 1]]
            xs, ys = tri[:, 0], tri[:, 1]
            area = -area

        px, py = np.meshgrid(
            np.arange(x0, x1, dtype=np.float32) + 0.5,
            np.arange(y0, y1, dtype=np.float32) + 0.5,
        )
        b0 = _edge(xs[1], ys[1], xs[2], ys[2], px, py) / area
        b1 = _edge(xs[2], ys[2], xs[0], ys[0], px, py) / area
        b2 = 1.0 - b0 - b1
        inside = (b0 >= -1e-5) & (b1 >= -1e-5) & (b2 >= -1e-5)
        if not inside.any():
            return

        u = b0 * tri[0, 2] + b1 * tri[1, 2] + b2 * tri[2, 2]
        v = b0 * tri[0, 3] + b1 * tri[1, 3] + b2 * tri[2, 3]
        atlas_h, atlas_w = atlas.shape[0], atlas.shape[1]
        tx = np.clip((u * atlas_w).astype(np.int32), 0, atlas_w - 1)
        ty = np.clip(((1.0 - v) * atlas_h).astype(np.int32), 0, atlas_h - 1)
        texel = atlas[ty, tx]

        r = b0 * tri[0, 4] + b1 * tri[1, 4] + b2 * tri[2, 4]
        g = b0 * tri[0, 5] + b1 * tri[1, 5] + b2 * tri[2, 5]
        b = b0 * tri[0, 6] + b1 * tri[1, 6] + b2 * tri[2, 6]
        a = b0 * tri[0, 7] + b1 * tri[1, 7] + b2 * tri[2, 7]

        rgb = texel[:, :, :3] * np.stack([r, g, b], axis=2)
        alpha = texel[:, :, 3] * a
        valid = inside & (alpha > 0.003)
        if not valid.any():
            return
        target = self.color[y0:y1, x0:x1]
        blended = target * (1.0 - alpha[:, :, None]) + rgb * alpha[:, :, None]
        target[valid] = blended[valid]

    # -- output ----------------------------------------------------------
    def save(self, path: str) -> None:
        from PIL import Image

        img = (np.clip(self.color, 0.0, 1.0) * 255.0).astype(np.uint8)
        Image.fromarray(img, "RGB").save(path)


def render_world(world, player, width=1280, height=720, radius=5,
                 fov=72.0, ui_shapes=None, ui_text=None,
                 underwater=None, fog_start_ratio=FOG_START,
                 fog_end_ratio=FOG_END) -> SoftwareRenderer:
    """Render one frame of ``world`` from ``player``'s point of view."""
    from voxelcraft.mesher import mesh_chunk

    renderer = SoftwareRenderer(width, height)
    aspect = width / max(1, height)
    projection = cam.perspective(fov, aspect, 0.08, 420.0)
    eye = player.eye_position
    view = cam.look_at(eye, eye + player.look_direction())
    view_proj = projection @ view

    underwater_flag = 0.0 if underwater is None else (1.0 if underwater else 0.0)
    renderer.draw_sky(view_proj, eye, underwater=underwater_flag)

    fog_end = radius * CHUNK_SIZE * fog_end_ratio
    fog_start = fog_end * fog_start_ratio
    if underwater_flag:
        fog_end, fog_start = 22.0, 12.0
        renderer.fog_color = np.array([0.10, 0.32, 0.52], dtype=np.float32)
    renderer.fog_end = fog_end
    renderer.fog_start = fog_start

    frustum = cam.Frustum(view_proj)
    pcx = int(math.floor(eye[0])) >> 4
    pcz = int(math.floor(eye[2])) >> 4

    opaque, transparent = [], []
    for (cx, cz), chunk in world.chunks.items():
        if max(abs(cx - pcx), abs(cz - pcz)) > radius:
            continue
        mn = (cx * CHUNK_SIZE, 0, cz * CHUNK_SIZE)
        mx = (cx * CHUNK_SIZE + CHUNK_SIZE, WORLD_HEIGHT, cz * CHUNK_SIZE + CHUNK_SIZE)
        if not frustum.intersects_aabb(mn, mx):
            continue
        geometry = mesh_chunk(world.padded_blocks(chunk), cx, cz)
        if len(geometry["opaque"]):
            opaque.append(geometry["opaque"])
        if len(geometry["transparent"]):
            transparent.append(geometry["transparent"])

    for buf in opaque:
        renderer.draw_vertices(buf, view_proj, transparent=False)
    for buf in transparent:
        renderer.draw_vertices(buf, view_proj, transparent=True)

    if ui_shapes is not None:
        renderer.draw_ui(ui_shapes, "block")
    if ui_text is not None:
        renderer.draw_ui(ui_text, "text")
    return renderer
