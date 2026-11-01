"""OpenGL 3.3 renderer.

The renderer owns every GL object: two texture atlases, the shader programs,
per-chunk vertex buffers and the dynamic buffers used by the HUD.  It only
draws - world generation, physics and input live elsewhere.

Render order
------------
1. procedural sky (gradient + sun)
2. opaque chunk geometry (back face culling, depth write)
3. block selection outline
4. water/glass (alpha blending, no depth write, sorted far -> near)
5. 2D HUD (screen space)
"""

from __future__ import annotations

import ctypes
import math
from typing import Optional

import numpy as np
from pyglet import gl as GL

from . import camera as cam
from . import font
from . import textures
from .constants import (
    CHUNK_SIZE,
    DEFAULT_RENDER_DISTANCE,
    FOG_END_RATIO,
    FOG_START_RATIO,
    WORLD_HEIGHT,
)
from .mesher import VERTEX_FLOATS, mesh_chunk

FOV_Y = 72.0
NEAR_PLANE = 0.08
FAR_PLANE = 420.0

SKY_ZENITH = (0.20, 0.45, 0.86)
SKY_HORIZON = (0.68, 0.85, 0.99)
SUN_DIR = (0.35, 0.72, 0.60)

# ---------------------------------------------------------------------------
# shaders
# ---------------------------------------------------------------------------
CHUNK_VERT = """#version 330 core
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec2 a_uv;
layout(location = 2) in float a_light;

uniform mat4 u_viewproj;

out vec2 v_uv;
out float v_light;
out vec3 v_world;

void main() {
    v_uv = a_uv;
    v_light = a_light;
    v_world = a_pos;
    gl_Position = u_viewproj * vec4(a_pos, 1.0);
}
"""

CHUNK_FRAG = """#version 330 core
in vec2 v_uv;
in float v_light;
in vec3 v_world;

uniform sampler2D u_atlas;
uniform vec3 u_camera;
uniform vec3 u_fog_color;
uniform float u_fog_start;
uniform float u_fog_end;

out vec4 o_color;

void main() {
    vec4 texel = texture(u_atlas, v_uv);
    if (texel.a < 0.02) {
        discard;
    }
    vec3 color = texel.rgb * v_light;
    float dist = length(v_world - u_camera);
    float fog = clamp((dist - u_fog_start) / max(u_fog_end - u_fog_start, 0.001), 0.0, 1.0);
    color = mix(color, u_fog_color, fog);
    o_color = vec4(color, texel.a);
}
"""

SKY_VERT = """#version 330 core
layout(location = 0) in vec2 a_ndc;
out vec2 v_ndc;
void main() {
    v_ndc = a_ndc;
    gl_Position = vec4(a_ndc, 1.0, 1.0);
}
"""

SKY_FRAG = """#version 330 core
in vec2 v_ndc;

uniform mat4 u_inv_viewproj;
uniform vec3 u_camera;
uniform vec3 u_sun_dir;
uniform vec3 u_zenith;
uniform vec3 u_horizon;
uniform float u_underwater;

out vec4 o_color;

void main() {
    vec4 p = u_inv_viewproj * vec4(v_ndc, 1.0, 1.0);
    vec3 dir = normalize(p.xyz / p.w - u_camera);
    float h = clamp(dir.y, -1.0, 1.0);
    float t = pow(clamp(h, 0.0, 1.0), 0.75);
    vec3 col = mix(u_horizon, u_zenith, t);
    col = mix(col, u_horizon * 0.90, clamp(-h * 7.0, 0.0, 1.0));

    float sun = max(dot(dir, normalize(u_sun_dir)), 0.0);
    col += vec3(1.0, 0.96, 0.84) * pow(sun, 420.0) * 1.4;
    col += vec3(1.0, 0.92, 0.72) * pow(sun, 10.0) * 0.10;

    // underwater: everything turns into warm blue haze
    col = mix(col, vec3(0.10, 0.30, 0.48), u_underwater);
    o_color = vec4(col, 1.0);
}
"""

UI_VERT = """#version 330 core
layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec2 a_uv;
layout(location = 2) in vec4 a_color;

uniform vec2 u_screen;

out vec2 v_uv;
out vec4 v_color;

void main() {
    vec2 ndc = vec2(a_pos.x / u_screen.x * 2.0 - 1.0,
                    1.0 - a_pos.y / u_screen.y * 2.0);
    v_uv = a_uv;
    v_color = a_color;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
"""

UI_FRAG = """#version 330 core
in vec2 v_uv;
in vec4 v_color;

uniform sampler2D u_texture;

out vec4 o_color;

void main() {
    vec4 texel = texture(u_texture, v_uv);
    o_color = vec4(v_color.rgb * texel.rgb, v_color.a * texel.a);
}
"""

LINE_VERT = """#version 330 core
layout(location = 0) in vec3 a_pos;
uniform mat4 u_viewproj;
void main() {
    gl_Position = u_viewproj * vec4(a_pos, 1.0);
}
"""

LINE_FRAG = """#version 330 core
uniform vec4 u_color;
out vec4 o_color;
void main() {
    o_color = u_color;
}
"""

UI_FLOATS = 8  # x, y, u, v, r, g, b, a


# ---------------------------------------------------------------------------
# small GL helpers
# ---------------------------------------------------------------------------
def gen_buffer() -> int:
    buf = GL.GLuint()
    GL.glGenBuffers(1, ctypes.byref(buf))
    return int(buf.value)


def gen_vertex_array() -> int:
    vao = GL.GLuint()
    GL.glGenVertexArrays(1, ctypes.byref(vao))
    return int(vao.value)


def gen_texture() -> int:
    tex = GL.GLuint()
    GL.glGenTextures(1, ctypes.byref(tex))
    return int(tex.value)


def delete_buffer(buffer_id: int) -> None:
    if buffer_id:
        buf = GL.GLuint(buffer_id)
        GL.glDeleteBuffers(1, ctypes.byref(buf))


def delete_vertex_array(vao: int) -> None:
    if vao:
        obj = GL.GLuint(vao)
        GL.glDeleteVertexArrays(1, ctypes.byref(obj))


def delete_texture(texture_id: int) -> None:
    if texture_id:
        tex = GL.GLuint(texture_id)
        GL.glDeleteTextures(1, ctypes.byref(tex))


def upload_array(target: int, data: np.ndarray, usage: int) -> None:
    """Upload a NumPy array to the currently bound buffer."""
    if data is None or data.size == 0:
        GL.glBufferData(target, 0, None, usage)
        return
    GL.glBufferData(target, int(data.nbytes),
                    ctypes.cast(data.ctypes.data, ctypes.c_void_p), usage)


def _compile_shader(kind: int, source: str, label: str) -> int:
    shader = GL.glCreateShader(kind)
    src = source.encode("utf-8")
    buf = ctypes.c_char_p(src)
    GL.glShaderSource(shader, 1, ctypes.byref(buf), None)
    GL.glCompileShader(shader)
    status = GL.GLint(0)
    GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS, ctypes.byref(status))
    if status.value == 0:
        log = GL.glGetShaderInfoLog(shader)
        raise RuntimeError(
            f"Failed to compile {label} shader:\n"
            + (log.decode("utf-8", "replace") if isinstance(log, bytes) else str(log))
            + f"\n--- source ---\n{source}"
        )
    return int(shader)


class Program:
    """A linked shader program with cached uniform lookups."""

    def __init__(self, vertex_src: str, fragment_src: str, label: str):
        vs = _compile_shader(GL.GL_VERTEX_SHADER, vertex_src, label + " vertex")
        fs = _compile_shader(GL.GL_FRAGMENT_SHADER, fragment_src, label + " fragment")
        self.id = int(GL.glCreateProgram())
        GL.glAttachShader(self.id, vs)
        GL.glAttachShader(self.id, fs)
        GL.glLinkProgram(self.id)
        status = GL.GLint(0)
        GL.glGetProgramiv(self.id, GL.GL_LINK_STATUS, ctypes.byref(status))
        if status.value == 0:
            log = GL.glGetProgramInfoLog(self.id)
            raise RuntimeError(f"Failed to link {label}: {log}")
        GL.glDeleteShader(vs)
        GL.glDeleteShader(fs)
        self._uniforms: dict[str, int] = {}
        self.label = label

    def use(self) -> None:
        GL.glUseProgram(self.id)

    def location(self, name: str) -> int:
        if name not in self._uniforms:
            self._uniforms[name] = int(GL.glGetUniformLocation(self.id, name.encode()))
        return self._uniforms[name]

    def set_mat4(self, name: str, matrix: np.ndarray) -> None:
        loc = self.location(name)
        if loc >= 0:
            data = cam.to_float32(matrix)
            GL.glUniformMatrix4fv(loc, 1, GL.GL_TRUE,
                                  data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

    def set_vec3(self, name: str, value) -> None:
        loc = self.location(name)
        if loc >= 0:
            GL.glUniform3f(loc, float(value[0]), float(value[1]), float(value[2]))

    def set_vec2(self, name: str, value) -> None:
        loc = self.location(name)
        if loc >= 0:
            GL.glUniform2f(loc, float(value[0]), float(value[1]))

    def set_float(self, name: str, value: float) -> None:
        loc = self.location(name)
        if loc >= 0:
            GL.glUniform1f(loc, float(value))

    def set_int(self, name: str, value: int) -> None:
        loc = self.location(name)
        if loc >= 0:
            GL.glUniform1i(loc, int(value))

    def set_vec4(self, name: str, value) -> None:
        loc = self.location(name)
        if loc >= 0:
            GL.glUniform4f(loc, float(value[0]), float(value[1]),
                           float(value[2]), float(value[3]))


class ChunkMesh:
    """GPU buffers for one chunk (one VAO per render pass)."""

    __slots__ = ("vao", "vbo", "count", "signature")

    def __init__(self, vertices: np.ndarray):
        self.vao = gen_vertex_array()
        self.vbo = gen_buffer()
        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        upload_array(GL.GL_ARRAY_BUFFER, vertices, GL.GL_STATIC_DRAW)
        stride = VERTEX_FLOATS * 4
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, stride,
                                 ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, stride,
                                 ctypes.c_void_p(12))
        GL.glEnableVertexAttribArray(2)
        GL.glVertexAttribPointer(2, 1, GL.GL_FLOAT, GL.GL_FALSE, stride,
                                 ctypes.c_void_p(20))
        GL.glBindVertexArray(0)
        self.count = int(vertices.shape[0]) if vertices.size else 0
        self.signature: tuple = ()

    def update(self, vertices: np.ndarray) -> None:
        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        upload_array(GL.GL_ARRAY_BUFFER, vertices, GL.GL_STATIC_DRAW)
        GL.glBindVertexArray(0)
        self.count = int(vertices.shape[0]) if vertices.size else 0

    def draw(self) -> None:
        if self.count:
            GL.glBindVertexArray(self.vao)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.count)

    def release(self) -> None:
        delete_vertex_array(self.vao)
        delete_buffer(self.vbo)
        self.vao = self.vbo = 0
        self.count = 0


class Renderer:
    """Draws the world and the HUD."""

    def __init__(self, window, world, render_distance: int = DEFAULT_RENDER_DISTANCE):
        self.window = window
        self.world = world
        self.render_distance = int(render_distance)
        self.mesh_budget = 3            # chunk remeshes per frame
        self.width, self.height = window.size
        self.fov = FOV_Y
        self.fog_start_ratio = FOG_START_RATIO
        self.stats = {
            "chunks_drawn": 0,
            "triangles": 0,
            "visible_chunks": 0,
        }
        # (cx, cz) -> {"opaque": ChunkMesh, "transparent": ChunkMesh, "version": tuple}
        self.meshes: dict[tuple[int, int], dict] = {}
        self._init_gl()

    # -- setup -----------------------------------------------------------
    def _init_gl(self) -> None:
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glCullFace(GL.GL_BACK)
        GL.glFrontFace(GL.GL_CCW)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glClearColor(*SKY_HORIZON, 1.0)

        self.block_atlas = self._make_atlas_texture(textures.build_atlas(), mipmap=True)
        self.font_atlas = self._make_atlas_texture(font.build_atlas(), mipmap=False)

        self.chunk_program = Program(CHUNK_VERT, CHUNK_FRAG, "chunk")
        self.sky_program = Program(SKY_VERT, SKY_FRAG, "sky")
        self.ui_program = Program(UI_VERT, UI_FRAG, "ui")
        self.line_program = Program(LINE_VERT, LINE_FRAG, "line")

        self._init_sky_geometry()
        self._init_ui_geometry()
        self._init_highlight_geometry()

    @staticmethod
    def _make_atlas_texture(data: np.ndarray, mipmap: bool) -> int:
        texture = gen_texture()
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        if mipmap:
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER,
                               GL.GL_LINEAR_MIPMAP_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        else:
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        payload = np.ascontiguousarray(data, dtype=np.uint8)
        height, width = payload.shape[0], payload.shape[1]
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, width, height, 0,
                        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE,
                        ctypes.cast(payload.ctypes.data, ctypes.c_void_p))
        if mipmap:
            GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        return texture

    def _init_sky_geometry(self) -> None:
        # One big triangle covering the whole screen.
        verts = np.array([[-1, -1], [3, -1], [-1, 3]], dtype=np.float32)
        self.sky_vao = gen_vertex_array()
        self.sky_vbo = gen_buffer()
        GL.glBindVertexArray(self.sky_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.sky_vbo)
        upload_array(GL.GL_ARRAY_BUFFER, verts, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 8, ctypes.c_void_p(0))
        GL.glBindVertexArray(0)

    def _init_ui_geometry(self) -> None:
        self.ui_vao = gen_vertex_array()
        self.ui_vbo = gen_buffer()
        GL.glBindVertexArray(self.ui_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.ui_vbo)
        stride = UI_FLOATS * 4
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(8))
        GL.glEnableVertexAttribArray(2)
        GL.glVertexAttribPointer(2, 4, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(16))
        GL.glBindVertexArray(0)
        self._ui_capacity = 0
        self._ui_count = 0

    def _init_highlight_geometry(self) -> None:
        e = 0.002
        lo, hi = -e, 1.0 + e
        c = [
            (lo, lo, lo), (hi, lo, lo), (hi, lo, hi), (lo, lo, hi),
            (lo, hi, lo), (hi, hi, lo), (hi, hi, hi), (lo, hi, hi),
        ]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        verts = np.array([c[i] for edge in edges for i in edge], dtype=np.float32)
        self.highlight_vao = gen_vertex_array()
        self.highlight_vbo = gen_buffer()
        GL.glBindVertexArray(self.highlight_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.highlight_vbo)
        upload_array(GL.GL_ARRAY_BUFFER, verts, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 12, ctypes.c_void_p(0))
        GL.glBindVertexArray(0)
        self.highlight_count = verts.shape[0]

    def on_resize(self, width: int, height: int) -> None:
        self.width, self.height = max(1, width), max(1, height)
        GL.glViewport(0, 0, self.width, self.height)

    # -- chunk meshing ---------------------------------------------------
    def _signature(self, cx: int, cz: int) -> tuple:
        """Versions of a chunk and all eight neighbours (for correct borders)."""
        chunks = self.world.chunks
        sig = []
        for dz in (-1, 0, 1):
            for dx in (-1, 0, 1):
                chunk = chunks.get((cx + dx, cz + dz))
                sig.append(-1 if chunk is None else chunk.version)
        return tuple(sig)

    def update_chunks(self, player, budget: Optional[int] = None,
                      force: bool = False) -> int:
        """Rebuild up to ``budget`` stale chunk meshes, closest first."""
        budget = self.mesh_budget if budget is None else budget
        eye = player.eye_position
        cx0 = int(math.floor(eye[0])) >> 4
        cz0 = int(math.floor(eye[2])) >> 4
        radius = self.render_distance

        # Forget meshes for chunks that are gone or too far away.
        for (cx, cz) in list(self.meshes):
            chunk = self.world.chunks.get((cx, cz))
            far = max(abs(cx - cx0), abs(cz - cz0)) > radius
            if chunk is None or far:
                self.meshes[(cx, cz)]["opaque"].release()
                self.meshes[(cx, cz)]["transparent"].release()
                del self.meshes[(cx, cz)]

        stale = []
        for (cx, cz), chunk in self.world.chunks.items():
            if max(abs(cx - cx0), abs(cz - cz0)) > radius:
                continue
            signature = self._signature(cx, cz)
            entry = self.meshes.get((cx, cz))
            if entry is None or entry["signature"] != signature:
                dx = (cx * CHUNK_SIZE + 8) - eye[0]
                dz = (cz * CHUNK_SIZE + 8) - eye[2]
                stale.append((dx * dx + dz * dz, cx, cz, signature))
        stale.sort(key=lambda item: item[0])

        if force:
            budget = len(stale)
        built = 0
        for _, cx, cz, signature in stale:
            if built >= budget:
                break
            self._rebuild(cx, cz, signature)
            built += 1
        return built

    def _rebuild(self, cx: int, cz: int, signature: tuple) -> None:
        chunk = self.world.chunks.get((cx, cz))
        if chunk is None:
            return
        padded = self.world.padded_blocks(chunk)
        geometry = mesh_chunk(padded, cx, cz)
        entry = self.meshes.get((cx, cz))
        if entry is None:
            entry = {
                "opaque": ChunkMesh(geometry["opaque"]),
                "transparent": ChunkMesh(geometry["transparent"]),
            }
            self.meshes[(cx, cz)] = entry
        else:
            entry["opaque"].update(geometry["opaque"])
            entry["transparent"].update(geometry["transparent"])
        entry["signature"] = signature

    # -- frame -----------------------------------------------------------
    def _camera_matrices(self, player):
        aspect = self.width / max(1, self.height)
        projection = cam.perspective(self.fov, aspect, NEAR_PLANE, FAR_PLANE)
        eye = player.eye_position
        view = cam.look_at(eye, eye + player.look_direction())
        return projection, view, projection @ view, eye

    def draw(self, player, ui_shapes=None, ui_text=None,
             selected_block: Optional[tuple] = None) -> None:
        """Draw one frame. ``ui_shapes`` / ``ui_text`` are :class:`UIBatch`es."""
        GL.glViewport(0, 0, self.width, self.height)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        projection, view, view_proj, eye = self._camera_matrices(player)
        frustum = cam.Frustum(view_proj)
        underwater = bool(player.head_in_water)
        fog_color = (0.10, 0.32, 0.52) if underwater else SKY_HORIZON
        fog_end = max(18.0, self.render_distance * CHUNK_SIZE * FOG_END_RATIO)
        if underwater:
            fog_end = 26.0
        fog_start = fog_end * self.fog_start_ratio

        self._draw_sky(view_proj, eye, underwater)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)

        # --- opaque geometry -------------------------------------------
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glDisable(GL.GL_BLEND)
        program = self.chunk_program
        program.use()
        program.set_mat4("u_viewproj", view_proj)
        program.set_vec3("u_camera", eye)
        program.set_vec3("u_fog_color", fog_color)
        program.set_float("u_fog_start", fog_start)
        program.set_float("u_fog_end", fog_end)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.block_atlas)
        program.set_int("u_atlas", 0)

        visible = []
        for (cx, cz), entry in self.meshes.items():
            min_corner = (cx * CHUNK_SIZE, 0, cz * CHUNK_SIZE)
            max_corner = (cx * CHUNK_SIZE + CHUNK_SIZE, WORLD_HEIGHT,
                          cz * CHUNK_SIZE + CHUNK_SIZE)
            if not frustum.intersects_aabb(min_corner, max_corner):
                continue
            dx = min_corner[0] + 8 - eye[0]
            dz = min_corner[2] + 8 - eye[2]
            visible.append((dx * dx + dz * dz, entry))

        triangles = 0
        chunks_drawn = 0
        for _, entry in visible:
            mesh = entry["opaque"]
            if mesh.count:
                mesh.draw()
                triangles += mesh.count // 3
                chunks_drawn += 1

        self.stats.update(chunks_drawn=chunks_drawn, triangles=triangles,
                          visible_chunks=len(visible))

        # --- block selection -------------------------------------------
        if selected_block is not None:
            self._draw_highlight(selected_block, view_proj)

        # --- transparent geometry --------------------------------------
        GL.glEnable(GL.GL_BLEND)
        GL.glDepthMask(GL.GL_FALSE)
        GL.glDisable(GL.GL_CULL_FACE)
        program.use()
        visible.sort(key=lambda item: item[0], reverse=True)   # far to near
        for _, entry in visible:
            mesh = entry["transparent"]
            if mesh.count:
                mesh.draw()
        GL.glDepthMask(GL.GL_TRUE)
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glDisable(GL.GL_BLEND)

        # --- HUD --------------------------------------------------------
        if ui_shapes is not None:
            self._draw_ui(ui_shapes, self.block_atlas)
        if ui_text is not None:
            self._draw_ui(ui_text, self.font_atlas)

    def draw_plain(self, ui_shapes=None, ui_text=None,
                   clear_color=(0.06, 0.09, 0.13)) -> None:
        """Clear to a flat colour and draw only the HUD (loading / error screens)."""
        GL.glViewport(0, 0, self.width, self.height)
        GL.glClearColor(*clear_color, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if ui_shapes is not None:
            self._draw_ui(ui_shapes, self.block_atlas)
        if ui_text is not None:
            self._draw_ui(ui_text, self.font_atlas)
        GL.glClearColor(*SKY_HORIZON, 1.0)

    def _draw_sky(self, view_proj: np.ndarray, eye, underwater: bool) -> None:
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(GL.GL_FALSE)
        GL.glDisable(GL.GL_CULL_FACE)
        program = self.sky_program
        program.use()
        inv = np.linalg.inv(view_proj)
        program.set_mat4("u_inv_viewproj", inv)
        program.set_vec3("u_camera", eye)
        program.set_vec3("u_sun_dir", SUN_DIR)
        program.set_vec3("u_zenith", SKY_ZENITH)
        program.set_vec3("u_horizon", SKY_HORIZON)
        program.set_float("u_underwater", 1.0 if underwater else 0.0)
        GL.glBindVertexArray(self.sky_vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        GL.glBindVertexArray(0)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_CULL_FACE)

    def _draw_highlight(self, block, view_proj: np.ndarray) -> None:
        GL.glEnable(GL.GL_BLEND)
        GL.glDepthMask(GL.GL_FALSE)
        GL.glLineWidth(2.0)
        program = self.line_program
        program.use()
        model = cam.translation(float(block[0]), float(block[1]), float(block[2]))
        program.set_mat4("u_viewproj", view_proj @ model)
        program.set_vec4("u_color", (0.03, 0.03, 0.03, 0.55))
        GL.glBindVertexArray(self.highlight_vao)
        GL.glDrawArrays(GL.GL_LINES, 0, self.highlight_count)
        GL.glBindVertexArray(0)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDisable(GL.GL_BLEND)

    def _draw_ui(self, batch, texture: int) -> None:
        data = batch.to_array()
        if data.size == 0:
            return
        count = data.size // UI_FLOATS
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glEnable(GL.GL_BLEND)
        program = self.ui_program
        program.use()
        program.set_vec2("u_screen", (self.width, self.height))
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
        program.set_int("u_texture", 0)
        GL.glBindVertexArray(self.ui_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.ui_vbo)
        if data.nbytes > self._ui_capacity:
            upload_array(GL.GL_ARRAY_BUFFER, data, GL.GL_DYNAMIC_DRAW)
            self._ui_capacity = data.nbytes
        else:
            GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, data.nbytes,
                               ctypes.cast(data.ctypes.data, ctypes.c_void_p))
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, count)
        GL.glBindVertexArray(0)
        GL.glDisable(GL.GL_BLEND)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_CULL_FACE)

    # -- shutdown --------------------------------------------------------
    def release(self) -> None:
        for entry in self.meshes.values():
            entry["opaque"].release()
            entry["transparent"].release()
        self.meshes.clear()
        delete_vertex_array(self.sky_vao)
        delete_buffer(self.sky_vbo)
        delete_vertex_array(self.ui_vao)
        delete_buffer(self.ui_vbo)
        delete_vertex_array(self.highlight_vao)
        delete_buffer(self.highlight_vbo)
        delete_texture(self.block_atlas)
        delete_texture(self.font_atlas)
        for program in (self.chunk_program, self.sky_program, self.ui_program,
                        self.line_program):
            GL.glDeleteProgram(program.id)
