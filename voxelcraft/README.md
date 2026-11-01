# ⛏️ Voxelcraft

A small **Minecraft-like voxel sandbox** written from scratch in Python with
**Pyglet** (window + OpenGL) and **NumPy** (terrain, meshing, physics helpers).

Procedurally generated Perlin-noise terrain, a first-person controller that can
walk, sprint, sneak, jump, swim and fly, and full block editing: **left click to
destroy**, **right click to place**. Everything you see is generated at runtime —
the project ships **no binary assets**: block textures, the font and the sky are
all drawn with code.

![Vista](docs/shots/vista.png)

| | |
|---|---|
| ![Beach](docs/shots/beach.png) | ![Forest](docs/shots/forest.png) |
| ![House](docs/shots/house.png) | ![Cave](docs/shots/cave.png) |
| ![Underwater](docs/shots/underwater.png) | *water, glass and HUD transparencies* |

> The pictures above were rendered by `tools/render_preview.py`, a pure-NumPy
> software rasterizer that consumes exactly the vertex data the OpenGL renderer
> consumes — useful for CI and for documenting the game on machines without a GPU.

---

## Quick start

```bash
# 1. install the two dependencies
python -m pip install -r voxelcraft/requirements.txt      # pyglet, numpy

# 2. play
python -m voxelcraft
# or
python voxelcraft/run.py
```

The first frame shows a *generating terrain* progress bar while the chunks
around the spawn point are built on background threads, then a short title
screen — click (or press Enter) and you are in the world.

### Requirements

* Python **3.9+** (tested on 3.11)
* Pyglet **2.x** and NumPy
* A GPU/driver with **OpenGL 3.3** (any hardware from the last decade).
  On a headless Linux box: `xvfb-run -s "-screen 0 1280x720x24" python -m voxelcraft`
  with Mesa's software driver (`libgl1-mesa-dri`).

### Command line

```bash
python -m voxelcraft --seed 12345 --render-distance 9 --width 1600 --height 900
python -m voxelcraft --load saves/world_12345.npz     # continue a world
python -m voxelcraft --list-blocks                    # dump the block registry
python -m voxelcraft --help
```

| option | meaning |
| --- | --- |
| `--seed N` | world seed (every seed is a different world) |
| `--render-distance N` | view distance in chunks, 2–14 (default 7) |
| `--width/--height` | window size (default 1280×720) |
| `--fullscreen` | start fullscreen |
| `--no-msaa` | disable 4× MSAA for weak GPUs |
| `--load FILE` / `--save FILE` | load / store block edits |

Your edits are saved automatically on exit, on pause and every 60 s to
`saves/world_<seed>.npz` (seed + the sparse list of changed blocks, so files
stay tiny).

---

## Controls

| input | action |
| --- | --- |
| **W A S D** | walk |
| **Space** | jump · swim up · fly up |
| **Shift** | sneak · fly down |
| **Ctrl** | sprint (slight FOV kick) |
| **F** or double-tap **Space** | toggle fly mode |
| **Mouse** | look around |
| **Left click** | destroy the targeted block (hold to keep digging) |
| **Right click** | place the selected block (hold to keep placing) |
| **Middle click** | pick the targeted block |
| **1 – 9**, mouse wheel | select hotbar slot |
| **F3** | debug overlay (FPS, position, chunks, triangles, target block) |
| **F1** | hide the HUD |
| **- / =** | decrease / increase render distance |
| **R** | respawn on the surface |
| **Esc** | pause (saves the world) · **Q** while paused quits |

---

## What is implemented

**World**
* Layered **Perlin noise** terrain: continents, rolling hills, micro-relief,
  ridged-noise mountain peaks, oceans, lakes, beaches and snowy summits.
* 3D noise **caves** (two intersected fBm fields) plus large deep caverns.
* Tree placement from a deterministic hash on a 6-block grid — trees that
  straddle a chunk border are grown identically in both chunks (no seams).
* 16×128×16 chunk columns streamed in/out around the player, generated on
  worker threads so the frame rate stays smooth.
* 14 block types with per-face textures, including non-solid water and glass.

**Player**
* First person camera with pitch clamping and a fixed-timestep simulation.
* AABB-vs-voxel collision with swept axis-separated movement, step-free
  auto-sliding along walls, jumping, sprinting, sneaking, swimming, flying.
* Amanatides–Woo **voxel raycast** for block targeting, with a reach limit.
* Placement refuses to put a block inside the player.

**Rendering**
* OpenGL **3.3 core** renderer: chunk VAOs, one 80×80 padded texture atlas,
  mipmapping, alpha-blended water/glass pass, distance fog, procedural sky with
  a sun, block selection outline and a full 2D HUD.
* Vectorised **chunk meshing**: only exposed faces become triangles, with
  per-vertex **ambient occlusion** and baked sky/face lighting.
* Frustum culling, and chunk meshes are rebuilt only when a chunk (or a
  neighbour, for correct borders/AO) actually changes.

---

## How it works

```
voxelcraft/
├── __main__.py     CLI entry point (python -m voxelcraft)
├── game.py         window, input handling, main loop, save/load
├── constants.py    tunable numbers (chunk size, speeds, fog ...)
├── blocks.py       block registry: ids, atlas tiles, physical properties
├── noise.py        vectorised 2D/3D Perlin noise, fBm, ridged noise, hashing
├── worldgen.py     terrain / caves / trees / spawn search
├── world.py        chunks, block access, edits, streaming, raycast, saves
├── mesher.py       chunk -> vertex buffer (faces + AO + light)
├── player.py       first person controller and AABB voxel physics
├── renderer.py     OpenGL 3.3 shaders, VAOs, passes, HUD drawing
├── camera.py       4x4 matrices and frustum culling
├── textures.py     procedurally painted texture atlas
├── font.py         5x7 bitmap font generated at runtime
├── ui.py           backend independent 2D quad batching for the HUD
├── hud.py          crosshair, hotbar, compass, debug overlay, menus
├── streaming.py    thread pool that generates chunks in the background
└── tests/          unit + end-to-end tests (no GPU required)
```

### Terrain generation

`worldgen.TerrainGenerator.height_at(x, z)` composes four noise layers:

```
height = sea_level + 2
       + continent(x/700, z/700) * 46     # slow fBm: ocean or land
       + hills(x/120, z/120)      * 16    # medium fBm: rolling hills
       + micro(x/38,  z/38)       *  5    # fast fBm: block-scale detail
       + mask² × ridges² × 78             # mountains
```

`mask` is a smoothstep of a large-scale fBm (where mountains are allowed) and
`ridges` is ridged multifractal noise (`1 - k·|perlin|`), which produces sharp
crests.  Surface blocks are chosen from the height and depth (grass, sand near
the water line, snow above the snow line, dirt then stone below), caves are
carved where two independent 3D noise fields are simultaneously near zero, and
water fills every empty cell above the terrain at or below sea level.

Because every layer is a pure function of `(seed, x, z)`, chunks can be built
in any order, in parallel, and regenerated later without seams.

### Meshing and lighting

`mesher.mesh_chunk` takes the 16×128×16 chunk padded with one block from each
neighbour and, for each of the six face directions, computes a boolean mask of
all exposed faces at once with NumPy, then expands those faces into triangle
lists.  Each vertex stores position, atlas UV and a baked light value:

```
light = face_brightness (top 1.0, ±X 0.74, ±Z 0.88, bottom 0.5)
      × sky_light (attenuated with depth below the column top)
      × ambient occlusion (0.55 / 0.72 / 0.86 / 1.0)
```

Ambient occlusion is the classic voxel rule — a corner darkens when the two
neighbouring side blocks are filled, and two filled sides make it darkest.
This is what gives the terrain its soft shaded creases at no runtime cost.

### Physics

The player is an axis-aligned box (0.6 × 1.8 × 0.6) moved one axis at a time at
a fixed 120 Hz sub-step.  For each axis the box is tested against all voxels it
would overlap; on a hit it is snapped flush against the block face, which
naturally gives smooth wall sliding, reliable step-free landing and a clean
`on_ground` flag for jumping.

---

## Tests

The test-suite needs **no GPU**: `voxelcraft/tests/glstub.py` replaces Pyglet's
GL layer with a recorder, so the real renderer, the real game loop and the real
input handlers can be exercised headlessly.

```bash
python voxelcraft/run_tests.py           # ~120 tests, ~1 minute
python voxelcraft/run_tests.py --fast    # quick unit tests only
```

Covered: noise properties (range, determinism, continuity, lattice zeros),
terrain invariants (bedrock floor, sea level, caves, seamless trees, spawn
safety), block access and edit persistence, padding of chunk borders, raycasting
through all six faces, mesh face culling / UV mapping / AO / lighting, matrices
and frustum culling, HUD geometry, the OpenGL renderer's draw calls, and
end-to-end gameplay (load → click → walk → jump → dig → place → save → reload).

## Regenerating the screenshots

```bash
python -m tools.render_preview --scene all --width 1280 --height 720
```

Scenes: `vista`, `beach`, `forest`, `cave`, `house`, `underwater`.

## Notes & limits

* Water does not flow, leaves are opaque, there is no day/night cycle and no
  mobs — this is a focused engine demo, not a Minecraft clone.
* The world is not infinite: chunks outside the render distance are unloaded
  (your edits are kept in `edits`, so the world is consistent when you return).
* Rendering distance is the main performance dial (7 chunks ⇒ 15×15 chunk
  columns). Use `-` to trade view distance for frame rate.
