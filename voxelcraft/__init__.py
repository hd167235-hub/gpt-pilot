"""Voxelcraft - a small Minecraft-like voxel sandbox built with Pyglet + NumPy.

The package is deliberately split so that everything except :mod:`voxelcraft.renderer`
and :mod:`voxelcraft.game` is pure Python/NumPy and can be unit tested headlessly:

    blocks.py      block registry (ids, textures, physical properties)
    noise.py       vectorised Perlin / fractal noise
    textures.py    procedurally generated texture atlas
    font.py        generated 5x7 bitmap font + atlas
    ui.py          backend independent 2D quad batching (HUD)
    camera.py      4x4 matrix maths + frustum culling
    worldgen.py    terrain, caves, beaches, mountains and trees
    world.py       chunk storage, block access, streaming, edit tracking
    mesher.py      vectorised chunk meshing with ambient occlusion
    player.py      first person controller, AABB physics, voxel raycast
    renderer.py    OpenGL 3.3 renderer (chunks, sky, HUD)
    hud.py         hotbar / crosshair / debug overlay layout
    game.py        window, input handling and main loop
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
