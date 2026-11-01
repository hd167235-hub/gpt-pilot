"""Block registry.

Every block type is described once here: which atlas tiles are used for the
top / side / bottom faces plus its physical and rendering properties.  Gameplay
code, the mesher and the HUD all read from this single table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Block ids (also stored in the chunk arrays, hence the small uint8 values).
AIR = 0
GRASS = 1
DIRT = 2
STONE = 3
COBBLE = 4
SAND = 5
WATER = 6
LOG = 7
LEAVES = 8
PLANKS = 9
BRICK = 10
GLASS = 11
SNOW = 12
BEDROCK = 13

NUM_BLOCKS = 14

# Texture tile indices (see voxelcraft.textures for the atlas layout).
T_GRASS_TOP = 0
T_GRASS_SIDE = 1
T_DIRT = 2
T_STONE = 3
T_COBBLE = 4
T_SAND = 5
T_LOG_SIDE = 6
T_LOG_TOP = 7
T_LEAVES = 8
T_PLANKS = 9
T_BRICK = 10
T_GLASS = 11
T_WATER = 12
T_SNOW = 13
T_BEDROCK = 14
T_WHITE = 15


@dataclass(frozen=True)
class BlockDef:
    """Static description of one block type."""

    id: int
    name: str
    top: Optional[int]
    side: Optional[int]
    bottom: Optional[int]
    opaque: bool = True          # hides the neighbouring face
    solid: bool = True           # collides with the player
    transparent: bool = False    # drawn in the alpha blended pass
    liquid: bool = False
    breakable: bool = True
    placeable: bool = True       # appears in the hotbar

    @property
    def is_air(self) -> bool:
        return self.id == AIR


BLOCKS: list[BlockDef] = [
    BlockDef(AIR, "Air", None, None, None, opaque=False, solid=False,
             breakable=False, placeable=False),
    BlockDef(GRASS, "Grass", T_GRASS_TOP, T_GRASS_SIDE, T_DIRT),
    BlockDef(DIRT, "Dirt", T_DIRT, T_DIRT, T_DIRT),
    BlockDef(STONE, "Stone", T_STONE, T_STONE, T_STONE),
    BlockDef(COBBLE, "Cobblestone", T_COBBLE, T_COBBLE, T_COBBLE),
    BlockDef(SAND, "Sand", T_SAND, T_SAND, T_SAND),
    BlockDef(WATER, "Water", T_WATER, T_WATER, T_WATER, opaque=False,
             solid=False, transparent=True, liquid=True, breakable=True,
             placeable=False),
    BlockDef(LOG, "Oak Log", T_LOG_TOP, T_LOG_SIDE, T_LOG_TOP),
    BlockDef(LEAVES, "Leaves", T_LEAVES, T_LEAVES, T_LEAVES),
    BlockDef(PLANKS, "Planks", T_PLANKS, T_PLANKS, T_PLANKS),
    BlockDef(BRICK, "Bricks", T_BRICK, T_BRICK, T_BRICK),
    BlockDef(GLASS, "Glass", T_GLASS, T_GLASS, T_GLASS, opaque=False,
             transparent=True),
    BlockDef(SNOW, "Snow", T_SNOW, T_SNOW, T_SNOW),
    BlockDef(BEDROCK, "Bedrock", T_BEDROCK, T_BEDROCK, T_BEDROCK,
             breakable=False),
]

BY_ID: dict[int, BlockDef] = {b.id: b for b in BLOCKS}
BY_NAME: dict[str, BlockDef] = {b.name.lower(): b for b in BLOCKS}

#: Blocks the player can select in the hotbar, in slot order.
HOTBAR_BLOCKS: tuple[int, ...] = (GRASS, DIRT, STONE, COBBLE, SAND, LOG, PLANKS, BRICK, GLASS)


def get(block_id: int) -> BlockDef:
    """Return the :class:`BlockDef` for ``block_id`` (air for unknown ids)."""
    return BY_ID.get(block_id, BLOCKS[AIR])


def by_name(name: str) -> BlockDef:
    """Look a block up by (case insensitive) name, e.g. ``"stone"``."""
    key = name.strip().lower().replace(" ", "_")
    if key == "cobblestone":
        key = "cobblestone"
    if key not in BY_NAME:
        raise KeyError(f"unknown block type: {name!r}")
    return BY_NAME[key]
