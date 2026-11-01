"""Tunable engine constants shared by every module."""

# --- world geometry -------------------------------------------------------
CHUNK_SIZE = 16          # blocks along X and Z
CHUNK_AREA = CHUNK_SIZE * CHUNK_SIZE
WORLD_HEIGHT = 128       # blocks along Y (0 = bottom layer)
SEA_LEVEL = 44           # water fills everything up to this height

# --- texture atlas --------------------------------------------------------
TILE_PX = 16             # pixels per block face
ATLAS_TILES = 4          # 4x4 grid of tiles
ATLAS_PX = TILE_PX * ATLAS_TILES

# --- player ---------------------------------------------------------------
PLAYER_WIDTH = 0.6
PLAYER_HEIGHT = 1.8
PLAYER_EYE = 1.62
WALK_SPEED = 4.6
SPRINT_SPEED = 7.2
FLY_SPEED = 10.0
SWIM_SPEED = 2.6
GRAVITY = 30.0
JUMP_SPEED = 8.8         # ~1.3 block jump
TERMINAL_VELOCITY = 55.0
REACH = 5.5              # block interaction distance

# --- rendering ------------------------------------------------------------
DEFAULT_RENDER_DISTANCE = 7   # in chunks
FOG_START_RATIO = 0.50
FOG_END_RATIO = 0.94

# --- simulation -----------------------------------------------------------
PHYSICS_STEP = 1.0 / 120.0
