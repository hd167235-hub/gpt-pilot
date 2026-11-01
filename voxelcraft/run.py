#!/usr/bin/env python3
"""Start Voxelcraft.

Two ways to run the game, both equivalent::

    python -m voxelcraft                 # from the repository root
    python voxelcraft/run.py             # from anywhere

This wrapper simply adds the repository root to ``sys.path`` so the package can
be imported even if the game folder itself is the working directory.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from voxelcraft.__main__ import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
