#!/usr/bin/env python3
"""Run the Voxelcraft test suite (stdlib ``unittest``, no GPU required).

Usage::

    python voxelcraft/run_tests.py            # everything
    python voxelcraft/run_tests.py --fast     # skip the slower game-loop tests
    python voxelcraft/run_tests.py -v         # verbose
"""

from __future__ import annotations

import argparse
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

FAST_MODULES = [
    "voxelcraft.tests.test_noise",
    "voxelcraft.tests.test_worldgen",
    "voxelcraft.tests.test_mesher",
    "voxelcraft.tests.test_render",
]

ALL_MODULES = FAST_MODULES + [
    "voxelcraft.tests.test_world",
    "voxelcraft.tests.test_player",
    "voxelcraft.tests.test_cli",
    "voxelcraft.tests.test_game",
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true",
                        help="only run the quick unit tests")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("module", nargs="?", help="run a single test module")
    args = parser.parse_args(argv)

    modules = [args.module] if args.module else (FAST_MODULES if args.fast else ALL_MODULES)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for name in modules:
        try:
            suite.addTests(loader.loadTestsFromName(name))
        except Exception as exc:                      # pragma: no cover
            print(f"could not load {name}: {exc}", file=sys.stderr)
            return 2

    runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
