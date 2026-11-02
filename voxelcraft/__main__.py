"""Command line entry point: ``python -m voxelcraft``."""

from __future__ import annotations

import argparse
import logging
import os
import sys


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="voxelcraft",
        description="A small Minecraft-like voxel game written in Python "
                    "(Pyglet + NumPy).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=20260201,
                        help="world seed; every seed generates a different world")
    parser.add_argument("--render-distance", type=int, default=7, metavar="N",
                        help="view distance in chunks (2-14)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--no-msaa", action="store_true",
                        help="disable 4x MSAA (faster on weak GPUs)")
    parser.add_argument("--load", metavar="FILE", default=None,
                        help="load a saved world (.npz) instead of a fresh one")
    parser.add_argument("--save", metavar="FILE", default=None,
                        help="where to store block edits (default: saves/world_<seed>.npz)")
    parser.add_argument("--list-blocks", action="store_true",
                        help="print the block registry and exit")
    parser.add_argument("--version", action="version", version="voxelcraft 1.0.0")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from . import blocks

    if args.list_blocks:
        for defn in blocks.BLOCKS:
            flags = []
            if not defn.solid:
                flags.append("non-solid")
            if defn.transparent:
                flags.append("transparent")
            if not defn.breakable:
                flags.append("unbreakable")
            if defn.placeable:
                flags.append("hotbar")
            print(f"{defn.id:2d}  {defn.name:<12s} {' '.join(flags)}")
        return 0

    try:
        import pyglet  # noqa: F401
    except ImportError:
        requirements = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "requirements.txt")
        hint = (f"    {sys.executable} -m pip install -r {requirements}"
                if os.path.exists(requirements)
                else f"    {sys.executable} -m pip install pyglet numpy")
        print("Pyglet is not installed.  Install the requirements first:\n" + hint,
              file=sys.stderr)
        return 1

    try:
        from .game import run
    except ImportError as exc:
        # Most commonly: no OpenGL libraries on this machine.
        print(f"Could not initialise the graphics layer: {exc}\n\n"
              "Voxelcraft needs OpenGL 3.3.  On Linux install Mesa:\n"
              "  Debian/Ubuntu: sudo apt install libgl1 libglx-mesa0 libgl1-mesa-dri\n"
              "  Fedora:        sudo dnf install mesa-libGL mesa-dri-drivers\n"
              "On a headless machine run it under a virtual display:\n"
              "  xvfb-run -s '-screen 0 1280x720x24' python -m voxelcraft",
              file=sys.stderr)
        return 1

    try:
        run(seed=args.seed,
            render_distance=max(2, min(14, args.render_distance)),
            width=args.width,
            height=args.height,
            fullscreen=args.fullscreen,
            save_path=args.save,
            load_path=args.load,
            msaa=not args.no_msaa)
    except OSError as exc:
        # Most commonly: no OpenGL 3.3 capable driver / display available.
        print(f"Could not open a window: {exc}\n\n"
              "Voxelcraft needs OpenGL 3.3.  On a headless machine try:\n"
              "  * install the GPU drivers (Mesa: `libgl1-mesa-dri`)\n"
              "  * or run under Xvfb: `xvfb-run -s '-screen 0 1280x720x24' "
              "python -m voxelcraft`", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
