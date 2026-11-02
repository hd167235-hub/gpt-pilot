#!/usr/bin/env python3
"""Build the downloadable Voxelcraft release.

Produces two artefacts in ``download/``:

``Voxelcraft.py``
    A **single-file** edition of the game.  Every module of the ``voxelcraft``
    package is embedded as source and installed into ``sys.modules`` at start-up,
    so all you need is Python + ``pip install pyglet numpy``.

``voxelcraft-1.0.zip``
    The full project: package, tests, preview tools, documentation and
    screenshots, plus double-click launchers for Windows and Linux/macOS.

Usage::

    python -m tools.build_release
"""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "voxelcraft"
TOOLS = ROOT / "tools"
SHOTS = ROOT / "docs" / "shots"
OUT = ROOT / "download"

VERSION = "1.0.0"
FIXED_TIME = (2024, 1, 1, 0, 0, 0)  # keeps the zip reproducible

#: modules in dependency order (imports only ever point backwards)
MODULE_ORDER = [
    "constants",
    "blocks",
    "noise",
    "textures",
    "font",
    "ui",
    "camera",
    "worldgen",
    "world",
    "mesher",
    "player",
    "hud",
    "streaming",
    "renderer",
    "game",
    "__main__",
]

BUNDLE_HEADER = "\n".join([
    "#!/usr/bin/env python3",
    '"""Voxelcraft {version} - single-file edition.',
    "",
    "A small Minecraft-like voxel sandbox: Perlin-noise terrain, a first-person",
    "controller, and left-click-to-destroy / right-click-to-place blocks.",
    "",
    "Requirements",
    "------------",
    "    Python 3.9+ and two libraries:",
    "        python -m pip install pyglet numpy",
    "",
    "Run",
    "---",
    "    python Voxelcraft.py",
    "    python Voxelcraft.py --seed 1234 --render-distance 9",
    "    python Voxelcraft.py --help          # all options",
    "    python Voxelcraft.py --list-blocks   # dump the block registry",
    "",
    "Controls",
    "--------",
    "    W A S D  walk       SPACE  jump / swim / fly up     SHIFT  sneak / fly down",
    "    CTRL     sprint     F or double-tap SPACE  fly      MOUSE  look",
    "    LEFT CLICK destroy   RIGHT CLICK place              MIDDLE CLICK pick",
    "    1-9 / wheel  select block   F3 debug   F1 hide HUD   R respawn   ESC pause",
    "",
    "This file is generated from the voxelcraft package by tools/build_release.py;",
    "the readable, modular source ships in the voxelcraft-{version}.zip release.",
    '"""',
    "",
    "from __future__ import annotations",
    "",
])

#: the loader appended to the single-file edition
BUNDLE_RUNTIME = "\n".join([
    "import importlib.abc",
    "import importlib.util",
    "import sys",
    "import types",
    "",
    "",
    "class _EmbeddedLoader(importlib.abc.Loader):",
    '    """Executes an embedded source string as a module."""',
    "",
    "    def create_module(self, spec):",
    "        return None                      # default module creation",
    "",
    "    def exec_module(self, module) -> None:",
    '        name = module.__name__.rsplit(".", 1)[-1]',
    "        source = SOURCES[name]",
    '        code = compile(source, f"<voxelcraft/{name}.py>", "exec")',
    "        exec(code, module.__dict__)",
    "",
    "",
    "class _EmbeddedFinder(importlib.abc.MetaPathFinder):",
    '    """Serves ``voxelcraft.*`` from the embedded sources, on demand."""',
    "",
    "    def find_spec(self, fullname, path=None, target=None):",
    '        if not fullname.startswith("voxelcraft."):',
    "            return None",
    '        name = fullname.rsplit(".", 1)[-1]',
    "        if name not in SOURCES:",
    "            return None",
    "        return importlib.util.spec_from_loader(fullname, _EmbeddedLoader())",
    "",
    "",
    "def _install() -> None:",
    '    """Publish the embedded package; modules load lazily as they are imported."""',
    '    package = types.ModuleType("voxelcraft")',
    "    package.__path__ = []            # plain package, no filesystem needed",
    '    package.__package__ = "voxelcraft"',
    "    package.__version__ = VERSION",
    '    sys.modules["voxelcraft"] = package',
    "    sys.meta_path.insert(0, _EmbeddedFinder())",
    "",
    "",
    "def main(argv=None) -> int:",
    "    argv = sys.argv[1:] if argv is None else list(argv)",
    "    _install()",
    '    from voxelcraft.__main__ import main as cli_main',
    "    return cli_main(argv)",
    "",
    "",
    'if __name__ == "__main__":',
    "    raise SystemExit(main())",
    "",
])


def render_single_file() -> str:
    """Build the single-file edition of the game as text."""
    sources: dict[str, str] = {}
    for name in MODULE_ORDER:
        path = PKG / f"{name}.py"
        if not path.exists():
            raise SystemExit(f"missing module: {path}")
        sources[name] = path.read_text(encoding="utf-8")

    # data first, then the loader (its __main__ guard calls main() at the end)
    chunks: list[str] = [BUNDLE_HEADER.format(version=VERSION)]
    chunks.append(f"VERSION = {VERSION!r}")
    chunks.append(f"ORDER = {MODULE_ORDER!r}")
    chunks.append("")
    chunks.append("SOURCES = {")
    for name in MODULE_ORDER:
        chunks.append(f"    {name!r}: {sources[name]!r},")
    chunks.append("}")
    chunks.append("")
    chunks.append(BUNDLE_RUNTIME)

    return "\n".join(chunks)


def build_single_file() -> Path:
    """Write the single-file edition to ``download/Voxelcraft.py``."""
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "Voxelcraft.py"
    target.write_text(render_single_file(), encoding="utf-8")
    os.chmod(target, 0o755)
    return target


def build_zip(single_file: Path) -> Path:
    """Package the full project for download.

    The archive is byte-for-byte reproducible: entries are sorted and every
    timestamp is pinned, so re-running the build never dirties the git tree.
    """
    target = OUT / f"voxelcraft-{VERSION}.zip"
    base = f"voxelcraft-{VERSION}"
    staging: list[tuple[str, bytes]] = []

    def add_file(path: Path, arcname: str) -> None:
        staging.append((arcname, path.read_bytes()))

    def add_text(arcname: str, text: str) -> None:
        staging.append((arcname, text.encode("utf-8")))

    # single-file edition at the top level
    add_file(single_file, f"{base}/Voxelcraft.py")

    # the modular package + tests
    for path in sorted(PKG.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        add_file(path, f"{base}/{path.relative_to(ROOT).as_posix()}")
    for name in ("README.md", "requirements.txt", ".gitignore"):
        add_file(PKG / name, f"{base}/voxelcraft/{name}")

    # preview tooling (optional: needs pillow for PNG output)
    for path in sorted(TOOLS.glob("*.py")):
        add_file(path, f"{base}/tools/{path.name}")

    # docs + screenshots so the README renders
    for path in sorted(SHOTS.glob("*.png")):
        add_file(path, f"{base}/docs/shots/{path.name}")

    # launchers, license and quick start
    add_text(f"{base}/start_windows.bat", WINDOWS_LAUNCHER)
    add_text(f"{base}/start_linux_mac.sh", UNIX_LAUNCHER)
    add_text(f"{base}/HUONG_DAN.txt", QUICK_START)
    add_file(PKG / "README.md", f"{base}/README.md")
    if (ROOT / "LICENSE").exists():
        add_file(ROOT / "LICENSE", f"{base}/LICENSE")

    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for arcname, data in sorted(staging):
            info = zipfile.ZipInfo(arcname, date_time=FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o644 & 0xFFFF) << 16
            zf.writestr(info, data)
    return target


WINDOWS_LAUNCHER = r"""@echo off
REM Voxelcraft launcher (Windows) - installs the two dependencies, then plays.
setlocal
cd /d "%~dp0"
echo Installing/checking dependencies (pyglet, numpy)...
python -m pip install --quiet --disable-pip-version-check -r voxelcraft\requirements.txt
if errorlevel 1 (
    echo.
    echo Could not install the dependencies. Is Python 3.9+ on your PATH?
    pause
    exit /b 1
)
echo Starting Voxelcraft...
python -m voxelcraft %*
if errorlevel 1 (
    echo.
    echo The game exited with an error (see the message above).
    pause
)
endlocal
"""

UNIX_LAUNCHER = r"""#!/usr/bin/env bash
# Voxelcraft launcher (Linux / macOS) - installs the dependencies, then plays.
set -e
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
echo "Installing/checking dependencies (pyglet, numpy)..."
"$PYTHON" -m pip install --quiet --disable-pip-version-check -r voxelcraft/requirements.txt \
  || "$PYTHON" -m pip install --user --quiet --disable-pip-version-check -r voxelcraft/requirements.txt
echo "Starting Voxelcraft..."
exec "$PYTHON" -m voxelcraft "$@"
"""

QUICK_START = """VOXELCRAFT 1.0 - HUONG DAN NHANH
=================================

CACH 1 - MOT FILE DUY NHAT (don gian nhat)
------------------------------------------
1. Cai Python 3.9+ tu https://python.org (Windows: nho tick "Add Python to PATH")
2. Mo terminal / cmd tai thu muc giai nen va chay:
       python -m pip install pyglet numpy
       python Voxelcraft.py

CACH 2 - BAN DAY DU (co test, co launcher)
------------------------------------------
   Windows:               nhan dup start_windows.bat
   Linux / macOS:         chmod +x start_linux_mac.sh && ./start_linux_mac.sh
   Hoac thu cong:         python -m pip install -r voxelcraft/requirements.txt
                          python -m voxelcraft

CHOI NHU THE NAO
----------------
   W A S D        di chuyen              SPACE   nhay / boi / bay len
   SHIFT          ngoi / bay xuong       CTRL    chay nhanh
   F (hoac SPACE 2 lan)  bat/tat bay     CHUOT   nhin
   CHUOT TRAI     pha block              CHUOT PHAI   dat block
   CHUOT GIUA     chon block dang nhin    1-9 / lan chuot   chon o hotbar
   F3  hien thong tin debug              F1  an HUD
   R   hoi sinh                         ESC  tam dung (tu dong luu world)

YEU CAU DO HOA
--------------
   Can OpenGL 3.3 (moi GPU trong ~10 nam tro lai deu duoc).
   Linux headless / may ao:  sudo apt install libgl1 libglx-mesa0 libgl1-mesa-dri
                             xvfb-run -s "-screen 0 1280x720x24" python -m voxelcraft
   Neu thay loi 'Library "GL" not found' nghia la may chua co driver OpenGL.

TEST (khong can GPU)
--------------------
   python voxelcraft/run_tests.py --fast     # 91 test nhanh
   python voxelcraft/run_tests.py            # toan bo 189 test

GHI CHU
-------
   * World duoc luu o saves/world_<seed>.npz (chi gom seed + cac block da thay doi).
   * Doi seed de co the gioi khac:  python Voxelcraft.py --seed 12345
   * Xem tat ca tuy chon:           python Voxelcraft.py --help
"""


def main() -> int:
    single = build_single_file()
    print(f"single file : {single.relative_to(ROOT)}  "
          f"({single.stat().st_size / 1024:.0f} KB)")
    archive = build_zip(single)
    print(f"zip package : {archive.relative_to(ROOT)}  "
          f"({archive.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
