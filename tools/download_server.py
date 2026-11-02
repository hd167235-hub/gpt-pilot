"""Serve the built Voxelcraft release in ``download/`` as clickable links.

Usage::

    python3 -m tools.build_release      # create download/Voxelcraft.py + zip
    python3 -m tools.download_server    # then open http://<host>:8000/

The index page is generated in memory, so nothing is added to ``download/``.
"""

from __future__ import annotations

import argparse
import html
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

RELEASE_DIR = Path(__file__).resolve().parent.parent / "download"

PAGE = """<!doctype html>
<html lang="vi">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Voxelcraft 1.0.0 - download</title>
<style>
 :root {{ color-scheme: dark; }}
 body {{ margin: 0; padding: 40px 20px; background: #14161a; color: #e8eaed;
        font: 16px/1.6 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }}
 main {{ max-width: 720px; margin: 0 auto; }}
 h1 {{ margin: 0 0 4px; font-size: 28px; }}
 p.sub {{ margin: 0 0 28px; color: #9aa4b2; }}
 a.card {{ display: block; padding: 18px 20px; margin-bottom: 14px; border-radius: 12px;
          background: #1e2127; border: 1px solid #2c313a; color: #e8eaed;
          text-decoration: none; transition: border-color .15s, transform .15s; }}
 a.card:hover {{ border-color: #5aa9ff; transform: translateY(-1px); }}
 b {{ font-size: 18px; }}
 span.meta {{ display: block; color: #9aa4b2; font-size: 14px; margin-top: 4px; }}
 code, pre {{ background: #0f1115; border-radius: 8px; color: #a5d6a7; }}
 pre {{ padding: 14px 16px; overflow-x: auto; font-size: 14px; }}
 code {{ padding: 2px 6px; }}
</style>
<main>
 <h1>⛏️ Voxelcraft 1.0.0</h1>
 <p class="sub">Game voxel 3D kiểu Minecraft viết bằng Python (Pyglet + NumPy)</p>
 {cards}
 <h2 style="font-size:18px;margin:28px 0 8px">Chạy sau khi tải</h2>
 <pre>python -m pip install pyglet numpy
python Voxelcraft.py</pre>
 <p class="sub">Bản zip: giải nén rồi chạy <code>start_windows.bat</code>
  hoặc <code>./start_linux_mac.sh</code>. Cần OpenGL 3.3.</p>
</main>
"""


def human_size(num: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024 or unit == "GB":
            return f"{num:.0f} {unit}" if unit == "B" else f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} GB"


def index_page() -> bytes:
    cards = []
    for path in sorted(RELEASE_DIR.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        name = html.escape(path.name)
        cards.append(
            f'<a class="card" href="/{path.name}" download>'
            f"<b>{name}</b>"
            f'<span class="meta">{human_size(path.stat().st_size)} — bấm để tải</span>'
            f"</a>"
        )
    body = PAGE.format(cards="\n ".join(cards) or "<p>Chưa có file nào.</p>")
    return body.encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    server_version = "VoxelcraftRelease/1.0"

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        route = unquote(urlparse(self.path).path)
        if route in ("/", "/index.html"):
            self._send(index_page(), "text/html; charset=utf-8", head_only=False)
            return
        target = (RELEASE_DIR / route.lstrip("/")).resolve()
        if RELEASE_DIR not in target.parents or not target.is_file():
            self.send_error(404, "not found")
            return
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self._send_file(target, ctype)

    def do_HEAD(self) -> None:  # noqa: N802 - http.server API
        route = unquote(urlparse(self.path).path)
        target = (RELEASE_DIR / route.lstrip("/")).resolve()
        if RELEASE_DIR not in target.parents or not target.is_file():
            self.send_error(404, "not found")
            return
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self._send_file(target, ctype, head_only=True)

    def _send(self, body: bytes, ctype: str, *, head_only: bool) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def _send_file(self, path: Path, ctype: str, *, head_only: bool = False) -> None:
        size = path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.end_headers()
        if head_only:
            return
        with path.open("rb") as handle:
            while chunk := handle.read(256 * 1024):
                self.wfile.write(chunk)

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the Voxelcraft release files.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args(argv)

    if not RELEASE_DIR.is_dir() or not any(RELEASE_DIR.iterdir()):
        print("download/ is empty - run `python3 -m tools.build_release` first")
        return 1

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"serving {RELEASE_DIR} on http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
