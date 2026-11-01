"""Background chunk streaming.

Terrain generation is pure NumPy and releases the GIL, so chunks can be built on
worker threads while the game keeps rendering.  Only the *application* of a
finished chunk (writing it into the world dictionary and applying player edits)
happens on the main thread.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

from .world import World

log = logging.getLogger(__name__)


class ChunkStreamer:
    """Generates chunks in a thread pool and installs them when they are ready."""

    def __init__(self, world: World, workers: int = 2, max_pending: int = 12):
        self.world = world
        self.max_pending = max_pending
        self._pool = ThreadPoolExecutor(max_workers=max(1, workers),
                                        thread_name_prefix="voxel-chunk")
        self._pending: dict[tuple[int, int], Future] = {}
        self._closed = False
        self.generated = 0

    # -- public API ------------------------------------------------------
    @property
    def pending(self) -> int:
        return len(self._pending)

    def request_near(self, center_cx: int, center_cz: int, radius: int,
                     budget: Optional[int] = None) -> int:
        """Queue generation for the closest missing chunks (non blocking)."""
        if self._closed:
            return 0
        budget = self.max_pending - len(self._pending) if budget is None else budget
        if budget <= 0:
            return 0
        submitted = 0
        for _, cx, cz in self.world.missing_chunks(center_cx, center_cz, radius):
            if submitted >= budget:
                break
            if (cx, cz) in self._pending:
                continue
            future = self._pool.submit(self.world.generator.generate_chunk, cx, cz)
            self._pending[(cx, cz)] = future
            submitted += 1
        return submitted

    def poll(self, limit: Optional[int] = None) -> int:
        """Install finished chunks. Must be called from the main thread.

        Returns the number of chunks installed.
        """
        if not self._pending:
            return 0
        done = [key for key, future in self._pending.items() if future.done()]
        if limit is not None:
            done = done[:limit]
        installed = 0
        for key in done:
            future = self._pending.pop(key)
            try:
                data = future.result()
            except Exception:                      # pragma: no cover - worker crash
                log.exception("chunk generation failed for %s", key)
                continue
            self.world.add_generated_chunk(key[0], key[1], data)
            self.generated += 1
            installed += 1
        return installed

    def busy(self, center_cx: int, center_cz: int, radius: int) -> bool:
        """True while chunks inside ``radius`` are still missing or in flight."""
        if self._pending:
            return True
        return bool(self.world.missing_chunks(center_cx, center_cz, radius))

    def wait_for(self, center_cx: int, center_cz: int, radius: int,
                 poll_callback=None, max_wait: float = 30.0) -> None:
        """Block until every chunk within ``radius`` exists.

        ``poll_callback`` is called after each poll (used to draw the loading
        screen); it receives the fraction of chunks that are ready.
        """
        import time

        total = self._expected_count(radius)
        deadline = time.monotonic() + max_wait
        self.request_near(center_cx, center_cz, radius, budget=self.max_pending)
        while time.monotonic() < deadline:
            self.poll()
            self.request_near(center_cx, center_cz, radius, budget=self.max_pending)
            if not self.busy(center_cx, center_cz, radius):
                if poll_callback:
                    poll_callback(1.0)
                return
            if poll_callback:
                ready = total - len(self.world.missing_chunks(center_cx, center_cz, radius))
                poll_callback(max(0.0, min(0.99, ready / max(1, total))))
            time.sleep(0.005)
        log.warning("timed out waiting for chunks around %s,%s", center_cx, center_cz)

    def shutdown(self) -> None:
        self._closed = True
        self._pool.shutdown(wait=False, cancel_futures=True)

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _expected_count(radius: int) -> int:
        """Number of chunk columns inside a (rounded) circular radius."""
        count = 0
        for dz in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dz * dz <= (radius + 0.5) ** 2:
                    count += 1
        return count
