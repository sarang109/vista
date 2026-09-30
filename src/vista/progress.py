"""Once-a-minute progress and time-remaining messages for long runs.

The reporter is a daemon thread. Experiments call PROGRESS.update(done, total, unit) as work completes;
steps without a progress counter fall back to the duration the same step took in a previous run.
"""
from __future__ import annotations

import sys
import threading
import time


def fmt_dur(s: float | None) -> str:
    if s is None or s != s:
        return "unknown"
    s = int(round(max(s, 0)))
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return f"{h}h{m:02d}m{sec:02d}s" if h else f"{m}m{sec:02d}s"


class Progress:
    def __init__(self, interval: float = 60.0, stream=None, clock=time.time):
        self.interval, self.stream, self.clock = interval, stream or sys.stdout, clock
        self.steps: list[tuple[str, float]] = []  # (name, prior seconds)
        self.cur: str | None = None
        self.t_run = self.t_step = 0.0
        self.done = self.total = 0.0
        self.unit, self.scale = "", 1.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, steps: list[tuple[str, float]]):
        with self._lock:
            self.steps = list(steps)
            self.t_run = self.clock()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        with self._lock:
            self.cur = None

    def begin(self, name: str):
        with self._lock:
            self.cur, self.t_step = name, self.clock()
            self.done = self.total = 0.0
            self.unit, self.scale = "", 1.0

    def update(self, done: float, total: float, unit: str = "rows", scale: float = 1.0):
        with self._lock:
            self.done, self.total, self.unit, self.scale = done, total, unit, scale

    def message(self) -> str | None:
        with self._lock:
            if self.cur is None:
                return None
            now = self.clock()
            el_run, el_step = now - self.t_run, now - self.t_step
            names = [n for n, _ in self.steps]
            prior = dict(self.steps)
            later = sum(prior.get(n, 0.0) for n in names[names.index(self.cur) + 1:]) if self.cur in names else 0.0
            if self.total > 0 and self.done > 0:
                step_eta = el_step * (self.total - self.done) / self.done
                prog = f"{self.done * self.scale:,.0f}/{self.total * self.scale:,.0f} {self.unit} ({100 * self.done / self.total:.1f}%)"
            else:
                step_eta = max(prior.get(self.cur, 0.0) - el_step, 0.0) if self.cur in prior else None
                prog = "working"
            total_eta = None if step_eta is None else step_eta + later
            return (f"[vista] {time.strftime('%H:%M:%S')} step {self.cur}: {prog} | step elapsed {fmt_dur(el_step)}, "
                    f"remaining ~{fmt_dur(step_eta)} | run elapsed {fmt_dur(el_run)}, total remaining ~{fmt_dur(total_eta)}")

    def _loop(self):
        while not self._stop.wait(self.interval):
            m = self.message()
            if m:
                print(m, file=self.stream, flush=True)


PROGRESS = Progress()
