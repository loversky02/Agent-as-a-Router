"""Experience Store — the memory that turns routing into a *learning* loop.

Keeps a Beta(alpha, beta) posterior over P(success) for every (dimension, model)
cell, plus running cost/latency/use counts. This is the "task-dimension performance
statistics" the paper shows is worth a +15.3% relative gain. Backed by SQLite so the
accumulated, execution-grounded experience persists across runs.

Thread-safe (check_same_thread=False + a lock) so the FastAPI proxy, which serves
requests from a threadpool, can share one store.
"""
from __future__ import annotations
from dataclasses import dataclass
import sqlite3
import threading


@dataclass
class Stat:
    dimension: str
    model: str
    alpha: float   # successes + 1 (Beta prior)
    beta: float    # failures + 1 (Beta prior)
    n: int
    avg_cost: float
    avg_latency: float

    @property
    def mean_success(self) -> float:
        return self.alpha / (self.alpha + self.beta)


class ExperienceStore:
    def __init__(self, path: str = ":memory:", decay: float = 1.0):
        # decay<1 exponentially forgets old evidence, so the router tracks drift
        # (non-stationarity). decay==1.0 keeps every observation forever.
        self.decay = decay
        self._lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS stats(
                 dimension TEXT, model TEXT,
                 alpha REAL DEFAULT 1, beta REAL DEFAULT 1, n INTEGER DEFAULT 0,
                 sum_cost REAL DEFAULT 0, sum_latency REAL DEFAULT 0,
                 PRIMARY KEY(dimension, model))"""
        )
        self.db.commit()

    def get(self, dimension: str, model: str) -> Stat:
        with self._lock:
            row = self.db.execute(
                "SELECT alpha, beta, n, sum_cost, sum_latency FROM stats "
                "WHERE dimension=? AND model=?", (dimension, model)).fetchone()
        if row is None:
            return Stat(dimension, model, 1.0, 1.0, 0, 0.0, 0.0)
        alpha, beta, n, sc, sl = row
        return Stat(dimension, model, alpha, beta, n,
                    sc / n if n else 0.0, sl / n if n else 0.0)

    def update(self, dimension: str, model: str, passed: bool, cost: float, latency: float):
        """Fold one execution outcome back into the posterior (the F->C step).

        With decay<1 the existing evidence (counts above the Beta(1,1) prior) is
        discounted before the new observation is added — exponential forgetting.
        """
        s = 1.0 if passed else 0.0
        with self._lock:
            self.db.execute(
                "INSERT OR IGNORE INTO stats(dimension, model) VALUES(?, ?)",
                (dimension, model))
            self.db.execute(
                "UPDATE stats SET "
                "alpha = 1 + (alpha - 1) * ? + ?, "
                "beta  = 1 + (beta  - 1) * ? + ?, "
                "n = n + 1, sum_cost = sum_cost + ?, sum_latency = sum_latency + ? "
                "WHERE dimension=? AND model=?",
                (self.decay, s, self.decay, 1.0 - s, cost, latency, dimension, model))
            self.db.commit()

    def snapshot(self):
        """All learned cells, for inspection/printing."""
        with self._lock:
            rows = self.db.execute(
                "SELECT dimension, model, alpha, beta, n, sum_cost, sum_latency "
                "FROM stats ORDER BY dimension, model").fetchall()
        return [Stat(d, m, a, b, n, sc / n if n else 0, sl / n if n else 0)
                for d, m, a, b, n, sc, sl in rows]
