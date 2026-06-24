"""Synthetic coding-task stream — a free, reproducible stand-in for CodeRouterBench.

Each task carries ground-truth fields (language, type, difficulty) so the mock
provider can simulate outcomes and the oracle can be computed. To graduate from
demo to reproduction, replace `make_stream` with a real loader (HumanEval / MBPP /
SWE-bench Lite) that yields the same Task shape — everything downstream is unchanged.
"""
from __future__ import annotations
from dataclasses import dataclass
import random

from src.featurizer import bucket_difficulty

LANGUAGES = ["python", "javascript", "go"]
TASK_TYPES = ["codegen", "bugfix", "refactor", "test"]


@dataclass
class Task:
    id: str
    prompt: str
    language: str
    task_type: str
    difficulty: float            # 0..1 ground truth; hidden from the router
    regime: int = 0              # environment regime; >0 = a drifted world (non-stationarity)

    @property
    def dimension(self) -> str:
        return f"{self.language}:{self.task_type}:{bucket_difficulty(self.difficulty)}"


def make_stream(n: int, seed: int = 7):
    """A reproducible stream of n synthetic tasks."""
    rng = random.Random(seed)
    tasks = []
    for i in range(n):
        lang = rng.choice(LANGUAGES)
        ttype = rng.choice(TASK_TYPES)
        diff = round(rng.random(), 3)
        tasks.append(Task(f"t{i:04d}", f"[{ttype}/{lang}] synthetic task #{i}",
                          lang, ttype, diff))
    return tasks


def make_drift_stream(n: int, switch_at: int, seed: int = 7):
    """Like make_stream, but the world's regime flips at `switch_at` (drift).

    After the switch every model's per-dimension competence shifts, so the optimal
    arm changes — a router must forget stale experience to keep routing well.
    """
    stream = make_stream(n, seed)
    for i, t in enumerate(stream):
        t.regime = 0 if i < switch_at else 1
    return stream


def make_ood_stream(n: int, seed: int = 11):
    """Out-of-distribution stream: hard, multi-file 'agentic' tasks.

    Its `agentic` task_type never appears in make_stream, so the per-dimension cells
    are unseen — only coarser (type/global) priors can transfer. Mirrors the paper's
    OOD agentic-programming evaluation.
    """
    rng = random.Random(seed)
    tasks = []
    for i in range(n):
        lang = rng.choice(LANGUAGES)
        diff = round(0.6 + 0.4 * rng.random(), 3)   # agentic tasks are hard
        tasks.append(Task(f"ood{i:04d}", f"[agentic/{lang}] multi-file task #{i}",
                          lang, "agentic", diff))
    return tasks
