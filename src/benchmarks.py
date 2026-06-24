"""Real coding benchmarks behind the same Task interface as src/tasks.py.

`load_mbpp` reads MBPP (Mostly Basic Python Problems): a natural-language prompt, a
reference solution, and a few `assert` tests per problem. The router treats each as a
task; the Verifier runs the asserts in a subprocess. Drop this in place of the
synthetic stream to reproduce the paper's findings on real data.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json

from src.featurizer import bucket_difficulty


@dataclass
class MBPPTask:
    id: str
    nl: str                     # the natural-language description
    tests: list                 # assert strings used by the Verifier
    reference: str              # canonical solution (for the executor self-test)
    imports: list = field(default_factory=list)
    language: str = "python"
    task_type: str = "codegen"

    @property
    def prompt(self) -> str:
        """What we actually send the model: spec + the tests it must satisfy."""
        return (f"{self.nl}\n\nYour solution must pass these tests:\n"
                + "\n".join(self.tests)
                + "\n\nReturn only the Python code.")

    @property
    def difficulty(self) -> float:
        # MBPP has no ground-truth difficulty; proxy by spec + test length.
        return min(1.0, (len(self.nl) + sum(len(t) for t in self.tests)) / 1200.0)

    @property
    def dimension(self) -> str:
        return f"{self.language}:{self.task_type}:{bucket_difficulty(self.difficulty)}"


def load_mbpp(path: str, limit=None):
    with open(path) as f:
        raw = json.load(f)
    tasks = []
    for i, item in enumerate(raw):
        nl = item.get("prompt") or item.get("text") or ""
        tests = item.get("test_list") or []
        ref = item.get("code") or ""
        imports = item.get("test_imports") or []
        tid = str(item.get("task_id", i))
        if not (nl and tests and ref):
            continue
        tasks.append(MBPPTask(f"mbpp-{tid}", nl, tests, ref, imports))
        if limit and len(tasks) >= limit:
            break
    return tasks


@dataclass
class SWEBenchTask:
    """A SWE-bench Lite instance as an agentic (out-of-distribution) routing task."""
    id: str
    nl: str                              # problem statement
    repo: str = ""
    base_commit: str = ""
    test_patch: str = ""
    fail_to_pass: list = field(default_factory=list)
    language: str = "python"
    task_type: str = "agentic"
    difficulty: float = 0.85             # agentic, multi-file -> hard

    @property
    def prompt(self) -> str:
        return self.nl

    @property
    def dimension(self) -> str:
        return f"{self.language}:{self.task_type}:{bucket_difficulty(self.difficulty)}"


def load_swebench_lite(path: str, limit=None):
    """Load SWE-bench Lite instances as agentic OOD routing tasks (JSON or JSONL).

    NOTE: *executing* SWE-bench needs the official harness + Docker (apply patch, run
    repo tests) — out of scope here. These tasks are for routing / OOD-generalization
    experiments. Export `princeton-nlp/SWE-bench_Lite` to JSON/JSONL to use this.
    """
    with open(path) as f:
        text = f.read().strip()
    rows = json.loads(text) if text.startswith("[") else \
        [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    tasks = []
    for i, r in enumerate(rows):
        ps = r.get("problem_statement") or ""
        if not ps:
            continue
        tasks.append(SWEBenchTask(
            id=str(r.get("instance_id", i)), nl=ps, repo=r.get("repo", ""),
            base_commit=r.get("base_commit", ""), test_patch=r.get("test_patch", ""),
            fail_to_pass=r.get("FAIL_TO_PASS") or []))
        if limit and len(tasks) >= limit:
            break
    return tasks
