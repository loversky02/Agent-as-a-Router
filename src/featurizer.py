"""Featurizer — the Context-building step of the loop.

Turns a raw coding task into the *task dimensions* the paper routes on. For
synthetic tasks the dimensions are ground truth; for real tasks `featurize`
estimates them from cheap heuristics (swap in an LLM classifier later). The
`dimension` string is the key under which the Experience Store accumulates
per-arm statistics — its granularity is the central knob the paper studies.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class TaskFeatures:
    language: str
    task_type: str
    difficulty_bucket: str

    @property
    def dimension(self) -> str:
        return f"{self.language}:{self.task_type}:{self.difficulty_bucket}"


def bucket_difficulty(d: float) -> str:
    return "easy" if d < 0.4 else "medium" if d < 0.7 else "hard"


def key_hierarchy(task):
    """Coarse-to-fine keys for the Experience Store.

    A multi-resolution view of experience: ``GLOBAL`` (overall arm strength),
    ``type:<task_type>`` (per-task-type), and the full per-dimension key. Coarser
    levels supply a prior when the specific cell is sparse — this is what lets
    learned priors generalize to *unseen* dimensions (out-of-distribution tasks).
    """
    f = featurize(task)
    return ["GLOBAL", f"type:{f.task_type}", f.dimension]


def featurize(task) -> TaskFeatures:
    """Map a task to the dimensions the router perceives.

    Synthetic tasks expose ground-truth fields, so we trust them. Real tasks fall
    through to heuristics over the prompt text and any provided metadata.
    """
    lang = getattr(task, "language", None) or _guess_language(task.prompt)
    ttype = getattr(task, "task_type", None) or _guess_type(task.prompt)
    diff = getattr(task, "difficulty", None)
    bucket = bucket_difficulty(diff) if diff is not None else _guess_difficulty(task.prompt)
    return TaskFeatures(lang, ttype, bucket)


# --- crude heuristics for real tasks (good enough to start; improve later) ---
def _guess_language(text: str) -> str:
    t = text.lower()
    # Explicit language names win over structural cues like the word "function".
    if "python" in t or "def " in t or ".py" in t or "django" in t or "flask" in t:
        return "python"
    if any(k in t for k in ("javascript", "typescript", "react", "node",
                            ".js", ".ts", "const ", "=>")):
        return "javascript"
    if "golang" in t or ".go" in t or " go " in t or "func " in t:
        return "go"
    return "python"  # sensible default


def _guess_type(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ("fix", "bug", "error", "fails")):
        return "bugfix"
    if any(k in t for k in ("refactor", "clean up", "rename")):
        return "refactor"
    if any(k in t for k in ("test", "assert", "coverage")):
        return "test"
    return "codegen"


def _guess_difficulty(text: str) -> str:
    n = len(text)
    return "easy" if n < 200 else "medium" if n < 800 else "hard"
