"""Run candidate code against a task's tests — the real Verifier path.

This is NOT a security sandbox: it executes model-generated code in a subprocess
with a timeout and a temp dir. Fine for trusted benchmarks (MBPP/HumanEval); for
untrusted code run it inside a container/VM. Returns (passed, detail).
"""
from __future__ import annotations
import os
import re
import subprocess
import sys
import tempfile


def extract_code(text: str) -> str:
    """Strip Markdown code fences a model may wrap its answer in."""
    if "```" in text:
        m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
        if m:
            return m.group(1)
    return text


def run_tests(code: str, tests, imports=None, timeout: float = 10.0):
    """Execute `code`, then the assert `tests`. Returns (passed, detail)."""
    program = "\n\n".join(filter(None, [
        "\n".join(imports or []),
        code,
        "\n".join(tests or []),
        "print('__ALL_TESTS_PASSED__')",
    ]))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "candidate.py")
        with open(path, "w") as f:
            f.write(program)
        try:
            proc = subprocess.run([sys.executable, path], capture_output=True,
                                  text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, "timeout"
    passed = "__ALL_TESTS_PASSED__" in proc.stdout
    if passed:
        return True, "ok"
    err = proc.stderr.strip().splitlines()
    return False, (err[-1] if err else "failed")
