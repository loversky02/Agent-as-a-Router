"""Verifier — the Feedback step: did the chosen action actually work?

Converts an execution into a scalar the store can learn from. In mock/benchmark
mode the ground-truth pass/fail comes from the simulator (standing in for running a
task's unit tests on CodeRouterBench). In real mode you plug in test execution or an
LLM-as-judge. Reward folds in cost: reward = correctness - lambda * cost.
"""
from __future__ import annotations

from src import sandbox


def verify(task, completion) -> bool:
    """Return True if the completion solves the task.

    Mock backend: trust the simulated outcome. Real benchmark task (one that carries
    `.tests`): run the candidate code against the asserts in a subprocess. Open-ended
    tasks with no tests are where you'd plug in an LLM-as-judge.
    """
    if completion.mock_passed is not None:
        return completion.mock_passed
    tests = getattr(task, "tests", None)
    if tests:
        code = sandbox.extract_code(completion.text)
        passed, _detail = sandbox.run_tests(code, tests, getattr(task, "imports", None))
        return passed
    raise NotImplementedError(
        "Open-ended task with no tests — plug in an LLM-as-judge here.")


def reward(passed: bool, cost: float, lambda_cost: float) -> float:
    """Cost-aware utility — what the router actually maximizes."""
    return (1.0 if passed else 0.0) - lambda_cost * cost
