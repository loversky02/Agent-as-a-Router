#!/usr/bin/env python3
"""Reproduce-the-paper entry point on real MBPP tasks.

    python scripts/run_mbpp.py [n]

If ANTHROPIC_API_KEY is set AND src/config.py BACKEND=='real', this routes each MBPP
task to a real Claude model, runs the asserts, and reports accuracy/reward/cost.

Otherwise it runs a *verifier self-test*: it pushes MBPP reference solutions through
the real test executor to prove the end-to-end harness works without spending a cent.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, sandbox
from src.benchmarks import load_mbpp
from src.providers import build_pool
from src.experience import ExperienceStore
from src.router import ACRouter
from src.policies import ThompsonSampling

DATA = "data/mbpp.json"


def self_test(tasks):
    ok = 0
    for t in tasks:
        passed, detail = sandbox.run_tests(t.reference, t.tests, t.imports)
        ok += passed
        if not passed:
            print(f"  [ref-fail] {t.id}: {detail}")
    print(f"\nVerifier self-test: {ok}/{len(tasks)} reference solutions pass "
          f"({ok / len(tasks):.0%}) — the real test-execution path works.")


def routed_eval(tasks):
    specs = config.POOL
    costs = {s.name: s.cost_per_task for s in specs}
    providers = build_pool(specs, "real")
    store = ExperienceStore("data/experience_mbpp.sqlite")
    policy = ThompsonSampling(store, costs, config.LAMBDA_COST, config.RANDOM_SEED)
    router = ACRouter(providers, store, policy, costs, config.LAMBDA_COST, True)
    passed = total_reward = total_cost = 0.0
    for i, t in enumerate(tasks, 1):
        rec = router.step(t)
        passed += rec.passed
        total_reward += rec.reward
        total_cost += rec.cost
        print(f"[{i}/{len(tasks)}] {t.id} dim={rec.dimension} -> {rec.model} "
              f"passed={rec.passed}")
    n = len(tasks)
    print(f"\nMBPP routed eval: accuracy={passed / n:.1%} "
          f"avg_reward={total_reward / n:.3f} total_cost={total_cost:.1f}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    if not os.path.exists(DATA):
        sys.exit(f"Missing {DATA} — run: python scripts/fetch_mbpp.py")
    tasks = load_mbpp(DATA, limit=n)
    print(f"Loaded {len(tasks)} MBPP tasks from {DATA}")
    if os.getenv("ANTHROPIC_API_KEY") and config.BACKEND == "real":
        routed_eval(tasks)
    else:
        print("(no ANTHROPIC_API_KEY or BACKEND!='real' -> offline verifier self-test)\n")
        self_test(tasks)
