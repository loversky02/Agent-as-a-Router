#!/usr/bin/env python3
"""Non-stationarity demo: does forgetting (decay) help when the world drifts?

Two per-dimension Thompson routers run the SAME drifting stream — one keeps all
evidence (decay=1.0), one forgets (decay<1). The regime switches half-way, changing
the optimal arms; the forgetting router should recover faster (lower post-switch
regret).

    python scripts/run_drift.py [n]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.providers import build_pool
from src.experience import ExperienceStore
from src.router import ACRouter
from src.policies import ThompsonSampling
from src.loop import run_stream
from src.tasks import make_drift_stream


def run(decay, tasks, providers, costs):
    store = ExperienceStore(":memory:", decay=decay)
    policy = ThompsonSampling(store, costs, config.LAMBDA_COST, config.RANDOM_SEED)
    router = ACRouter(providers, store, policy, costs, config.LAMBDA_COST, True)
    er = lambda m, t: providers[m].p_success(t) - config.LAMBDA_COST * costs[m]
    return run_stream(router, tasks, er)


def seg(records, lo, hi):
    return sum(r.regret for r in records[lo:hi])


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    switch = n // 2
    specs = config.POOL
    costs = {s.name: s.cost_per_task for s in specs}
    providers = build_pool(specs, "mock")
    tasks = make_drift_stream(n, switch, config.RANDOM_SEED)

    print(f"Drift at task {switch}/{n}; pool={[s.name for s in specs]}\n")
    print(f"{'router':<28}{'pre-switch regret':>18}{'post-switch regret':>20}")
    print("-" * 66)
    for label, decay in [("No forgetting (decay=1.0)", 1.0), ("Forgetting (decay=0.9)", 0.9)]:
        recs = run(decay, tasks, providers, costs)
        print(f"{label:<28}{seg(recs, 0, switch):>18.1f}{seg(recs, switch, n):>20.1f}")
    print("\nLower post-switch regret = faster adaptation to the drifted world.")
