#!/usr/bin/env python3
"""Out-of-distribution generalization demo.

Train two routers on an in-distribution stream (simple task types), then evaluate on
an OOD stream of unseen 'agentic' tasks. A flat per-dimension router cold-starts on
every new dimension; a hierarchical router reuses coarser (task-type / global) priors,
so experience transfers. Lower OOD regret = better generalization.

    python scripts/run_ood.py [n_train] [n_ood]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.providers import build_pool
from src.experience import ExperienceStore
from src.router import ACRouter
from src.policies import ThompsonSampling, HierarchicalThompson
from src.loop import evaluate_frozen
from src.tasks import make_stream, make_ood_stream


def build(hierarchical, costs, providers):
    store = ExperienceStore(":memory:")
    if hierarchical:
        policy = HierarchicalThompson(store, costs, config.LAMBDA_COST, config.RANDOM_SEED)
    else:
        policy = ThompsonSampling(store, costs, config.LAMBDA_COST, config.RANDOM_SEED)
    return ACRouter(providers, store, policy, costs, config.LAMBDA_COST,
                    dimension_aware=True, hierarchical=hierarchical)


if __name__ == "__main__":
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    n_ood = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    specs = config.POOL
    costs = {s.name: s.cost_per_task for s in specs}
    providers = build_pool(specs, "mock")
    er = lambda m, t: providers[m].p_success(t) - config.LAMBDA_COST * costs[m]
    in_dist = make_stream(n_train, config.RANDOM_SEED)
    ood = make_ood_stream(n_ood)

    print(f"Train on {n_train} in-distribution tasks, then route {n_ood} unseen "
          f"'agentic' OOD tasks.\n")
    print(f"{'router':<32}{'OOD avg_reward':>16}{'OOD cum_regret':>16}")
    print("-" * 64)
    for label, hier in [("Flat per-dimension", False), ("Hierarchical (backoff priors)", True)]:
        router = build(hier, costs, providers)
        for t in in_dist:
            router.step(t)                      # warm up on in-distribution
        s = evaluate_frozen(router, ood, er)    # route OOD with frozen priors
        print(f"{label:<32}{s['avg_reward']:>16.3f}{s['cum_regret']:>16.1f}")
    print("\nLower OOD regret = in-distribution experience transferred to new tasks.")
