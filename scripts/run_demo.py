#!/usr/bin/env python3
"""Entry point: run the offline ACRouter demo.

    python scripts/run_demo.py [n_tasks]

Runs the streaming, regret-based comparison on synthetic tasks (mock backend, no
API key, no cost), then shows one live routing decision with the reasoning the
router had at that moment. Flip src/config.py BACKEND='real' to call Claude for real.
"""
import os
import sys

# Make `import src.*` work when run as a plain script from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, eval as eval_mod
from src.providers import build_pool
from src.experience import ExperienceStore
from src.router import ACRouter
from src.policies import ThompsonSampling
from src.tasks import make_stream, Task


def demo_single_decision():
    """Warm up on a stream, then show how the router routes the next task."""
    specs = config.POOL
    costs = {s.name: s.cost_per_task for s in specs}
    providers = build_pool(specs, config.BACKEND)
    store = ExperienceStore(":memory:")
    policy = ThompsonSampling(store, costs, config.LAMBDA_COST, config.RANDOM_SEED)
    router = ACRouter(providers, store, policy, costs, config.LAMBDA_COST, dimension_aware=True)

    warmup = make_stream(1500, config.RANDOM_SEED)
    for t in warmup:
        router.step(t)

    # A deliberately easy task: a well-trained router should confidently pick the
    # cheap arm here (high success at a fraction of the cost).
    probe = Task("probe-easy", "[codegen/python] write a function that reverses a string",
                 "python", "codegen", difficulty=0.2)
    key = router.dim_key(probe)
    print("\n--- One routing decision (after warm-up) ---")
    print(f"task     : {probe.prompt}")
    print(f"dimension: {key}")
    print(f"{'arm':<10}{'learned P(success)':>20}{'cost':>8}{'samples':>9}")
    for m in router.model_names:
        s = store.get(key, m)
        print(f"{m:<10}{s.mean_success:>19.0%}{costs[m]:>8.2f}{s.n:>9}")
    chosen = router.step(probe)
    print(f"-> routed to: {chosen.model}  (passed={chosen.passed}, reward={chosen.reward:.3f})\n")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    eval_mod.run(n)
    demo_single_decision()
