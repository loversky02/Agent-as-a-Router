#!/usr/bin/env python3
"""Live, at-scale check of the paper's headline with a REAL LLM router.

Phase 1 (free, mock): a Thompson bandit explores the mock world to accumulate
per-dimension performance statistics — execution-grounded experience.
Phase 2 (real LLM, parallel): on a held-out test set, a REAL LLM router routes each
task WITH those statistics (+stats) vs WITHOUT them (vanilla). Providers stay mock,
so a ground-truth oracle lets us score cumulative regret. The independent frozen
decisions run concurrently to keep wall-clock (and token spend) sane.

    python scripts/run_live_eval.py [n_train] [m_test] [concurrency]
"""
import concurrent.futures as cf
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.providers import build_pool
from src.experience import ExperienceStore
from src.router import ACRouter
from src.featurizer import featurize
from src.policies import ThompsonSampling, LLMRouterPolicy, OraclePolicy, RandomPolicy
from src.loop import run_stream
from src.tasks import make_stream


def make_route(policy, models, retries=1):
    """Wrap a policy into task -> model, with a retry and graceful failure."""
    def route(task):
        key = featurize(task).dimension
        for attempt in range(retries + 1):
            try:
                return policy.choose(key, task, models)
            except Exception:
                if attempt == retries:
                    return None
    return route


def evaluate(route, tasks, er, models, parallel=1):
    if parallel > 1:
        with cf.ThreadPoolExecutor(max_workers=parallel) as ex:
            chosen = list(ex.map(route, tasks))
    else:
        chosen = [route(t) for t in tasks]
    cum, rewards, fails = 0.0, [], 0
    for t, m in zip(tasks, chosen):
        if m not in models:
            fails += 1
            m = models[0]
        best = max(er(x, t) for x in models)
        rewards.append(er(m, t))
        cum += best - rewards[-1]
    return {"avg_reward": sum(rewards) / len(rewards), "cum_regret": cum, "fails": fails}


def main():
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    m_test = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    conc = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    lmbda = config.LAMBDA_COST
    specs = config.POOL                 # (mock) arms we route between — oracle available
    costs = {s.name: s.cost_per_task for s in specs}
    providers = build_pool(specs, "mock")
    models = list(providers)
    er = lambda m, t: providers[m].p_success(t) - lmbda * costs[m]

    # Phase 1 — accumulate per-dimension stats for free.
    store = ExperienceStore(":memory:")
    trainer = ACRouter(providers, store,
                       ThompsonSampling(store, costs, lmbda, config.RANDOM_SEED),
                       costs, lmbda, True)
    run_stream(trainer, make_stream(n_train, config.RANDOM_SEED))
    print(f"Phase 1: trained per-dimension stats on {n_train} mock tasks (free).\n")

    test = make_stream(m_test, seed=4242)
    print(f"Phase 2: routing {m_test} held-out tasks. Real LLM brain = "
          f"{os.getenv('ACROUTER_ROUTER_MODEL', '?')}, concurrency = {conc}.\n")

    configs = [
        ("Oracle (free)", OraclePolicy(providers, costs, lmbda), 1),
        ("Thompson +stats (free)", ThompsonSampling(store, costs, lmbda, config.RANDOM_SEED), 1),
        ("Random (free)", RandomPolicy(1), 1),
        ("LLM router VANILLA (real)", LLMRouterPolicy(store, costs, lmbda, use_stats=False), conc),
        ("LLM router +STATS (real)", LLMRouterPolicy(store, costs, lmbda, use_stats=True), conc),
    ]
    results = {}
    print(f"{'router':<30}{'avg_reward':>12}{'cum_regret':>12}{'fails':>7}")
    print("-" * 61)
    for name, policy, par in configs:
        if par > 1:
            print(f"  ... {m_test} real LLM calls for: {name}", flush=True)
        results[name] = evaluate(make_route(policy, models), test, er, models, parallel=par)
        r = results[name]
        print(f"{name:<30}{r['avg_reward']:>12.3f}{r['cum_regret']:>12.1f}{r['fails']:>7}")

    v = results["LLM router VANILLA (real)"]["cum_regret"]
    s = results["LLM router +STATS (real)"]["cum_regret"]
    if v > 0:
        print(f"\n>> Per-dimension stats cut the REAL LLM router's regret by "
              f"{(v - s) / v:+.1%} (the paper reports +15.3%).")
    _plot(results)


def _plot(results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    names = list(results)
    vals = [results[n]["cum_regret"] for n in names]
    colors = ["#d6336c" if "+STATS" in n else "#1c7ed6" if "VANILLA" in n else "#adb5bd"
              for n in names]
    plt.figure(figsize=(9, 4.5))
    plt.barh(names, vals, color=colors)
    plt.gca().invert_yaxis()
    plt.xlabel("cumulative regret on held-out tasks (lower = better)")
    plt.title("Live: a real LLM router routes better WITH per-dimension statistics")
    plt.tight_layout()
    plt.savefig("live_regret.png", dpi=140)
    print("Saved plot -> live_regret.png")


if __name__ == "__main__":
    main()
