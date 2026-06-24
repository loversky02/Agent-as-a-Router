"""Streaming, regret-based comparison of routing strategies (the paper's eval).

Runs several policies over the *same* task stream with a fresh Experience Store
each, then reports cumulative regret, accuracy, avg reward, and total cost. The key
contrast is "Bandit (global, vanilla)" vs "ACRouter (per-dim)": same learner, but
one ignores task dimensions and one keeps per-dimension stats — isolating the gain
the paper attributes to task-dimension statistics. Oracle is the zero-regret floor.
"""
from __future__ import annotations

from src import config
from src.providers import build_pool
from src.experience import ExperienceStore
from src.router import ACRouter
from src.loop import run_stream, summarize
from src import policies as P

SPARK = "▁▂▃▄▅▆▇█"


def _costs(specs):
    return {s.name: s.cost_per_task for s in specs}


def _expected_reward_fn(providers, costs, lmbda):
    """True expected reward of an arm — only valid in the mock backend."""
    if not all(hasattr(p, "p_success") for p in providers.values()):
        return None
    return lambda m, task: providers[m].p_success(task) - lmbda * costs[m]


def _sparkline(values, width=48):
    if not values:
        return ""
    step = max(1, len(values) // width)
    pts = values[::step]
    lo, hi = min(pts), max(pts)
    rng = (hi - lo) or 1.0
    return "".join(SPARK[min(7, int((v - lo) / rng * 7))] for v in pts)


def _configs(providers, costs, lmbda, specs, seed):
    cheapest = min(specs, key=lambda s: s.cost_per_task).name
    strongest = max(specs, key=lambda s: s.cost_per_task).name
    return [
        # (label, make_policy, dimension_aware, hierarchical)
        (f"Static: {cheapest} (cheapest)", lambda s: P.StaticPolicy(cheapest), True, False),
        (f"Static: {strongest} (strongest)", lambda s: P.StaticPolicy(strongest), True, False),
        ("Random", lambda s: P.RandomPolicy(seed), True, False),
        ("Heuristic (rule on priors)", lambda s: P.HeuristicPolicy(s, costs), True, False),
        ("Bandit (global, vanilla)",
         lambda s: P.ThompsonSampling(s, costs, lmbda, seed), False, False),
        ("ACRouter (per-dim, +stats)",
         lambda s: P.ThompsonSampling(s, costs, lmbda, seed), True, False),
        ("ACRouter (hierarchical)",
         lambda s: P.HierarchicalThompson(s, costs, lmbda, seed), True, True),
        ("Oracle (zero-regret floor)",
         lambda s: P.OraclePolicy(providers, costs, lmbda), True, False),
    ]


def run(n_tasks=800):
    from src.tasks import make_stream  # imported here to avoid a cycle at module load

    specs = config.POOL
    lmbda, seed = config.LAMBDA_COST, config.RANDOM_SEED
    providers = build_pool(specs, config.BACKEND)
    costs = _costs(specs)
    er_fn = _expected_reward_fn(providers, costs, lmbda)
    tasks = make_stream(n_tasks, seed)

    print(f"\nACRouter eval — backend={config.BACKEND}, pool={[s.name for s in specs]}, "
          f"λ={lmbda}, tasks={n_tasks}\n")
    header = f"{'strategy':<32}{'accuracy':>10}{'avg_reward':>12}{'cost':>9}{'cum_regret':>12}"
    print(header)
    print("-" * len(header))

    results = {}
    for name, make_policy, dim_aware, hierarchical in _configs(providers, costs, lmbda, specs, seed):
        store = ExperienceStore(":memory:")
        router = ACRouter(providers, store, make_policy(store), costs, lmbda,
                          dim_aware, hierarchical)
        records = run_stream(router, tasks, er_fn)
        s = summarize(records)
        results[name] = (s, records)
        print(f"{name:<32}{s['accuracy']:>9.1%}{s['avg_reward']:>12.3f}"
              f"{s['total_cost']:>9.1f}{s['cum_regret']:>12.1f}")

    # The headline: how much do task-dimension stats buy us?
    g = results["Bandit (global, vanilla)"][0]["cum_regret"]
    a = results["ACRouter (per-dim, +stats)"][0]["cum_regret"]
    if g > 0:
        print(f"\n>> Task-dimension stats cut cumulative regret by "
              f"{(g - a) / g:+.1%} vs the vanilla router (paper reports +15.3%).")

    print("\nACRouter cumulative-regret curve (left=start, right=end):")
    print("  " + _sparkline([r.cum_regret for r in results["ACRouter (per-dim, +stats)"][1]]))
    print("  A flattening curve = the router has stopped making routing mistakes.\n")

    _maybe_plot(results, n_tasks)
    return results


def _maybe_plot(results, n_tasks):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("(install matplotlib for a regret.png plot — text summary shown above)")
        return
    plt.figure(figsize=(10, 6))
    # Worst (highest final regret) first, so the legend reads top-to-bottom like the plot.
    curves = [(name, recs) for name, (_s, recs) in results.items() if "Oracle" not in name]
    curves.sort(key=lambda nr: nr[1][-1].cum_regret, reverse=True)
    for name, recs in curves:
        plt.plot([r.cum_regret for r in recs], linewidth=1.8,
                 label=f"{name}  ({recs[-1].cum_regret:.0f})")
    plt.xlabel("tasks seen (streaming)")
    plt.ylabel("cumulative regret  (lower = better)")
    plt.title(f"Agent-as-a-Router: cumulative regret over {n_tasks:,} streaming tasks "
              "(CodeRouterBench-scale)")
    plt.grid(True, alpha=0.25)
    plt.legend(title="router  (final regret)", fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig("regret.png", dpi=140)
    print("Saved plot -> regret.png")


if __name__ == "__main__":
    run()
