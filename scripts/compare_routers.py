#!/usr/bin/env python3
"""Compare two routing brains on the SAME task stream: a Thompson-sampling bandit vs
an LLM-as-Router, both reading the same per-dimension experience.

Providers stay on the MOCK backend so we keep ground-truth regret — only the router's
*decision* differs. By default the LLM-router uses a free offline greedy stub that
reasons over the stats string. With --live it calls a real Claude model for each
routing decision (needs ANTHROPIC_API_KEY): the faithful "Agent-as-a-Router".

    python scripts/compare_routers.py            # offline, free
    python scripts/compare_routers.py 60 --live  # real Claude routing (one call/task)
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.providers import build_pool
from src.experience import ExperienceStore
from src.router import ACRouter
from src.policies import ThompsonSampling, LLMRouterPolicy, _default_complete
from src.loop import run_stream, summarize
from src.tasks import make_stream

LAMBDA = config.LAMBDA_COST
_ROW = re.compile(r"- ([\w-]+): success=(\d+)% over \d+ runs, cost=([\d.]+)")


def greedy_stub(prompt: str) -> str:
    """A free stand-in for the LLM: pick the best (success - lambda*cost) from the stats."""
    best, best_score = "", -1e9
    for name, succ, cost in _ROW.findall(prompt):
        score = int(succ) / 100 - LAMBDA * float(cost)
        if score > best_score:
            best, best_score = name, score
    return '{"model": "%s", "why": "greedy over stats"}' % best


def run(make_policy, n):
    specs = config.POOL
    costs = {s.name: s.cost_per_task for s in specs}
    providers = build_pool(specs, "mock")
    store = ExperienceStore(":memory:")
    router = ACRouter(providers, store, make_policy(store, costs), costs, LAMBDA, True)
    er = lambda m, t: providers[m].p_success(t) - LAMBDA * costs[m]
    return summarize(run_stream(router, make_stream(n, config.RANDOM_SEED), er))


if __name__ == "__main__":
    live = "--live" in sys.argv
    pos = [a for a in sys.argv[1:] if not a.startswith("-")]
    n = int(pos[0]) if pos else (60 if live else 1000)
    complete_fn = _default_complete if live else greedy_stub
    mode = "real Claude" if live else "offline greedy stub"

    print(f"Comparing routers on {n} mock tasks — LLM-router brain = {mode}\n")
    th = run(lambda s, c: ThompsonSampling(s, c, LAMBDA, config.RANDOM_SEED), n)
    llm = run(lambda s, c: LLMRouterPolicy(s, c, LAMBDA, complete_fn=complete_fn), n)

    print(f"{'router':<34}{'avg_reward':>12}{'cum_regret':>12}")
    print("-" * 58)
    print(f"{'ThompsonSampling (bandit)':<34}{th['avg_reward']:>12.3f}{th['cum_regret']:>12.1f}")
    print(f"{'LLM-as-Router (' + mode + ')':<34}{llm['avg_reward']:>12.3f}{llm['cum_regret']:>12.1f}")
    print("\nLower cum_regret = better. Both read the same per-dimension stats; this "
          "isolates\nthe decision-maker (bandit formula vs. language-model reasoning).")
