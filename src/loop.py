"""Run the C->A->F->C loop over a stream of tasks, logging reward and regret.

`expected_reward_fn(model, task)` returns the *true* expected reward of an arm
(available in the mock backend via provider.p_success). When present we compute
pseudo-regret per step — the gap to the best arm — and accumulate it; this is the
streaming, regret-based evaluation the paper uses instead of static accuracy.
"""
from __future__ import annotations

from src.featurizer import key_hierarchy


def run_stream(router, tasks, expected_reward_fn=None):
    """Drive `router` across `tasks`; return the list of StepRecords."""
    records = []
    cum_regret = 0.0
    for task in tasks:
        rec = router.step(task)
        if expected_reward_fn is not None:
            best = max(expected_reward_fn(m, task) for m in router.model_names)
            rec.regret = best - expected_reward_fn(rec.model, task)
            cum_regret += rec.regret
            rec.cum_regret = cum_regret
        records.append(rec)
    return records


def evaluate_frozen(router, tasks, expected_reward_fn):
    """Route tasks WITHOUT learning — measures the quality of already-learned priors.

    Used to isolate out-of-distribution transfer: the store is frozen, so a flat
    router with empty cells for unseen dimensions samples blindly, while a
    hierarchical router still benefits from its coarser priors.
    """
    cum_regret = 0.0
    rewards = []
    for task in tasks:
        key = key_hierarchy(task)[-1] if router.hierarchical else router.dim_key(task)
        model = router.policy.choose(key, task, router.model_names)
        best = max(expected_reward_fn(m, task) for m in router.model_names)
        rewards.append(expected_reward_fn(model, task))
        cum_regret += best - rewards[-1]
    return {"cum_regret": cum_regret, "avg_reward": sum(rewards) / len(rewards), "n": len(tasks)}


def summarize(records):
    """Aggregate a run into headline numbers."""
    n = len(records)
    return {
        "n": n,
        "accuracy": sum(r.passed for r in records) / n,
        "avg_reward": sum(r.reward for r in records) / n,
        "total_cost": sum(r.cost for r in records),
        "cum_regret": records[-1].cum_regret if records else 0.0,
    }
