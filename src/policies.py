"""Routing policies — the Action-selection brain.

Given a dimension key, the task, and the candidate arms, pick one model. All
policies share `choose(dim_key, task, models) -> model_name`.

  StaticPolicy / RandomPolicy   -> baselines (no learning)
  EpsilonGreedy                 -> simple learner on per-dimension means
  ThompsonSampling              -> ACRouter core: sample P(success) from each Beta
                                   posterior, score reward = p - lambda*cost, argmax.
                                   Uncertainty in the posterior drives exploration.
  OraclePolicy                  -> knows true P(success) (mock only); defines the
                                   zero-regret reference.
"""
from __future__ import annotations
import random

from src.featurizer import key_hierarchy


class StaticPolicy:
    """Always route to one fixed arm."""
    def __init__(self, model: str):
        self.model = model

    def choose(self, dim_key, task, models):
        return self.model


class RandomPolicy:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def choose(self, dim_key, task, models):
        return self.rng.choice(models)


class EpsilonGreedy:
    """Exploit the best mean (cost-adjusted), explore at rate epsilon."""
    def __init__(self, store, costs, lambda_cost, epsilon=0.1, seed=0):
        self.store, self.costs, self.lmbda = store, costs, lambda_cost
        self.epsilon, self.rng = epsilon, random.Random(seed)

    def choose(self, dim_key, task, models):
        if self.rng.random() < self.epsilon:
            return self.rng.choice(models)
        return max(models, key=lambda m:
                   self.store.get(dim_key, m).mean_success - self.lmbda * self.costs[m])


class ThompsonSampling:
    """ACRouter core. Posterior sampling balances explore/exploit automatically."""
    def __init__(self, store, costs, lambda_cost, seed=0):
        self.store, self.costs, self.lmbda = store, costs, lambda_cost
        self.rng = random.Random(seed)

    def choose(self, dim_key, task, models):
        best, best_score = models[0], -1e9
        for m in models:
            s = self.store.get(dim_key, m)
            p = self.rng.betavariate(s.alpha, s.beta)        # sample P(success)
            score = p - self.lmbda * self.costs[m]           # cost-aware utility
            if score > best_score:
                best, best_score = m, score
        return best


class OraclePolicy:
    """Picks the truly-best arm using simulator ground truth (mock backend only)."""
    def __init__(self, providers, costs, lambda_cost):
        self.providers, self.costs, self.lmbda = providers, costs, lambda_cost

    def choose(self, dim_key, task, models):
        return max(models, key=lambda m:
                   self.providers[m].p_success(task) - self.lmbda * self.costs[m])


class LLMRouterPolicy:
    """The literal 'Agent-as-a-Router': an LLM reads the retrieved per-dimension
    statistics and picks an arm, with a one-line rationale.

    `complete_fn(prompt) -> text` is injectable so the policy is testable offline; by
    default it calls a cheap Claude model. The retrieved stats are exactly the
    Context the paper says closes the information gap — here a language model, rather
    than a bandit formula, does the reasoning over them.
    """
    def __init__(self, store, costs, lambda_cost, complete_fn=None, model_notes=None,
                 use_stats=True):
        self.store, self.costs, self.lmbda = store, costs, lambda_cost
        self.complete_fn = complete_fn or _default_complete
        self.model_notes = model_notes or {}
        self.use_stats = use_stats   # False == the "vanilla" LLM router (no stats)
        self.last_rationale = ""

    def choose(self, dim_key, task, models):
        if self.use_stats:
            rows = []
            for m in models:
                s = self.store.get(dim_key, m)
                note = f" — {self.model_notes[m]}" if m in self.model_notes else ""
                rows.append(f"- {m}: success={s.mean_success:.0%} over {s.n} runs, "
                            f"cost={self.costs[m]:.2f}{note}")
            candidates = "with their measured per-dimension performance:\n" + "\n".join(rows)
        else:
            # Vanilla router: the model only knows the menu, not what works where.
            candidates = "(no performance history available):\n" + \
                "\n".join(f"- {m}: cost={self.costs[m]:.2f}" for m in models)
        prompt = (
            "You route a coding task to exactly ONE model. Prefer higher success and "
            "lower cost; spend on strong models only when the task needs it.\n\n"
            f"Task dimension: {dim_key}\n"
            f"Task: {getattr(task, 'prompt', '')[:300]}\n\n"
            f"Candidates {candidates}"
            '\n\nReply with ONLY JSON: {"model": "<name>", "why": "<short reason>"}.'
        )
        choice, why = _parse_choice(self.complete_fn(prompt), models)
        self.last_rationale = why
        return choice


def _default_complete(prompt: str) -> str:
    """Default LLM call for LLMRouterPolicy — a cheap, fast Claude model."""
    from anthropic import Anthropic
    client = Anthropic()  # reads ANTHROPIC_API_KEY
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=200,
        messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")


def _parse_choice(text, models):
    """Pull {"model","why"} out of an LLM reply; degrade gracefully on bad output."""
    import json
    import re
    choice, why = "", ""
    try:
        m = re.search(r"\{.*\}", text or "", re.S)
        obj = json.loads(m.group(0)) if m else {}
        choice, why = obj.get("model", ""), obj.get("why", "")
    except Exception:
        pass
    if choice in models:
        return choice, why
    for m in models:                       # fallback: first arm named in the text
        if m in (text or ""):
            return m, why or "name-matched"
    return models[0], why or "fallback"


class HeuristicPolicy:
    """A fixed escalation rule over the same priors — the paper's heuristic baseline.

    Route to the cheapest arm whose learned success clears a threshold; if none is
    confident, use the strongest. A little exploration keeps the priors it relies on
    fresh (a pure rule with no exploration just collapses onto the fallback arm). The
    paper's point: a stats-augmented *learned* router surpasses exactly this rule.
    """
    def __init__(self, store, costs, threshold: float = 0.7, explore: float = 0.15, seed=0):
        self.store, self.costs, self.threshold = store, costs, threshold
        self.explore, self.rng = explore, random.Random(seed)

    def choose(self, dim_key, task, models):
        if self.rng.random() < self.explore:
            return self.rng.choice(models)
        for m in sorted(models, key=lambda x: self.costs[x]):     # cheapest first
            if self.store.get(dim_key, m).mean_success >= self.threshold:
                return m
        return max(models, key=lambda x: self.costs[x])           # else the strongest


class HierarchicalThompson:
    """Thompson sampling over a *multi-resolution* posterior (OOD generalization).

    For each arm it blends evidence from GLOBAL -> task-type -> specific-dimension
    cells, with finer levels weighted more. When the specific cell is unseen (an
    out-of-distribution dimension), the coarser levels still supply an informed
    prior — so experience transfers instead of cold-starting. Pair with
    ACRouter(hierarchical=True) so all levels get updated.
    """
    LEVEL_WEIGHTS = (0.25, 0.5, 1.0)   # GLOBAL, type, specific

    def __init__(self, store, costs, lambda_cost, seed=0):
        self.store, self.costs, self.lmbda = store, costs, lambda_cost
        self.rng = random.Random(seed)

    def choose(self, dim_key, task, models):
        keys = key_hierarchy(task)
        best, best_score = models[0], -1e9
        for m in models:
            alpha, beta = 1.0, 1.0
            for k, w in zip(keys, self.LEVEL_WEIGHTS):
                s = self.store.get(k, m)
                alpha += w * (s.alpha - 1.0)
                beta += w * (s.beta - 1.0)
            p = self.rng.betavariate(alpha, beta)
            score = p - self.lmbda * self.costs[m]
            if score > best_score:
                best, best_score = m, score
        return best
