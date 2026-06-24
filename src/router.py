"""ACRouter — the agent that routes, and runs one turn of the C->A->F->C loop.

  Context  : featurize(task) -> dimension key + the stats the policy reads
  Action   : policy picks an arm; the provider executes the task
  Feedback : verifier scores pass/fail; reward = correctness - lambda*cost
  Context' : store.update(...) folds the outcome back in -> better next decision

`dimension_aware=False` collapses every task to a single "ALL" bucket — the
"vanilla" router with no task-dimension statistics, used to measure the +stats gain.
"""
from __future__ import annotations
from dataclasses import dataclass

from src.featurizer import featurize, key_hierarchy
from src.verifier import verify, reward


@dataclass
class StepRecord:
    task_id: str
    dimension: str
    model: str
    passed: bool
    reward: float
    cost: float
    regret: float = 0.0
    cum_regret: float = 0.0


class ACRouter:
    def __init__(self, providers, store, policy, costs, lambda_cost,
                 dimension_aware: bool = True, hierarchical: bool = False):
        self.providers = providers
        self.store = store
        self.policy = policy
        self.costs = costs
        self.lmbda = lambda_cost
        self.dimension_aware = dimension_aware
        self.hierarchical = hierarchical    # also accumulate coarse keys for OOD transfer
        self.model_names = list(providers)

    def dim_key(self, task) -> str:
        return featurize(task).dimension if self.dimension_aware else "ALL"

    def step(self, task) -> StepRecord:
        keys = key_hierarchy(task) if self.hierarchical else [self.dim_key(task)]  # Context
        decision_key = keys[-1]
        model = self.policy.choose(decision_key, task, self.model_names)           # Action
        completion = self.providers[model].complete(task)
        passed = verify(task, completion)                                          # Feedback
        r = reward(passed, completion.cost, self.lmbda)
        for k in keys:                                                             # Context'
            self.store.update(k, model, passed, completion.cost, completion.latency)
        return StepRecord(task.id, decision_key, model, passed, r, completion.cost)
