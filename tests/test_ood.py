import json

from src import config
from src.providers import build_pool
from src.experience import ExperienceStore
from src.router import ACRouter
from src.policies import ThompsonSampling, HierarchicalThompson
from src.loop import evaluate_frozen
from src.tasks import make_stream, make_ood_stream
from src.benchmarks import load_swebench_lite, SWEBenchTask


def _ood_regret(hierarchical, n_train=2000, n_ood=600):
    specs = config.POOL
    costs = {s.name: s.cost_per_task for s in specs}
    providers = build_pool(specs, "mock")
    store = ExperienceStore(":memory:")
    if hierarchical:
        policy = HierarchicalThompson(store, costs, config.LAMBDA_COST, config.RANDOM_SEED)
    else:
        policy = ThompsonSampling(store, costs, config.LAMBDA_COST, config.RANDOM_SEED)
    router = ACRouter(providers, store, policy, costs, config.LAMBDA_COST,
                      dimension_aware=True, hierarchical=hierarchical)
    for t in make_stream(n_train, config.RANDOM_SEED):
        router.step(t)
    er = lambda m, t: providers[m].p_success(t) - config.LAMBDA_COST * costs[m]
    return evaluate_frozen(router, make_ood_stream(n_ood), er)["cum_regret"]


def test_hierarchical_priors_transfer_to_ood():
    assert _ood_regret(True) < _ood_regret(False)


def test_load_swebench_lite(tmp_path):
    f = tmp_path / "swe.jsonl"
    rows = [
        {"instance_id": "django__django-1", "repo": "django/django",
         "base_commit": "abc", "problem_statement": "Fix the admin bug.",
         "test_patch": "...", "FAIL_TO_PASS": ["t1"]},
        {"instance_id": "x", "problem_statement": ""},   # skipped: no statement
    ]
    f.write_text("\n".join(json.dumps(r) for r in rows))
    tasks = load_swebench_lite(str(f))
    assert len(tasks) == 1 and isinstance(tasks[0], SWEBenchTask)
    assert tasks[0].task_type == "agentic" and tasks[0].dimension.endswith("hard")
