"""The behaviors that matter most — including the paper's central claim, as a test."""
from src import config
from src.providers import build_pool
from src.experience import ExperienceStore
from src.router import ACRouter
from src.policies import ThompsonSampling, RandomPolicy
from src.loop import run_stream, summarize
from src.tasks import make_stream


def _run(make_policy, dimension_aware, n=1500):
    specs = config.POOL
    costs = {s.name: s.cost_per_task for s in specs}
    providers = build_pool(specs, "mock")
    store = ExperienceStore(":memory:")
    router = ACRouter(providers, store, make_policy(store, costs), costs,
                      config.LAMBDA_COST, dimension_aware)
    er = lambda m, t: providers[m].p_success(t) - config.LAMBDA_COST * costs[m]
    return summarize(run_stream(router, make_stream(n, config.RANDOM_SEED), er))


def test_per_dimension_stats_reduce_regret():
    """ACRouter's thesis: the same learner with per-dimension stats beats one without."""
    per_dim = _run(lambda s, c: ThompsonSampling(s, c, config.LAMBDA_COST, 7), True)
    global_ = _run(lambda s, c: ThompsonSampling(s, c, config.LAMBDA_COST, 7), False)
    assert per_dim["cum_regret"] < global_["cum_regret"]


def test_learning_beats_random():
    learner = _run(lambda s, c: ThompsonSampling(s, c, config.LAMBDA_COST, 7), True)
    random_ = _run(lambda s, c: RandomPolicy(7), True)
    assert learner["avg_reward"] > random_["avg_reward"]


def test_step_record_shape():
    specs = config.POOL
    costs = {s.name: s.cost_per_task for s in specs}
    providers = build_pool(specs, "mock")
    store = ExperienceStore(":memory:")
    router = ACRouter(providers, store, ThompsonSampling(store, costs, 0.3, 1), costs, 0.3, True)
    rec = router.step(make_stream(1, 7)[0])
    assert rec.model in costs and isinstance(rec.passed, bool)
