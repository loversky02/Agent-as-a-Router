from src import config
from src.providers import build_pool
from src.experience import ExperienceStore
from src.router import ACRouter
from src.policies import ThompsonSampling
from src.loop import run_stream
from src.tasks import make_drift_stream


def test_decay_one_is_plain_counting():
    s = ExperienceStore(":memory:", decay=1.0)
    for _ in range(5):
        s.update("d", "m", True, 0.1, 0.5)
    st = s.get("d", "m")
    assert st.alpha == 6.0 and st.beta == 1.0     # 5 successes on a Beta(1,1) prior


def test_decay_forgets_old_evidence():
    s = ExperienceStore(":memory:", decay=0.5)
    for _ in range(3):
        s.update("d", "m", True, 0.1, 0.5)
    assert s.get("d", "m").alpha < 4.0            # would be 4.0 without forgetting


def _post_switch_regret(decay, n=3000, switch=1500):
    specs = config.POOL
    costs = {s.name: s.cost_per_task for s in specs}
    providers = build_pool(specs, "mock")
    store = ExperienceStore(":memory:", decay=decay)
    router = ACRouter(providers, store,
                      ThompsonSampling(store, costs, config.LAMBDA_COST, config.RANDOM_SEED),
                      costs, config.LAMBDA_COST, True)
    er = lambda m, t: providers[m].p_success(t) - config.LAMBDA_COST * costs[m]
    recs = run_stream(router, make_drift_stream(n, switch, config.RANDOM_SEED), er)
    return sum(r.regret for r in recs[switch:])


def test_forgetting_adapts_faster_to_drift():
    assert _post_switch_regret(0.9) < _post_switch_regret(1.0)
