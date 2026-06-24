from src.experience import ExperienceStore
from src.policies import ThompsonSampling, LLMRouterPolicy, HeuristicPolicy

COSTS = {"haiku": 0.05, "sonnet": 0.3, "opus": 1.0}
MODELS = list(COSTS)
TASK = type("T", (), {"prompt": "do a thing"})()


def _store():
    return ExperienceStore(":memory:")


def test_llm_router_parses_json_choice():
    p = LLMRouterPolicy(_store(), COSTS, 0.3,
                        complete_fn=lambda _: '{"model": "opus", "why": "hard task"}')
    assert p.choose("d", TASK, MODELS) == "opus"
    assert p.last_rationale == "hard task"


def test_llm_router_falls_back_to_named_arm():
    p = LLMRouterPolicy(_store(), COSTS, 0.3, complete_fn=lambda _: "I'd use sonnet here")
    assert p.choose("d", TASK, MODELS) == "sonnet"


def test_llm_router_garbage_defaults_safely():
    p = LLMRouterPolicy(_store(), COSTS, 0.3, complete_fn=lambda _: "??? no idea")
    assert p.choose("d", TASK, MODELS) == MODELS[0]


def test_thompson_returns_a_valid_arm():
    p = ThompsonSampling(_store(), COSTS, 0.3, seed=1)
    assert p.choose("d", TASK, MODELS) in MODELS


def test_vanilla_llm_router_returns_valid_arm():
    p = LLMRouterPolicy(_store(), COSTS, 0.3, use_stats=False,
                        complete_fn=lambda _: '{"model": "sonnet", "why": "guess"}')
    assert p.choose("d", TASK, MODELS) == "sonnet"


def test_heuristic_picks_cheapest_confident_arm():
    s = _store()
    for _ in range(20):
        s.update("d", "haiku", True, 0.05, 0.4)       # haiku becomes confidently good
    p = HeuristicPolicy(s, COSTS, threshold=0.7, explore=0.0)
    assert p.choose("d", TASK, MODELS) == "haiku"


def test_heuristic_falls_back_to_strongest_when_unsure():
    p = HeuristicPolicy(_store(), COSTS, threshold=0.7, explore=0.0)  # nobody confident
    assert p.choose("d", TASK, MODELS) == "opus"                     # strongest = highest cost
