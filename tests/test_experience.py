from src.experience import ExperienceStore


def test_beta_posterior_updates():
    s = ExperienceStore(":memory:")
    prior = s.get("d", "m").mean_success           # 0.5 from Beta(1,1)
    for _ in range(8):
        s.update("d", "m", True, 0.1, 0.5)
    s.update("d", "m", False, 0.1, 0.5)
    st = s.get("d", "m")
    assert st.n == 9
    assert st.mean_success > prior                 # mostly successes -> above prior
    assert abs(st.avg_cost - 0.1) < 1e-9


def test_cells_are_isolated():
    s = ExperienceStore(":memory:")
    s.update("d1", "m", True, 0.1, 0.5)
    assert s.get("d2", "m").n == 0


def test_snapshot_lists_cells():
    s = ExperienceStore(":memory:")
    s.update("d", "haiku", True, 0.05, 0.4)
    s.update("d", "opus", False, 1.0, 2.0)
    cells = {(c.dimension, c.model) for c in s.snapshot()}
    assert ("d", "haiku") in cells and ("d", "opus") in cells
