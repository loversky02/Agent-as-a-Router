import json

from src.benchmarks import load_mbpp, MBPPTask
from src import sandbox


def test_load_mbpp_and_run_reference(tmp_path):
    f = tmp_path / "mini_mbpp.json"
    f.write_text(json.dumps([{
        "task_id": 1,
        "prompt": "Write a function add(a, b) that returns a + b.",
        "code": "def add(a, b):\n    return a + b",
        "test_list": ["assert add(2, 3) == 5"],
        "test_imports": [],
    }]))
    tasks = load_mbpp(str(f))
    assert len(tasks) == 1 and isinstance(tasks[0], MBPPTask)
    t = tasks[0]
    assert "must pass these tests" in t.prompt
    # The reference solution should pass its own tests through the real executor.
    ok, _ = sandbox.run_tests(t.reference, t.tests, t.imports)
    assert ok
