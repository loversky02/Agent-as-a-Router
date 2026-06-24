from src import sandbox


def test_pass_case():
    ok, detail = sandbox.run_tests("def add(a, b):\n    return a + b", ["assert add(2, 3) == 5"])
    assert ok and detail == "ok"


def test_fail_case():
    ok, detail = sandbox.run_tests("def add(a, b):\n    return a - b", ["assert add(2, 3) == 5"])
    assert not ok and "Assert" in detail


def test_timeout():
    ok, detail = sandbox.run_tests("while True:\n    pass", [], timeout=1.0)
    assert not ok and detail == "timeout"


def test_extract_code_strips_fences():
    assert sandbox.extract_code("```python\nx = 1\n```").strip() == "x = 1"
    assert sandbox.extract_code("no fences here") == "no fences here"
