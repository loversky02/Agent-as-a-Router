from src.featurizer import featurize, bucket_difficulty


def _f(prompt):
    return featurize(type("P", (), {"prompt": prompt})())


def test_python_function_not_misread_as_javascript():
    # Regression: "python function" used to match the JS "function" cue first.
    assert _f("write a python function to reverse a string").language == "python"


def test_language_detection():
    assert _f("fix the bug in my javascript function").language == "javascript"
    assert _f("refactor this golang module").language == "go"
    assert _f("build a React component").language == "javascript"


def test_type_detection():
    assert _f("fix this error in the code").task_type == "bugfix"
    assert _f("refactor the module").task_type == "refactor"
    assert _f("write tests for the parser").task_type == "test"
    assert _f("implement a json parser").task_type == "codegen"


def test_difficulty_buckets():
    assert bucket_difficulty(0.2) == "easy"
    assert bucket_difficulty(0.5) == "medium"
    assert bucket_difficulty(0.9) == "hard"
