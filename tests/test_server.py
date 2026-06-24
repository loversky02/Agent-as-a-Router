import importlib
import os

import pytest


def test_server_routes_records_and_learns(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    os.environ["ACROUTER_DB"] = str(tmp_path / "srv.sqlite")
    import src.server as srv
    importlib.reload(srv)                 # rebuild the store at the env-configured path
    from fastapi.testclient import TestClient

    c = TestClient(srv.app)
    r = c.post("/v1/chat/completions",
               json={"messages": [{"role": "user", "content": "write a python function to add two numbers"}]})
    assert r.status_code == 200
    j = r.json()
    assert j["model"] in j["x_router"]["candidate_success"]
    assert j["x_router"]["dimension"].startswith("python")

    fb = c.post("/v1/feedback", json={"id": j["id"], "success": True})
    assert fb.json()["ok"] is True
    assert len(c.get("/stats").json()["cells"]) >= 1
