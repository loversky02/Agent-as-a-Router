"""FastAPI proxy: an OpenAI-compatible endpoint backed by the learning router.

  POST /v1/chat/completions  -> routes the request to a model, returns OpenAI-shaped
                                JSON plus `x_router` (which arm, why, candidate stats).
  POST /v1/feedback          -> {id, success}: closes the loop, updates the store.
  GET  /stats                -> current per-(dimension, model) experience.

Runs on the mock backend by default (no API key). Set config.BACKEND='real' to serve
real model output. Experience persists to data/experience_server.sqlite, so the proxy
keeps learning across restarts.

Run:  uvicorn src.server:app --reload      (from the repo root)
"""
from __future__ import annotations
import os
import time
import uuid

from fastapi import FastAPI
from pydantic import BaseModel

from src import config
from src.providers import build_pool
from src.experience import ExperienceStore
from src.policies import ThompsonSampling
from src.featurizer import featurize
from src.tasks import Task

app = FastAPI(title="Agent-as-a-Router proxy")

_specs = config.POOL
_costs = {s.name: s.cost_per_task for s in _specs}
_providers = build_pool(_specs, config.BACKEND)
_store = ExperienceStore(os.getenv("ACROUTER_DB", "data/experience_server.sqlite"))
_policy = ThompsonSampling(_store, _costs, config.LAMBDA_COST, config.RANDOM_SEED)
_models = list(_providers)
_pending = {}  # request id -> (dimension, model, cost, latency) awaiting feedback

_BUCKET_TO_DIFF = {"easy": 0.3, "medium": 0.55, "hard": 0.8}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list
    model: str = None  # ignored — the router decides


class Feedback(BaseModel):
    id: str
    success: bool


def _task_from_request(req: ChatRequest) -> Task:
    msgs = [ChatMessage(**m) if isinstance(m, dict) else m for m in req.messages]
    user = next((m.content for m in reversed(msgs) if m.role == "user"), "")
    feats = featurize(type("P", (), {"prompt": user})())  # featurize a prompt holder
    diff = _BUCKET_TO_DIFF[feats.difficulty_bucket]
    return Task(f"req-{uuid.uuid4().hex[:8]}", user, feats.language, feats.task_type, diff)


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    task = _task_from_request(req)
    dim = task.dimension
    model = _policy.choose(dim, task, _models)        # Action
    comp = _providers[model].complete(task)
    _pending[task.id] = (dim, model, comp.cost, comp.latency)  # await Feedback
    return {
        "id": task.id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": comp.text},
            "finish_reason": "stop",
        }],
        "x_router": {
            "dimension": dim,
            "chosen_model": model,
            "cost": comp.cost,
            "candidate_success": {m: round(_store.get(dim, m).mean_success, 3) for m in _models},
        },
    }


@app.post("/v1/feedback")
def feedback(fb: Feedback):
    if fb.id not in _pending:
        return {"ok": False, "error": "unknown or already-resolved id"}
    dim, model, cost, latency = _pending.pop(fb.id)
    _store.update(dim, model, fb.success, cost, latency)   # Context'
    return {"ok": True, "updated": {"dimension": dim, "model": model, "success": fb.success}}


@app.get("/stats")
def stats():
    return {"cells": [s.__dict__ | {"mean_success": round(s.mean_success, 3)}
                      for s in _store.snapshot()]}
