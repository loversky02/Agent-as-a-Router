"""Providers = the "arms" the router chooses between.

Every backend implements the same `complete(task) -> Completion` interface, so the
router is provider-agnostic. `MockProvider` simulates outcomes deterministically,
which makes the whole C->A->F->C loop run offline, free, and reproducibly — and it
exposes `p_success`, the ground truth the oracle/regret needs. Real providers
(Claude / OpenAI / Ollama) share the interface and are filled in when you add keys.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import hashlib
import math


@dataclass
class Completion:
    text: str                       # produced code / answer
    model: str                      # logical arm id
    cost: float                     # cost incurred by this call
    latency: float                  # seconds (simulated or measured)
    mock_passed: Optional[bool] = None  # ground-truth pass/fail (mock only)


def _hash01(*parts) -> float:
    """Deterministic float in [0,1) from arbitrary parts — stable across runs."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16) / 0x100000000


class Provider:
    def __init__(self, spec):
        self.spec = spec

    def complete(self, task) -> Completion:
        raise NotImplementedError


class MockProvider(Provider):
    """A simulated coding model.

    Hidden success probability rises with skill, falls with task difficulty, and is
    tilted by a deterministic per-(model, dimension) offset so that *different models
    win on different dimensions* — exactly what the router must discover. The outcome
    is sampled reproducibly from (task id, model), so there is no run-to-run noise.
    """
    K = 9.0  # logistic sharpness

    def p_success(self, task) -> float:
        tilt = (_hash01(self.spec.name, task.dimension) - 0.5) * 0.30  # +/- 0.15
        regime = getattr(task, "regime", 0)
        if regime:  # a drift regime shifts each model's competence (non-stationarity)
            tilt += (_hash01(self.spec.name, "regime", regime) - 0.5) * 0.60
        skill = self.spec.mock_skill + tilt
        return 1.0 / (1.0 + math.exp(-self.K * (skill - task.difficulty)))

    def complete(self, task) -> Completion:
        passed = _hash01(task.id, self.spec.name) < self.p_success(task)
        latency = 0.4 + self.spec.cost_per_task * 3.0  # pricier ~ slower (sim)
        return Completion(
            text=f"# [{self.spec.name}] solution for {task.id}",
            model=self.spec.name,
            cost=self.spec.cost_per_task,
            latency=latency,
            mock_passed=passed,
        )


class ClaudeProvider(Provider):
    """Real Anthropic provider. Requires `anthropic` + ANTHROPIC_API_KEY.

    Returns the model's code; pass/fail is decided later by the Verifier (run tests
    or LLM-judge), NOT here — so `mock_passed` stays None.
    """
    def complete(self, task) -> Completion:
        import os
        import time
        from anthropic import Anthropic  # lazy import keeps the demo dependency-free

        client = Anthropic()  # reads ANTHROPIC_API_KEY
        t0 = time.time()
        msg = client.messages.create(
            model=self.spec.model_id,
            max_tokens=4096,
            messages=[{"role": "user", "content": task.prompt}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        # Cost in $ from token usage * per-token price is more accurate; for routing
        # we keep the relative cost knob from config so arms stay comparable.
        return Completion(text, self.spec.name, self.spec.cost_per_task, time.time() - t0)


def make_provider(spec, backend: str):
    """Factory: in 'mock' backend everything is simulated; in 'real' we dispatch
    on spec.provider. OpenAI/Ollama are left as clearly-marked extension points."""
    if backend == "mock":
        return MockProvider(spec)
    if spec.provider == "claude":
        return ClaudeProvider(spec)
    raise NotImplementedError(
        f"Real provider '{spec.provider}' not wired yet — add a Provider subclass "
        f"mirroring ClaudeProvider (same interface)."
    )


def build_pool(specs, backend: str):
    """Return {arm_name: provider} for a list of ModelSpecs."""
    return {s.name: make_provider(s, backend) for s in specs}
