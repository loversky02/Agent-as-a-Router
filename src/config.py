"""Central configuration: the model pool, the cost/quality knob, and the backend.

The three "pools" from the project goal are all expressed as ModelSpec lists, so
switching which models the router chooses between is a one-line change. The mock
`mock_skill` field is used ONLY by the offline simulator (src/providers.py) and is
ignored by real providers.
"""
from __future__ import annotations
import os
from dataclasses import dataclass


@dataclass
class ModelSpec:
    name: str            # logical id used everywhere — the bandit "arm"
    provider: str        # "claude" | "openai" | "ollama"
    model_id: str        # provider-specific model id
    cost_per_task: float # relative cost in [0,1], used in the reward penalty
    mock_skill: float = 0.5  # hidden competence, simulator-only


# --- Claude ladder: one API key, a clean cost<->skill ramp (default pool) ---
CLAUDE_POOL = [
    ModelSpec("haiku",  "claude", "claude-haiku-4-5-20251001", 0.05, mock_skill=0.45),
    ModelSpec("sonnet", "claude", "claude-sonnet-4-6",         0.30, mock_skill=0.68),
    ModelSpec("opus",   "claude", "claude-opus-4-8",           1.00, mock_skill=0.90),
]

# --- Multi-provider: add real ids/keys when ready ---
MULTI_POOL = CLAUDE_POOL + [
    ModelSpec("gpt", "openai", "gpt-5-mini", 0.20, mock_skill=0.62),
]

# --- Local / open models via Ollama (~free) ---
LOCAL_POOL = [
    ModelSpec("qwen-coder", "ollama", "qwen2.5-coder:7b",  0.01, mock_skill=0.40),
    ModelSpec("deepseek",   "ollama", "deepseek-coder-v2", 0.02, mock_skill=0.55),
]

# --- 8 simulated "frontier" models (mock-only), mirroring the paper's 8-LLM pool.
#     A richer cost<->skill ladder makes the routing problem harder/more realistic. ---
EIGHT_POOL = [
    ModelSpec("nano",  "mock", "sim-nano",  0.02, mock_skill=0.30),
    ModelSpec("mini",  "mock", "sim-mini",  0.06, mock_skill=0.45),
    ModelSpec("small", "mock", "sim-small", 0.12, mock_skill=0.55),
    ModelSpec("mid",   "mock", "sim-mid",   0.25, mock_skill=0.65),
    ModelSpec("large", "mock", "sim-large", 0.45, mock_skill=0.74),
    ModelSpec("xl",    "mock", "sim-xl",    0.70, mock_skill=0.82),
    ModelSpec("pro",   "mock", "sim-pro",   1.00, mock_skill=0.90),
    ModelSpec("ultra", "mock", "sim-ultra", 1.40, mock_skill=0.94),
]

# --- Real Claude ladder via a self-hosted OpenAI-compatible gateway (9router) ---
NINEROUTER_POOL = [
    ModelSpec("haiku",  "openai", "cc/claude-haiku-4-5-20251001", 0.05),
    ModelSpec("sonnet", "openai", "cc/claude-sonnet-4-6",         0.30),
    ModelSpec("opus",   "openai", "cc/claude-opus-4-8",           1.00),
]


def _load_dotenv(path=".env"):
    """Load KEY=VALUE lines from a .env file into the environment (no dependency)."""
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()

# --- Active settings (env-overridable; defaults stay offline & free) ----------
# Switch a live run with, e.g.:  ACROUTER_POOL=ninerouter ACROUTER_BACKEND=real ...
_POOLS = {"claude": CLAUDE_POOL, "multi": MULTI_POOL, "local": LOCAL_POOL,
          "eight": EIGHT_POOL, "ninerouter": NINEROUTER_POOL}
POOL = _POOLS.get(os.getenv("ACROUTER_POOL", "claude"), CLAUDE_POOL)
BACKEND = os.getenv("ACROUTER_BACKEND", "mock")      # "real" -> call provider APIs
LAMBDA_COST = float(os.getenv("ACROUTER_LAMBDA", "0.3"))
RANDOM_SEED = 7
