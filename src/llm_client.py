"""Week 2: pluggable LLM client interface for the agent loop.

`ClaudeClient` calls the Anthropic Messages API (requires ANTHROPIC_API_KEY).
`LocalQwenClient` runs Qwen2.5-Coder-14B-Instruct entirely locally via
llama.cpp (GGUF, Q4_K_M) — no API key, no network calls after the one-time
model download, and no per-token cost, at the price of running slower than a
hosted API and needing a CUDA GPU with a few GB of free VRAM.
`MockLLMClient` is a deterministic, template-based stand-in that requires no
model at all: it reads the diagnostic feedback embedded in the prompt (the
same text a real LLM would read) and proposes the next factor_code by
mutating the best-so-far expression. Because it is reward-driven off the
*same* feedback text the agent loop constructs, running it under
`mode="execution_aware"` vs. `mode="baseline"` reproduces the qualitative
effect the paper studies (turnover-aware search converges to lower-turnover,
higher Net_IR factors) without spending API credits or GPU time — useful for
developing/debugging the pipeline before committing to a paid or local run.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path

from . import config


class LLMClient(ABC):
    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        """Return raw text completion for a single-turn system+user prompt."""


class ClaudeClient(LLMClient):
    def __init__(self, model: str = config.CLAUDE_MODEL, api_key: str | None = None, max_tokens: int = 1024):
        import anthropic  # imported lazily so MockLLMClient never needs the package/key

        self._client = anthropic.Anthropic(api_key=api_key)  # falls back to ANTHROPIC_API_KEY env var
        self._model = model
        self._max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


class LocalQwenClient(LLMClient):
    """Generator & Refiner running entirely on-box via llama.cpp.

    Loads a quantized GGUF checkpoint (default: Qwen2.5-Coder-14B-Instruct,
    Q4_K_M, ~9GB) once at construction and reuses it in-process for every
    round of the agent loop — no server to manage, no API key, no per-call
    cost. Download the weights first with
    `python scripts/download_local_model.py` (or pass an explicit
    `model_path` to an existing GGUF file, e.g. an AWQ/other-quant build).
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        n_gpu_layers: int = config.LOCAL_LLM_N_GPU_LAYERS,
        n_ctx: int = config.LOCAL_LLM_N_CTX,
        temperature: float = config.LOCAL_LLM_TEMPERATURE,
        top_p: float = config.LOCAL_LLM_TOP_P,
        max_tokens: int = config.LOCAL_LLM_MAX_TOKENS,
        tensor_split: list[float] | None = None,
        main_gpu: int = 0,
        seed: int | None = None,
        verbose: bool = False,
    ):
        from llama_cpp import Llama  # imported lazily: heavy, GPU-bound, optional dependency

        self._model_path = Path(model_path) if model_path else config.LOCAL_MODEL_PATH
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Local model not found at {self._model_path}. Run "
                f"`python scripts/download_local_model.py` to fetch "
                f"{config.LOCAL_MODEL_REPO}/{config.LOCAL_MODEL_FILE} first."
            )

        if seed is None:
            # llama.cpp's usual "-1 = randomize from system time" sentinel is not
            # honored by every llama-cpp-python build/version (verified here: passing
            # -1 produced byte-identical output across two separate process runs).
            # Generate the randomness ourselves so two LocalQwenClient() calls without
            # an explicit seed actually sample differently.
            import secrets
            seed = secrets.randbelow(2**31)
        self.seed = seed
        print(f"[LocalQwenClient] seed={seed} main_gpu={main_gpu}")

        self._llm = Llama(
            model_path=str(self._model_path),
            n_gpu_layers=n_gpu_layers,
            main_gpu=main_gpu,   # a matched pair (make_matched_pair) pins each instance to its own GPU
                                 # so two ~9GB model copies don't both try to fit on one 12GB card
            n_ctx=n_ctx,
            tensor_split=tensor_split,   # None = single GPU (device 0); e.g. [0.5, 0.5] to split across both
            seed=seed,
            verbose=verbose,
        )
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        response = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self._temperature,
            top_p=self._top_p,
            max_tokens=self._max_tokens,
        )
        return response["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Mock client
# ---------------------------------------------------------------------------

# (template_name, expression format string, (n_min, n_max))
TEMPLATES: list[tuple[str, str, tuple[int, int]]] = [
    ("reversal_fast", "rank(-1 * delta(close, {n}))", (1, 8)),
    ("reversal_slow", "rank(-1 * delta(close, {n}))", (9, 30)),
    ("momentum", "rank(delta(close, {n}) / (ts_std(returns, {n}) + 0.0001))", (10, 60)),
    ("volume_weighted_reversal", "rank(-1 * delta(close, {n}) * rank(volume))", (1, 10)),
    ("low_volatility", "rank(-1 * ts_std(returns, {n}))", (10, 60)),
    ("mean_reversion_z", "rank(-1 * (close - ts_mean(close, {n})) / (ts_std(close, {n}) + 0.0001))", (5, 40)),
    ("smoothed_momentum", "rank(ts_mean(delta(close, 1), {n}))", (5, 40)),
    ("range_breakout", "rank((close - ts_min(low, {n})) / (ts_max(high, {n}) - ts_min(low, {n}) + 0.0001))", (5, 40)),
    ("ts_rank_momentum", "rank(ts_rank(close, {n}))", (10, 60)),
    ("decayed_reversal", "rank(decay_linear(-1 * delta(close, 5), {n}))", (5, 60)),
    ("decayed_momentum", "rank(decay_linear(delta(close, 20) / (ts_std(returns, 20) + 0.0001), {n}))", (5, 60)),
    ("decayed_volume_reversal", "rank(decay_linear(-1 * delta(close, 5) * rank(volume), {n}))", (5, 60)),
]

_TEMPLATE_REGEX = {
    name: re.compile("^" + re.escape(fmt).replace(r"\{n\}", r"(\d+)") + "$")
    for name, fmt, _ in TEMPLATES
}


class MockLLMClient(LLMClient):
    def __init__(self, seed: int = config.RANDOM_SEED):
        import random

        self._rng = random.Random(seed)

    def _random_template(self) -> str:
        name, fmt, (lo, hi) = self._rng.choice(TEMPLATES)
        n = self._rng.randint(lo, hi)
        return fmt.format(n=n)

    def _match_template(self, code: str) -> tuple[str, int] | None:
        for name, pattern in _TEMPLATE_REGEX.items():
            m = pattern.match(code.strip())
            if m:
                return name, int(m.group(1))
        return None

    def _mutate(self, code: str) -> str:
        matched = self._match_template(code)
        if matched is None:
            return self._random_template()
        name, n = matched
        if self._rng.random() < 0.3:
            # swap to a different template, keep exploring nearby window sizes
            other = self._rng.choice([t for t in TEMPLATES if t[0] != name])
            other_name, fmt, (lo, hi) = other
            n_new = int(min(max(n, lo), hi))
            return fmt.format(n=n_new)
        fmt, (lo, hi) = next((f, r) for nm, f, r in TEMPLATES if nm == name)
        jitter = self._rng.uniform(0.5, 1.8)
        n_new = int(round(n * jitter))
        n_new = max(lo, min(hi, max(2, n_new)))
        return fmt.format(n=n_new)

    @staticmethod
    def _parse_field(user: str, label: str) -> tuple[str | None, float | None]:
        pattern = re.compile(
            re.escape(label) + r".*?code=`([^`]+)`.*?reward=(-?[\d.]+)", re.S
        )
        m = pattern.search(user)
        if not m:
            return None, None
        return m.group(1), float(m.group(2))

    def complete(self, system: str, user: str) -> str:
        best_code, _ = self._parse_field(user, "Best factor so far")
        if best_code is None or self._rng.random() < 0.25:
            code = self._random_template()
            rationale = "Exploring a new candidate template."
        else:
            code = self._mutate(best_code)
            rationale = "Mutated the best-so-far factor toward higher reward."
        return json.dumps({"factor_code": code, "rationale": rationale})


def make_client(kind: str = "mock", **kwargs) -> LLMClient:
    if kind == "mock":
        return MockLLMClient(**kwargs)
    if kind == "claude":
        return ClaudeClient(**kwargs)
    if kind in ("local", "local_qwen", "qwen"):
        return LocalQwenClient(**kwargs)
    raise ValueError(f"Unknown LLM client kind: {kind}")


def make_matched_pair(kind: str = "mock", seed: int | None = None, **kwargs) -> tuple[LLMClient, LLMClient]:
    """Two independently-constructed LLMClient instances with identical
    parameters -- including an identical explicit seed, for backends that
    have one -- so a baseline-mode and execution-aware-mode run_agent_loop
    call start from matched conditions rather than sharing one client's
    continuing random/sampling state across two sequential 50-round runs
    (the latter was this project's original design and is what motivated
    this function: see the "is it apples-to-apples" discussion in the
    manuscript's methodology section).

    For `kind="local"`, each instance is pinned to a different GPU
    (`main_gpu=0` / `main_gpu=1`) so two ~9GB model copies don't compete for
    one 12GB card; override by passing `main_gpu` explicitly in kwargs to
    disable this (e.g. on single-GPU hardware, construct sequentially
    instead of via this helper).
    """
    if kind in ("mock", "local", "local_qwen", "qwen"):
        if seed is None:
            if kind == "mock":
                seed = config.RANDOM_SEED
            else:
                import secrets
                seed = secrets.randbelow(2**31)
        if kind in ("local", "local_qwen", "qwen") and "main_gpu" not in kwargs:
            client_a = make_client(kind, seed=seed, main_gpu=0, **kwargs)
            client_b = make_client(kind, seed=seed, main_gpu=1, **kwargs)
            return client_a, client_b
        return make_client(kind, seed=seed, **kwargs), make_client(kind, seed=seed, **kwargs)
    # claude (or any future stateless hosted-API backend): no client-side RNG
    # stream to desynchronize between two sequential run_agent_loop calls, so
    # two independently-constructed instances are already apples-to-apples.
    return make_client(kind, **kwargs), make_client(kind, **kwargs)
