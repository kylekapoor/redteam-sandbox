"""One rate-limited, OpenAI-compatible client for every model in the harness.

Groq's free tier is generous on requests and tight on tokens per minute, and a
genetic algorithm is exactly the workload that trips it: hundreds of small calls
in bursts. Everything funnels through `complete` so backoff and accounting
happen in one place.
"""
from __future__ import annotations

import os
import random
import threading
import time
from dataclasses import dataclass, field

from openai import OpenAI

# Attacker needs to be creative, target should be cheap because it is called
# most, and the two defence models are purpose-built safety classifiers that
# Groq hosts for free.
ATTACKER_MODEL = "llama-3.3-70b-versatile"
TARGET_MODEL = "llama-3.1-8b-instant"
GUARD_MODEL = "meta-llama/llama-prompt-guard-2-86m"
JUDGE_MODEL = "openai/gpt-oss-safeguard-20b"


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retries: int = 0
    failures: int = 0
    seconds: float = 0.0
    by_model: dict = field(default_factory=dict)

    def record(self, model, prompt_tokens, completion_tokens, elapsed):
        self.calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.seconds += elapsed
        slot = self.by_model.setdefault(model, {"calls": 0, "tokens": 0, "seconds": 0.0})
        slot["calls"] += 1
        slot["tokens"] += prompt_tokens + completion_tokens
        slot["seconds"] += elapsed

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "retries": self.retries,
            "failures": self.failures,
            "seconds": round(self.seconds, 1),
            "by_model": self.by_model,
        }


USAGE = Usage()
_lock = threading.Lock()
_last_call = [0.0]
MIN_GAP = 0.35  # seconds between any two requests


def client() -> OpenAI:
    """Groq by default; point OPENAI_BASE_URL at Ollama to run this offline."""
    return OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"),
        api_key=os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY") or "ollama",
        timeout=60.0,
    )


_shared = None


def shared() -> OpenAI:
    global _shared
    if _shared is None:
        _shared = client()
    return _shared


def complete(
    messages,
    model: str = TARGET_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 400,
    retries: int = 4,
) -> str:
    """A completion, or an empty string. Never raises.

    A red-team harness that crashes on a rate limit halfway through generation
    six has lost the run. Failures are counted and returned as empty, which the
    scorer reads as "no leak" -- the conservative direction.
    """
    llm = shared()
    for attempt in range(retries):
        with _lock:
            gap = MIN_GAP - (time.time() - _last_call[0])
            if gap > 0:
                time.sleep(gap)
            _last_call[0] = time.time()

        started = time.time()
        try:
            response = llm.chat.completions.create(
                model=model, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
            )
            usage = response.usage
            USAGE.record(
                model,
                getattr(usage, "prompt_tokens", 0),
                getattr(usage, "completion_tokens", 0),
                time.time() - started,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            text = str(exc)
            if "rate_limit" in text or "429" in text or "503" in text:
                USAGE.retries += 1
                # Jitter matters: a synchronised population all retrying at the
                # same instant just re-triggers the limit.
                time.sleep(min(20.0, 3.0 * (attempt + 1)) + random.uniform(0, 1.5))
                continue
            USAGE.failures += 1
            return ""
    USAGE.failures += 1
    return ""
