"""One OpenAI-compatible client for every model in the harness.

Everything runs locally through Ollama. This started on Groq's free tier and
moved after Groq decommissioned the attacker model with two days' notice, on top
of the per-minute and per-day token caps that already made a genetic algorithm
awkward to run: hundreds of small calls in bursts is the exact workload those
limits punish.

Local inference removes the API key, the quotas and the decommission risk, and
anyone cloning this repo can run it without an account. Set OPENAI_BASE_URL and
the model names to point back at a hosted provider if you want one.
"""
from __future__ import annotations

import os
import random
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache

from openai import OpenAI

# The attacker rewrites prompts a filter has just blocked, and most current
# models decline that request. Testing the alternatives: gpt-oss-120b answers
# "I'm sorry, but I can't help with that", gpt-oss-20b returns nothing,
# qwen3.6-27b refuses once you stop it reasoning, and groq/compound-mini refuses
# every variant. Llama 3.1 8B complied on all three rewrite cases, so it plays
# attacker, target and judge.
#
# Attacker and target sharing a model is worth knowing when reading the results.
# They share a refusal boundary, which flatters the defence.
DEFAULT_BASE_URL = "http://localhost:11434/v1"
ATTACKER_MODEL = os.getenv("ATTACKER_MODEL", "llama3.1:8b")
TARGET_MODEL = os.getenv("TARGET_MODEL", "llama3.1:8b")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama3.1:8b")


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
# Local inference has no rate limit, so nothing needs spacing out. Raise this
# if you point the client back at a hosted provider.
MIN_GAP = float(os.getenv("MIN_REQUEST_GAP", "0.0"))


@lru_cache(maxsize=1)
def client() -> OpenAI:
    """Local Ollama by default; point OPENAI_BASE_URL at a hosted provider to swap.

    Cached so every call site shares one connection pool. `lru_cache` is the
    stdlib's singleton; a module-level global plus a None check is the same
    thing written by hand.
    """
    return OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        # Ollama ignores the key and the OpenAI client insists on one.
        api_key=os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY") or "ollama",
        timeout=180.0,   # a local 8B model on CPU is slower than a hosted one
    )


def complete(
    messages,
    model: str = TARGET_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 400,
    retries: int = 4,
) -> str | None:
    """A completion, or None if the call failed. Never raises.

    Returning None rather than "" is load-bearing. An earlier version returned
    an empty string on failure, and the guardrail judge treated empty as BLOCK
    -- so every rate-limited call silently became a security decision. The
    result was a defence that blocked 100% of benign traffic while reporting a
    perfect 0% attack success rate. "The model said no" and "the model never
    answered" must never be the same value.
    """
    llm = client()
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
        except Exception as exc:  # noqa: BLE001 - any failure is just "no answer"
            text = str(exc)
            if "rate_limit" in text or "429" in text or "503" in text:
                USAGE.retries += 1
                # Jitter matters: a synchronised population all retrying at the
                # same instant just re-triggers the limit.
                time.sleep(min(20.0, 3.0 * (attempt + 1)) + random.uniform(0, 1.5))
                continue
            USAGE.failures += 1
            return None
    USAGE.failures += 1
    return None
