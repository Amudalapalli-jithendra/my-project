"""
llm_client.py — one small wrapper that talks to any OpenAI-compatible API
(Groq, OpenAI, OpenRouter, Ollama) and counts calls and tokens.

In "mock" mode it fakes answers so you can test everything without a key.
"""
import os
import random
import re
import time
from collections import Counter

import config


class LLMClient:
    def __init__(self):
        self.provider = config.PROVIDER
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._answer_key = {}  # used only by mock mode
        self._rng = random.Random(config.SEED)

        if self.provider != "mock":
            from openai import OpenAI
            key_env = config.SETTINGS["api_key_env"]
            api_key = os.getenv(key_env) if key_env else "ollama"
            if not api_key:
                raise RuntimeError(f"Set {key_env} in your environment or .env file.")
            self.client = OpenAI(base_url=config.SETTINGS["base_url"], api_key=api_key)

    # ------------------------------------------------------------------
    def reset_counters(self):
        self.calls = self.prompt_tokens = self.completion_tokens = 0

    def register_answers(self, questions):
        """Mock mode needs the true answers to simulate right/wrong replies."""
        self._answer_key = {item["q"]: item["a"] for item in questions}

    # ------------------------------------------------------------------
    def chat(self, model, messages, temperature=None):
        """Send a list of {"role", "content"} messages; return the reply text."""
        temperature = config.TEMPERATURE if temperature is None else temperature
        self.calls += 1

        if self.provider == "mock":
            return self._mock_reply(model, messages)

        for attempt in range(4):  # simple retry for rate limits
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=config.MAX_TOKENS,
                )
                if resp.usage:
                    self.prompt_tokens += resp.usage.prompt_tokens
                    self.completion_tokens += resp.usage.completion_tokens
                return resp.choices[0].message.content or ""
            except Exception as e:
                wait = 2 ** attempt * 3
                print(f"   [retry {attempt + 1}] {type(e).__name__}: {e} — waiting {wait}s")
                time.sleep(wait)
        return ""

    # ------------------------------------------------------------------
    # Mock model: small/medium/large get the answer right 55/70/85% of the
    # time. If the prompt already contains other agents' answers, it tends to
    # agree with the majority (this mimics how debate/aggregation behave).
    MOCK_SKILL = {"mock-small": 0.55, "mock-medium": 0.70, "mock-large": 0.85}

    def _mock_reply(self, model, messages):
        text = "\n".join(m["content"] for m in messages)
        truth = next((a for q, a in self._answer_key.items() if q in text), 0)
        skill = self.MOCK_SKILL.get(model, 0.6)

        seen = [float(x) for x in re.findall(r"Final answer:\s*(-?[\d.]+)", text)]
        if seen and self._rng.random() < 0.8:
            value = Counter(seen).most_common(1)[0][0]
            # an aggregator can still fix the group's mistake sometimes
            if value != truth and self._rng.random() < skill * 0.5:
                value = truth
        elif self._rng.random() < skill:
            value = truth
        else:
            value = truth + self._rng.choice([-10, -3, -2, -1, 1, 2, 5, 10])

        self.prompt_tokens += len(text.split())
        self.completion_tokens += 40
        value = int(value) if float(value).is_integer() else value
        return f"(mock reasoning by {model})\nFinal answer: {value}"
