"""
methods.py — the single-LLM baselines and the multi-LLM methods.

Every method has the same signature:
    method(client, question) -> final answer text
so it is easy to add your own: write a function and add it to METHOD_REGISTRY.
"""
import re
from collections import Counter

import config

SYSTEM = ("You are a careful math assistant. Solve the problem step by step, "
          "then end with a line exactly like: Final answer: <number>")


# ---------------------------------------------------------------- helpers
def extract_answer(text):
    """Pull the number after 'Final answer:' (falls back to the last number)."""
    if not text:
        return None
    m = re.findall(r"Final answer:\s*\$?\s*(-?[\d,]*\.?\d+)", text, re.IGNORECASE)
    nums = m or re.findall(r"-?\d[\d,]*\.?\d*", text)
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


def majority(answers):
    valid = [a for a in answers if a is not None]
    return Counter(valid).most_common(1)[0][0] if valid else None


def ask(client, model, question, extra=None):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": question}]
    if extra:
        messages[-1]["content"] += "\n\n" + extra
    return client.chat(model, messages)


def as_text(value):
    return f"Final answer: {value}"


# ---------------------------------------------------------------- baselines
def single_cheap(client, question):
    """Baseline 1: one small model, one call."""
    return ask(client, config.SETTINGS["cheap_model"], question)


def single_strong(client, question):
    """Baseline 2: one large model, one call."""
    return ask(client, config.SETTINGS["strong_model"], question)


# ---------------------------------------------------------------- multi-LLM methods
def self_consistency(client, question):
    """Same model sampled N times; majority vote (Wang et al., 2022)."""
    model = config.SETTINGS["cheap_model"]
    answers = [extract_answer(ask(client, model, question))
               for _ in range(config.VOTE_SAMPLES)]
    return as_text(majority(answers))


def multi_model_vote(client, question):
    """Different models each answer once; majority vote (ensemble)."""
    answers = [extract_answer(ask(client, m, question))
               for m in config.SETTINGS["models"]]
    return as_text(majority(answers))


def debate(client, question):
    """Multi-agent debate (Du et al., 2023): agents read each other and revise."""
    models = config.SETTINGS["models"]
    replies = [ask(client, m, question) for m in models]

    for _ in range(config.DEBATE_ROUNDS):
        new_replies = []
        for i, m in enumerate(models):
            others = "\n\n".join(f"Agent {j + 1} said:\n{r}"
                                 for j, r in enumerate(replies) if j != i)
            prompt = (f"Here are solutions from other agents:\n\n{others}\n\n"
                      "Using their reasoning as extra information, check your own "
                      "work and give an updated solution.")
            new_replies.append(ask(client, m, question, extra=prompt))
        replies = new_replies

    return as_text(majority([extract_answer(r) for r in replies]))


def mixture_of_agents(client, question):
    """Mixture-of-Agents (Wang et al., 2024): layers of proposers, then an aggregator."""
    models = config.SETTINGS["models"]
    replies = [ask(client, m, question) for m in models]

    for _ in range(config.MOA_LAYERS - 1):
        context = "\n\n".join(f"Response {i + 1}:\n{r}" for i, r in enumerate(replies))
        prompt = (f"Previous responses from several models:\n\n{context}\n\n"
                  "Critically evaluate them (some may be wrong) and write an "
                  "improved solution.")
        replies = [ask(client, m, question, extra=prompt) for m in models]

    context = "\n\n".join(f"Response {i + 1}:\n{r}" for i, r in enumerate(replies))
    final_prompt = (f"Responses from several models:\n\n{context}\n\n"
                    "Synthesize these into one single, correct final solution.")
    return ask(client, config.SETTINGS["strong_model"], question, extra=final_prompt)


def cascade_router(client, question):
    """Cascade (FrugalGPT-style): ask the cheap model twice; if the two answers
    agree, trust it. If not, escalate to the strong model. Saves cost."""
    cheap = config.SETTINGS["cheap_model"]
    a1 = extract_answer(ask(client, cheap, question))
    a2 = extract_answer(ask(client, cheap, question))
    if a1 is not None and a1 == a2:
        return as_text(a1)
    return ask(client, config.SETTINGS["strong_model"], question)


METHOD_REGISTRY = {
    "single_cheap": single_cheap,
    "single_strong": single_strong,
    "self_consistency": self_consistency,
    "multi_model_vote": multi_model_vote,
    "debate": debate,
    "mixture_of_agents": mixture_of_agents,
    "cascade_router": cascade_router,
}
