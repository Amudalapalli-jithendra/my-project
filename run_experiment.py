"""
run_experiment.py — runs every method in config.METHODS on the questions,
then saves results and charts to the results/ folder.

Usage:
    python run_experiment.py                  # uses PROVIDER from config / .env
    python run_experiment.py --provider groq  # override provider
    python run_experiment.py --n 10           # only first 10 questions
"""
import argparse
import json
import os
import time

import pandas as pd

import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", help="mock | groq | openai | openrouter | ollama")
    parser.add_argument("--n", type=int, help="number of questions")
    args = parser.parse_args()

    if args.provider:  # override before other modules read config
        config.PROVIDER = args.provider
        config.SETTINGS = config.PROVIDERS[args.provider]

    from llm_client import LLMClient
    from methods import METHOD_REGISTRY, extract_answer
    from data import load_questions

    questions = load_questions(args.n or config.NUM_QUESTIONS)
    client = LLMClient()
    client.register_answers(questions)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print(f"Provider: {config.PROVIDER} | models: {config.SETTINGS['models']}")
    print(f"Questions: {len(questions)} | methods: {config.METHODS}\n")

    rows, summary = [], []
    for name in config.METHODS:
        method = METHOD_REGISTRY[name]
        client.reset_counters()
        correct, start = 0, time.time()
        print(f"▶ {name}")

        for i, item in enumerate(questions):
            output = method(client, item["q"])
            pred = extract_answer(output)
            ok = pred is not None and abs(pred - item["a"]) < 1e-6
            correct += ok
            rows.append({"method": name, "question_id": i, "question": item["q"],
                         "gold": item["a"], "predicted": pred, "correct": ok,
                         "raw_output": output})
            print(f"   Q{i + 1:02d}: pred={pred} gold={item['a']} {'✓' if ok else '✗'}")

        n = len(questions)
        summary.append({
            "method": name,
            "accuracy_%": round(100 * correct / n, 1),
            "llm_calls_per_q": round(client.calls / n, 2),
            "tokens_per_q": round((client.prompt_tokens + client.completion_tokens) / n),
            "seconds_per_q": round((time.time() - start) / n, 2),
        })
        print(f"   → accuracy {summary[-1]['accuracy_%']}%\n")

    detail_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(summary)
    tag = config.PROVIDER
    detail_df.to_csv(f"{config.RESULTS_DIR}/details_{tag}.csv", index=False)
    summary_df.to_csv(f"{config.RESULTS_DIR}/summary_{tag}.csv", index=False)
    with open(f"{config.RESULTS_DIR}/run_config_{tag}.json", "w") as f:
        json.dump({k: getattr(config, k) for k in
                   ["PROVIDER", "TEMPERATURE", "NUM_QUESTIONS", "VOTE_SAMPLES",
                    "DEBATE_ROUNDS", "MOA_LAYERS", "METHODS"]} | {"models": config.SETTINGS},
                  f, indent=2)

    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("=" * 70)

    from plots import make_plots
    make_plots(summary_df, tag)
    print(f"\nSaved CSVs and charts to ./{config.RESULTS_DIR}/")


if __name__ == "__main__":
    main()
