"""plots.py — charts for the report (accuracy, and accuracy vs. cost)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

SINGLE = "#9aa5b1"   # grey for single-LLM baselines
MULTI = "#2a6fdb"    # blue for multi-LLM methods


def _colors(methods):
    return [SINGLE if m.startswith("single") else MULTI for m in methods]


def make_plots(df, tag):
    out = config.RESULTS_DIR

    # 1) Accuracy bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(df["method"], df["accuracy_%"], color=_colors(df["method"]))
    ax.bar_label(bars, fmt="%.0f%%", padding=3)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Single LLM vs. multi-LLM methods")
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=25, ha="right")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=SINGLE),
                       plt.Rectangle((0, 0), 1, 1, color=MULTI)],
              labels=["Single LLM", "Multiple LLMs"], frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    plt.tight_layout()
    plt.savefig(f"{out}/accuracy_{tag}.png", dpi=200)
    plt.close()

    # 2) Accuracy vs. cost (LLM calls per question)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(df["llm_calls_per_q"], df["accuracy_%"], s=80, color=_colors(df["method"]))
    for _, r in df.iterrows():
        ax.annotate(r["method"], (r["llm_calls_per_q"], r["accuracy_%"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("LLM calls per question (cost)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs. cost")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"{out}/accuracy_vs_cost_{tag}.png", dpi=200)
    plt.close()
