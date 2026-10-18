# Beyond a Single LLM

Compares one LLM answering alone against methods that use several LLMs together,
on the same set of math word problems. It measures accuracy and cost (LLM calls,
tokens, time).

## Methods

| Method | Type | Idea | Based on |
|---|---|---|---|
| `single_cheap` | baseline | one small model, one call | — |
| `single_strong` | baseline | one large model, one call | — |
| `self_consistency` | multi-sample | same model sampled 5×, majority vote | Wang et al., 2022 |
| `multi_model_vote` | ensemble | 3 different models, majority vote | LLM ensembles |
| `debate` | collaboration | agents read each other's answers and revise | Du et al., 2023 |
| `mixture_of_agents` | aggregation | layers of models refine, a strong model synthesizes | Wang et al., 2024 |
| `cascade_router` | routing | cheap model first, escalate to strong if unsure | FrugalGPT, Chen et al., 2023 |

## Setup

```bash
pip install -r requirements.txt
python run_experiment.py --provider mock     # test without any API key
```

For real results, get a free key at https://console.groq.com, then:

```bash
cp .env.example .env          # put your GROQ_API_KEY in .env
python run_experiment.py --provider groq
```

Other options: `openai`, `openrouter`, or `ollama` (free, local — install from
https://ollama.com and `ollama pull llama3.2 qwen2.5 mistral`).

**Mock mode uses fake models**, so its numbers only prove the code runs.
Do not report mock numbers as results.

## Output (in `results/`)

- `summary_<provider>.csv` — accuracy, calls, tokens, time per method (your main table)
- `details_<provider>.csv` — every question, prediction, and raw model output
- `accuracy_<provider>.png`, `accuracy_vs_cost_<provider>.png` — charts for the report

## Things you can change

- **Models** — `config.py` → `PROVIDERS` → `models`, `cheap_model`, `strong_model`
- **Number of samples / rounds / layers** — `VOTE_SAMPLES`, `DEBATE_ROUNDS`, `MOA_LAYERS`
- **Which methods run** — `METHODS` list in `config.py`
- **Questions** — edit `QUESTIONS` in `data.py`, or set `USE_GSM8K = True` for the real GSM8K benchmark
- **Add your own method** — write `def my_method(client, question)` in `methods.py`
  and add it to `METHOD_REGISTRY` and `config.METHODS`

## Files

```
config.py          settings (provider, models, experiment parameters)
data.py            benchmark questions
llm_client.py      API wrapper + call/token counter + mock mode
methods.py         all baselines and multi-LLM methods
run_experiment.py  main script
plots.py           charts
```
