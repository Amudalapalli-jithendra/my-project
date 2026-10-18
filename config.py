"""
config.py — all the settings you are likely to change live here.

PROVIDER options:
  "mock"       -> no API key needed; fake models so you can test the pipeline
  "groq"       -> free tier, fast (https://console.groq.com) — recommended for students
  "openai"     -> paid (https://platform.openai.com)
  "openrouter" -> one key for many models (https://openrouter.ai)
  "ollama"     -> free, runs models on your own laptop (https://ollama.com)
"""
import os

# Load a .env file if python-dotenv is installed (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PROVIDER = os.getenv("PROVIDER", "mock")

PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        # Pick models listed on https://console.groq.com/docs/models
        "models": ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "gemma2-9b-it"],
        "cheap_model": "llama-3.1-8b-instant",
        "strong_model": "llama-3.3-70b-versatile",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
        "cheap_model": "gpt-4o-mini",
        "strong_model": "gpt-4o",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "models": ["meta-llama/llama-3.1-8b-instruct", "mistralai/mistral-7b-instruct",
                   "google/gemma-2-9b-it"],
        "cheap_model": "meta-llama/llama-3.1-8b-instruct",
        "strong_model": "meta-llama/llama-3.3-70b-instruct",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key_env": None,  # no key needed
        # Run `ollama pull llama3.2` etc. first
        "models": ["llama3.2", "qwen2.5", "mistral"],
        "cheap_model": "llama3.2",
        "strong_model": "qwen2.5",
    },
    "mock": {
        "base_url": None,
        "api_key_env": None,
        "models": ["mock-small", "mock-medium", "mock-large"],
        "cheap_model": "mock-small",
        "strong_model": "mock-large",
    },
}

SETTINGS = PROVIDERS[PROVIDER]

# ---- Experiment settings ----
TEMPERATURE = 0.7          # >0 so repeated samples differ (needed for voting)
MAX_TOKENS = 512
NUM_QUESTIONS = 20         # how many questions from data.py to use (max = all)
VOTE_SAMPLES = 5           # samples for self-consistency voting
DEBATE_ROUNDS = 2          # rounds of discussion in multi-agent debate
MOA_LAYERS = 2             # layers in Mixture-of-Agents
RESULTS_DIR = "results"
SEED = 42

# Which methods to run (comment out any you don't want)
METHODS = [
    "single_cheap",       # baseline 1: one small model
    "single_strong",      # baseline 2: one big model
    "self_consistency",   # one model, many samples, majority vote
    "multi_model_vote",   # different models, majority vote (ensemble)
    "debate",             # multi-agent debate (Du et al., 2023)
    "mixture_of_agents",  # MoA (Wang et al., 2024)
    "cascade_router",     # cheap first, escalate if unsure (FrugalGPT-style)
]
