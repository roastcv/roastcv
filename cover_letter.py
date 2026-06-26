"""
Config loader — supports 6 LLM providers with automatic fallback chain.

Providers (switched via LLM_PROVIDER in .env):
  - "gemini"    -> Google Gemini / AI Studio
  - "groq"      -> Groq (llama-3.3-70b etc.)
  - "cerebras"  -> Cerebras (llama3.1-70b etc.)
  - "mistral"   -> Mistral AI
  - "github"    -> GitHub Models
  - "azure"     -> Azure OpenAI / Azure AI Foundry

FALLBACK CHAIN (auto-switch when credits run out):
  Set FALLBACK_PROVIDERS=groq,cerebras,mistral,github in .env
  If primary provider fails, it tries each fallback in order.
"""

import os
from dotenv import load_dotenv

load_dotenv()

_PROVIDER_ALIASES = {
    "foundry": "azure",
    "azure-foundry": "azure",
    "azure_foundry": "azure",
    "google": "gemini",
}

_raw_provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
LLM_PROVIDER = _PROVIDER_ALIASES.get(_raw_provider, _raw_provider)

# ── Fallback chain ────────────────────────────────────────────
# Comma-separated list of providers to try if primary fails
# Example: FALLBACK_PROVIDERS=groq,cerebras,mistral,github
_raw_fallbacks = os.getenv("FALLBACK_PROVIDERS", "").strip()
FALLBACK_PROVIDERS = [
    _PROVIDER_ALIASES.get(p.strip().lower(), p.strip().lower())
    for p in _raw_fallbacks.split(",")
    if p.strip()
]

# ── Gemini ────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# ── Groq ─────────────────────────────────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
GROQ_MODEL    = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_ENDPOINT = os.getenv("GROQ_ENDPOINT", "https://api.groq.com/openai/v1")

# ── Cerebras ──────────────────────────────────────────────────
CEREBRAS_API_KEY  = os.getenv("CEREBRAS_API_KEY")
CEREBRAS_MODEL    = os.getenv("CEREBRAS_MODEL", "llama-3.3-70b")
CEREBRAS_ENDPOINT = os.getenv("CEREBRAS_ENDPOINT", "https://api.cerebras.ai/v1")

# ── Mistral ───────────────────────────────────────────────────
MISTRAL_API_KEY  = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL    = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
MISTRAL_ENDPOINT = os.getenv("MISTRAL_ENDPOINT", "https://api.mistral.ai/v1")

# ── GitHub Models ─────────────────────────────────────────────
GITHUB_TOKEN    = os.getenv("GITHUB_TOKEN")
GITHUB_MODEL    = os.getenv("GITHUB_MODEL", "gpt-4.1-mini")
GITHUB_ENDPOINT = os.getenv("GITHUB_ENDPOINT", "https://models.inference.ai.azure.com")

# ── Azure OpenAI ──────────────────────────────────────────────
AZURE_ENDPOINT    = os.getenv("AZURE_ENDPOINT")
AZURE_API_KEY     = os.getenv("AZURE_API_KEY")
AZURE_DEPLOYMENT  = os.getenv("AZURE_DEPLOYMENT", "gpt-4.1-mini")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-10-21")

MAX_TOKENS = int(os.getenv("MAX_TOKENS", "8192"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")

SUPPORTED_PROVIDERS = ("gemini", "groq", "cerebras", "mistral", "github", "azure")


def _require(value, var_hint: str, provider_label: str):
    if not value:
        raise EnvironmentError(
            f"LLM_PROVIDER is set to '{provider_label}' but {var_hint} is missing. "
            f"Set it in your .env file."
        )


def validate_provider(provider: str):
    """Validate that required env vars exist for a given provider."""
    if provider == "gemini":
        _require(GEMINI_API_KEY, "GEMINI_API_KEY or GOOGLE_API_KEY", "gemini")
    elif provider == "groq":
        _require(GROQ_API_KEY, "GROQ_API_KEY", "groq")
    elif provider == "cerebras":
        _require(CEREBRAS_API_KEY, "CEREBRAS_API_KEY", "cerebras")
    elif provider == "mistral":
        _require(MISTRAL_API_KEY, "MISTRAL_API_KEY", "mistral")
    elif provider == "github":
        _require(GITHUB_TOKEN, "GITHUB_TOKEN", "github")
    elif provider == "azure":
        _require(AZURE_ENDPOINT, "AZURE_ENDPOINT", "azure")
        _require(AZURE_API_KEY, "AZURE_API_KEY", "azure")
    elif provider not in SUPPORTED_PROVIDERS:
        raise EnvironmentError(
            f"Unknown LLM_PROVIDER '{provider}'. "
            f"Supported: {', '.join(SUPPORTED_PROVIDERS)}"
        )


# Validate primary provider on startup
validate_provider(LLM_PROVIDER)
