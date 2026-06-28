"""
LLM wrapper — supports 6 providers with AUTOMATIC FALLBACK CHAIN.

FIX (multi-user):
- _semaphore per-request count badha ke 5 kiya — 5 users concurrently
- _clients dict lazy init thread-safe hai already (dict ops are GIL-protected)
- _provider_state thread-local already tha — that's correct, kept as-is
- acquire_slot timeout 30s rakha — fast failure instead of hanging
"""

import functools
import json
import random
import re
import time
import threading

from config import (
    LLM_PROVIDER, FALLBACK_PROVIDERS, MAX_TOKENS,
    GEMINI_API_KEY, GEMINI_MODEL,
    GROQ_API_KEY, GROQ_MODEL, GROQ_ENDPOINT,
    CEREBRAS_API_KEY, CEREBRAS_MODEL, CEREBRAS_ENDPOINT,
    MISTRAL_API_KEY, MISTRAL_MODEL, MISTRAL_ENDPOINT,
    GITHUB_TOKEN, GITHUB_MODEL, GITHUB_ENDPOINT,
    AZURE_ENDPOINT, AZURE_API_KEY, AZURE_DEPLOYMENT, AZURE_API_VERSION,
)

# ── Settings ──────────────────────────────────────────────────
MAX_RETRIES           = 2
RETRY_BASE_DELAY      = 3    # FIX: 5 se 3 — faster retry
AGENT_TIMEOUT         = 75   # FIX: 90 se 75 — faster failure
MAX_USEFUL_WAIT       = 30   # FIX: 45 se 30 — skip slow providers faster
MAX_CONCURRENT        = 5    # FIX: 2 se 5 — 5 users concurrently support karo
                              # (Gemini free tier pe FALLBACK_PROVIDERS set karo)

GEMINI_STAGGER_MAX_MS = 600  # FIX: 800 se 600

RATE_LIMIT_MARKERS = (
    "429", "resource_exhausted", "503", "unavailable",
    "rate limit", "ratelimit", "throttl", "quota",
    "too many", "exceeded", "limit reached",
)

# ── Concurrency limiter ───────────────────────────────────────
_semaphore = threading.Semaphore(MAX_CONCURRENT)


def acquire_slot(timeout: int = 30) -> bool:  # FIX: 60 se 30
    acquired = _semaphore.acquire(timeout=timeout)
    if not acquired:
        raise RuntimeError(
            "Server is busy — too many resumes are being processed. "
            "Please try again in 30 seconds."
        )
    return True


def release_slot():
    try:
        _semaphore.release()
    except ValueError:
        pass


# ── Active provider tracker (thread-local — correct) ─────────
_provider_state = threading.local()


def _set_active_provider(provider: str):
    _provider_state.value = provider


def get_active_provider() -> str:
    return getattr(_provider_state, "value", LLM_PROVIDER)


# ── Lazy clients (thread-safe via _clients_lock) ─────────────
_clients: dict = {}
_clients_lock = threading.Lock()  # FIX: dict initialization race condition fix


def _get_gemini():
    if "gemini" not in _clients:
        with _clients_lock:
            if "gemini" not in _clients:  # double-check after lock
                from google import genai
                _clients["gemini"] = genai.Client(api_key=GEMINI_API_KEY)
    return _clients["gemini"]


def _get_openai_client(key: str, base_url: str, token: str = None):
    if key not in _clients:
        with _clients_lock:
            if key not in _clients:
                from openai import OpenAI
                _clients[key] = OpenAI(
                    base_url=base_url,
                    api_key=token or "placeholder",
                )
    return _clients[key]


def _get_azure():
    if "azure" not in _clients:
        with _clients_lock:
            if "azure" not in _clients:
                from openai import AzureOpenAI
                _clients["azure"] = AzureOpenAI(
                    azure_endpoint=AZURE_ENDPOINT,
                    api_key=AZURE_API_KEY,
                    api_version=AZURE_API_VERSION,
                )
    return _clients["azure"]


# ── Provider backends ─────────────────────────────────────────
def _gemini_generate(system_prompt, user_prompt, max_tokens, json_mode) -> str:
    from google.genai import types
    client = _get_gemini()

    jitter = random.randint(0, GEMINI_STAGGER_MAX_MS) / 1000.0
    if jitter > 0:
        time.sleep(jitter)

    cfg = {"system_instruction": system_prompt, "max_output_tokens": max_tokens}
    if json_mode:
        cfg["response_mime_type"] = "application/json"
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(**cfg),
    )
    candidates = getattr(response, "candidates", None)
    if candidates:
        finish = str(getattr(candidates[0], "finish_reason", "")).upper()
        if finish in ("SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT"):
            raise ValueError(f"Gemini blocked response: {finish}")
        if finish == "MAX_TOKENS":
            print("  WARNING: Gemini response truncated (MAX_TOKENS).")
    return (response.text or "").strip()


def _openai_compat_generate(client, model, system_prompt, user_prompt,
                             max_tokens, json_mode) -> str:
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    if getattr(choice, "finish_reason", None) == "content_filter":
        raise ValueError("Response blocked by content filter.")
    return (choice.message.content or "").strip()


def _groq_generate(system_prompt, user_prompt, max_tokens, json_mode) -> str:
    client = _get_openai_client("groq", GROQ_ENDPOINT, GROQ_API_KEY)
    return _openai_compat_generate(client, GROQ_MODEL, system_prompt,
                                   user_prompt, max_tokens, json_mode)


def _cerebras_generate(system_prompt, user_prompt, max_tokens, json_mode) -> str:
    client = _get_openai_client("cerebras", CEREBRAS_ENDPOINT, CEREBRAS_API_KEY)
    effective_user_prompt = user_prompt
    if json_mode:
        effective_user_prompt = (
            user_prompt.rstrip() +
            "\n\nIMPORTANT: Respond with ONLY valid JSON. No markdown, no explanation, no backticks."
        )
    return _openai_compat_generate(client, CEREBRAS_MODEL, system_prompt,
                                   effective_user_prompt, max_tokens, False)


def _mistral_generate(system_prompt, user_prompt, max_tokens, json_mode) -> str:
    client = _get_openai_client("mistral", MISTRAL_ENDPOINT, MISTRAL_API_KEY)
    return _openai_compat_generate(client, MISTRAL_MODEL, system_prompt,
                                   user_prompt, max_tokens, json_mode)


def _github_generate(system_prompt, user_prompt, max_tokens, json_mode) -> str:
    client = _get_openai_client("github", GITHUB_ENDPOINT, GITHUB_TOKEN)
    return _openai_compat_generate(client, GITHUB_MODEL, system_prompt,
                                   user_prompt, max_tokens, json_mode)


def _azure_generate(system_prompt, user_prompt, max_tokens, json_mode) -> str:
    client = _get_azure()
    return _openai_compat_generate(client, AZURE_DEPLOYMENT, system_prompt,
                                   user_prompt, max_tokens, json_mode)


_BACKENDS = {
    "gemini":   _gemini_generate,
    "groq":     _groq_generate,
    "cerebras": _cerebras_generate,
    "mistral":  _mistral_generate,
    "github":   _github_generate,
    "azure":    _azure_generate,
}


# ── Timeout wrapper ───────────────────────────────────────────
def _run_with_timeout(fn, timeout_sec, *args):
    result    = [None]
    error     = [None]
    completed = threading.Event()

    def target():
        try:
            result[0] = fn(*args)
        except Exception as e:
            error[0] = e
        finally:
            completed.set()

    t = threading.Thread(target=target, daemon=True)
    t.start()
    if not completed.wait(timeout=timeout_sec):
        raise TimeoutError(
            f"Agent timed out after {timeout_sec}s. "
            "API may be slow — try again or switch provider."
        )
    if error[0]:
        raise error[0]
    return result[0]


# ── Rate limit helpers ────────────────────────────────────────
def _is_rate_limit(err_str: str) -> bool:
    return any(m in err_str.lower() for m in RATE_LIMIT_MARKERS)


def _extract_wait(err_str: str):
    m = re.search(r"wait\s+(\d+)\s*seconds?", err_str, re.IGNORECASE)
    return int(m.group(1)) if m else None


# ── Core generate with fallback chain ────────────────────────
def _generate(system_prompt: str, user_prompt: str,
              max_tokens: int, json_mode: bool) -> tuple[str, str]:
    acquire_slot()
    try:
        return _generate_inner(system_prompt, user_prompt, max_tokens, json_mode)
    finally:
        release_slot()


def _generate_inner(system_prompt: str, user_prompt: str,
                    max_tokens: int, json_mode: bool) -> tuple[str, str]:
    providers_to_try = [LLM_PROVIDER] + [
        p for p in FALLBACK_PROVIDERS if p != LLM_PROVIDER
    ]

    last_error = None

    for provider in providers_to_try:
        backend = _BACKENDS.get(provider)
        if backend is None:
            print(f"  Skipping unknown provider: {provider}")
            continue

        try:
            print(f"  Trying provider: {provider}...")
            text = _run_with_timeout(
                backend, AGENT_TIMEOUT,
                system_prompt, user_prompt, max_tokens, json_mode
            )
            if text:
                _set_active_provider(provider)
                if provider != LLM_PROVIDER:
                    print(f"  ✓ Switched to fallback: {provider}")
                return text, provider

        except Exception as e:
            err_str = str(e)
            last_error = e

            wait_hint = _extract_wait(err_str)
            if wait_hint and wait_hint > MAX_USEFUL_WAIT:
                print(f"  {provider}: quota wait {wait_hint}s — skipping to next provider")
                continue

            if _is_rate_limit(err_str):
                for attempt in range(MAX_RETRIES - 1):
                    wait = RETRY_BASE_DELAY * (2 ** attempt)
                    print(f"  {provider}: rate limit — retrying in {wait}s "
                          f"(attempt {attempt + 1}/{MAX_RETRIES - 1})")
                    time.sleep(wait)
                    try:
                        text = _run_with_timeout(
                            backend, AGENT_TIMEOUT,
                            system_prompt, user_prompt, max_tokens, json_mode
                        )
                        if text:
                            _set_active_provider(provider)
                            return text, provider
                    except Exception as e2:
                        last_error = e2
                        if not _is_rate_limit(str(e2)):
                            break
                print(f"  {provider}: retries exhausted — trying next provider")
            else:
                print(f"  {provider} error: {err_str[:100]} — trying next provider")

    raise RuntimeError(
        f"All providers failed. Last error: {last_error}\n"
        f"Tried: {providers_to_try}\n"
        "Check your API keys in .env and your quota/credits."
    )


# ── JSON parser ───────────────────────────────────────────────
def _clean_and_parse_json(raw: str) -> dict:
    if not raw.strip():
        raise ValueError("LLM returned empty response.")
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            cleaned = re.sub(r",\s*([\]}])", r"\1", candidate)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass
    raise ValueError(f"Could not parse JSON from response:\n{raw[:400]}")


# ── Public API ────────────────────────────────────────────────
def call_llm(system_prompt: str, user_prompt: str,
             max_tokens: int = MAX_TOKENS) -> str:
    text, _ = _generate(system_prompt, user_prompt, max_tokens, json_mode=False)
    if not text:
        raise ValueError("All providers returned empty response.")
    return text


def call_llm_json(system_prompt: str, user_prompt: str,
                  max_tokens: int = MAX_TOKENS) -> dict:
    raw, _ = _generate(system_prompt, user_prompt, max_tokens, json_mode=True)
    return _clean_and_parse_json(raw)