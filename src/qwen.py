"""Qwen Cloud client + helpers.

Chat/LLM is OpenAI-compatible:
  base_url = https://dashscope-intl.aliyuncs.com/compatible-mode/v1
  api key  = env DASHSCOPE_API_KEY (loaded from secrets/.env)
Image and video use the DashScope async task API (wired up on D1 part 2 / D3).
"""
import os
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent


def _load_env():
    """Tiny .env loader so we don't need python-dotenv."""
    env_path = ROOT / "secrets" / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip())


_load_env()

BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

# Qwen Cloud model ids
LLM_MAX = "qwen3.7-max"      # complex reasoning / coding
LLM_PLUS = "qwen3.7-plus"    # balanced
LLM_FLASH = "qwen3.6-flash"  # fast / cheap — use for dev to keep cost near zero


def get_client() -> OpenAI:
    key = os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY not set. Put it in secrets/.env as:\n"
            "  DASHSCOPE_API_KEY=sk-..."
        )
    return OpenAI(api_key=key, base_url=BASE_URL)


def chat(prompt: str, model: str = LLM_FLASH, system: str | None = None, **kw) -> str:
    """Single-turn chat completion. Returns the reply text."""
    client = get_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(model=model, messages=messages, **kw)
    return resp.choices[0].message.content
