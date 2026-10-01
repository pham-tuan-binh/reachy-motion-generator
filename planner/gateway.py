"""Vercel AI Gateway client (OpenAI-compatible Chat Completions), used to call the teacher LLM.

Set ``AI_GATEWAY_API_KEY`` (or run on Vercel with an OIDC token in ``VERCEL_OIDC_TOKEN``).
Models are addressed as ``provider/model``, e.g. ``anthropic/claude-opus-5.5``.
"""
import json
import os
import re
import time

from openai import OpenAI, BadRequestError

BASE_URL = os.environ.get("AI_GATEWAY_BASE_URL", "https://ai-gateway.vercel.sh/v1")
DEFAULT_MODEL = os.environ.get("PLANNER_MODEL", "anthropic/claude-opus-5.5")


def client():
    key = os.environ.get("AI_GATEWAY_API_KEY") or os.environ.get("VERCEL_OIDC_TOKEN")
    if not key:
        raise SystemExit("Set AI_GATEWAY_API_KEY (Vercel dashboard -> AI Gateway -> API keys).")
    return OpenAI(api_key=key, base_url=BASE_URL)


def _parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON object in model output: {text[:200]!r}")
    return json.loads(m.group(0))


def chat_json(messages, schema, name, model=None, temperature=0.7, retries=3):
    """Chat completion constrained to a JSON schema (structured outputs); falls back to
    'reply with JSON' parsing for models that do not support ``response_format``."""
    c = client(); model = model or DEFAULT_MODEL; structured = True
    for attempt in range(retries):
        try:
            kw = dict(model=model, messages=messages, temperature=temperature)
            if structured:
                kw["response_format"] = {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}}
            r = c.chat.completions.create(**kw)
            return _parse_json(r.choices[0].message.content)
        except BadRequestError:
            if not structured:
                raise
            structured = False   # model/provider without structured outputs: retry as plain JSON
        except (ValueError, json.JSONDecodeError):
            if attempt == retries - 1:
                raise
        time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")

