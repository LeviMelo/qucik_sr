# src/text/llm.py
from __future__ import annotations
import json, re, requests, time
from typing import Optional, Dict, Any
from src.config.defaults import LMSTUDIO_BASE, LMSTUDIO_CHAT, HTTP_TIMEOUT, USER_AGENT

HEADERS = {"Content-Type":"application/json","User-Agent":USER_AGENT}

_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9]*\s*|\s*```\s*$", re.M)

def _strip_md_fences(s: str) -> str:
    return _FENCE_RE.sub("", s).strip()

def _extract_json_block(text: str) -> str:
    m = re.search(r"\{[\s\S]*\}", text)
    if m: return m.group(0)
    m = re.search(r"\[[\s\S]*\]", text)
    if m: return m.group(0)
    return text.strip()

def _quick_sanitize(js: str) -> str:
    s = js
    s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    s = re.sub(r",\s*(\}|\])", r"\1", s)
    s = s.replace("```", "")
    return s

def _post_chat(body: dict) -> str:
    url = f"{LMSTUDIO_BASE.rstrip('/')}/v1/chat/completions"
    r = requests.post(url, headers=HEADERS, json=body, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def _repair_json_via_llm(bad: str) -> str:
    # Last-ditch fixer: rewrite as strict JSON
    system = (
        "You are a JSON fixer. Convert the given content into STRICT, VALID JSON.\n"
        "Rules:\n"
        "- Use only double-quoted keys and values.\n"
        "- Escape internal quotes inside strings.\n"
        "- No markdown, no comments, no trailing commas.\n"
        "- Keep existing keys if present.\n"
        "Return JSON ONLY."
    )
    body = {
        "model": LMSTUDIO_CHAT,
        "messages": [{"role":"system","content":system},{"role":"user","content":bad}],
        "temperature": 0.0,
        "max_tokens": 900,
        "stream": False
    }
    content = _post_chat(body)
    content = _strip_md_fences(content)
    repaired = _extract_json_block(content)
    repaired = _quick_sanitize(repaired)
    return repaired

def _mk_schema_response_format(schema: dict) -> dict:
    # LM Studio supports JSON Schema with strict mode
    # https://lmstudio.ai/docs/guides/structured-output
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.get("name", "payload"),
            "schema": schema["schema"],
            "strict": True
        }
    }

def chat_json(
    system: str,
    user: str,
    temperature: float = 0.1,
    max_tokens: int = 700,
    schema: Optional[dict] = None,
    retries: int = 2
) -> dict:
    """
    Try 1: JSON Schema (strict) if provided.
    Try 2: response_format=json_object (plain JSON mode).
    Try 3: re-ask with stricter system "JSON ONLY".
    Try 4: string-repair via fixer LLM.
    """
    # --- Attempt 1: schema (if provided)
    if schema is not None:
        body = {
            "model": LMSTUDIO_CHAT,
            "messages": [{"role":"system","content":system},{"role":"user","content":user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "response_format": _mk_schema_response_format(schema),
        }
        try:
            content = _post_chat(body)
            raw = _quick_sanitize(_extract_json_block(_strip_md_fences(content)))
            return json.loads(raw)
        except Exception:
            # fall through to next attempts
            pass

    # --- Attempt 2: json_object mode
    body2 = {
        "model": LMSTUDIO_CHAT,
        "messages": [{"role":"system","content":system},{"role":"user","content":user}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "response_format": {"type": "json_object"}  # usually supported; if not, server ignores
    }
    try:
        content = _post_chat(body2)
        raw = _quick_sanitize(_extract_json_block(_strip_md_fences(content)))
        return json.loads(raw)
    except Exception:
        pass

    # --- Attempt 3: re-ask with an ultra-strict system prompt
    strict_system = (
        "Return STRICT JSON only. No prose. No markdown. No trailing commas. "
        "If an enum is requested, ONLY use one of the allowed values."
    )
    body3 = {
        "model": LMSTUDIO_CHAT,
        "messages": [
            {"role":"system","content":strict_system + "\n\n" + system},
            {"role":"user","content":user}
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False
    }
    try:
        content = _post_chat(body3)
        raw = _quick_sanitize(_extract_json_block(_strip_md_fences(content)))
        return json.loads(raw)
    except Exception:
        pass

    # --- Attempt 4: repair whatever we got from attempt 2/3 if any, else from the user prompt echo
    # Re-run the best-effort request once and try repair
    try:
        content = _post_chat(body2)
    except Exception:
        content = _post_chat(body3)

    raw = _quick_sanitize(_extract_json_block(_strip_md_fences(content)))
    try:
        return json.loads(raw)
    except Exception:
        repaired = _repair_json_via_llm(raw)
        return json.loads(repaired)
