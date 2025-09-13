# sr/llm/client.py — full replacement

from __future__ import annotations

import json
import logging
import re
import time
from typing import Optional, Any, List, Dict

import requests
from pydantic import BaseModel

from sr.config.defaults import (
    LMSTUDIO_BASE,
    LMSTUDIO_CHAT,
    LMSTUDIO_EMB,
    HTTP_TIMEOUT,
    USER_AGENT,
)
from sr.llm.prompts import REPAIR_SYSTEM, repair_user

# -----------------------------------------------------------------------------
# HTTP defaults
# -----------------------------------------------------------------------------

HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": USER_AGENT,
}

log = logging.getLogger("llm")

# -----------------------------------------------------------------------------
# Low-level chat call
# -----------------------------------------------------------------------------

def chat(
    messages: List[Dict[str, str]],
    *,
    temperature: float = 0.0,
    max_tokens: int = 900,
    response_format: Optional[dict] = None,
    retries: int = 2,
) -> str:
    """
    Thin wrapper for LM Studio /v1/chat/completions.
    Returns the message.content string.
    """
    url = f"{LMSTUDIO_BASE.rstrip('/')}/v1/chat/completions"
    body = {
        "model": LMSTUDIO_CHAT,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": False,
    }
    if response_format is not None:
        body["response_format"] = response_format

    backoff = 0.8
    last_err = None
    for _ in range(retries + 1):
        try:
            r = requests.post(url, headers=HEADERS, json=body, timeout=HTTP_TIMEOUT)
            r.raise_for_status()
            js = r.json()
            return js["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            time.sleep(backoff)
            backoff *= 1.7
    raise RuntimeError(f"chat call failed: {last_err}")

def _post_chat(messages: List[Dict[str, str]], *, temperature: float, max_tokens: int) -> str:
    """
    Back-compat helper used by chat_json(); delegates to chat().
    """
    return chat(messages, temperature=temperature, max_tokens=max_tokens, response_format=None)

# -----------------------------------------------------------------------------
# JSON extraction / sanitation
# -----------------------------------------------------------------------------

_BEGIN = re.compile(r"BEGIN_JSON\s*", re.I)
_END   = re.compile(r"\s*END_JSON", re.I)
FENCE_BLOCK = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.I)

def _extract_fenced_json(text: str) -> str:
    """
    Preference order:
      1) last BEGIN_JSON … END_JSON block,
      2) last ```json ... ``` fenced block (or any ``` ... ```),
      3) last {...} object,
      4) last [...] array.
    Raises if nothing JSON-like found.
    """
    # BEGIN/END blocks (support multiple; take last)
    blocks = []
    pos = 0
    while True:
        m1 = _BEGIN.search(text, pos)
        if not m1:
            break
        m2 = _END.search(text, m1.end())
        if not m2:
            break
        blocks.append(text[m1.end():m2.start()])
        pos = m2.end()
    if blocks:
        return blocks[-1].strip()

    # ```json fenced
    fences = FENCE_BLOCK.findall(text)
    if fences:
        return fences[-1].strip()

    # any ``` ... ```
    m_any_fence = re.findall(r"```([\s\S]*?)```", text)
    if m_any_fence:
        return m_any_fence[-1].strip()

    # last { ... }
    objs = list(re.finditer(r"\{[\s\S]*\}", text))
    if objs:
        return objs[-1].group(0)

    # last [ ... ]
    arrs = list(re.finditer(r"\[[\s\S]*\]", text))
    if arrs:
        return arrs[-1].group(0)

    raise ValueError("No JSON found in model output")

def _sanitize_json(s: str) -> str:
    # Normalize quotes and remove trailing commas
    s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    s = re.sub(r",\s*(\}|\])", r"\1", s)
    # strip stray code fences if any remain
    s = s.replace("```", "").strip()
    return s

def _coerce_arrays(obj: Any) -> Any:
    """
    Minimal structural repair for known list fields:
      picos.outcomes, picos.study_design, picos.languages,
      picos.synonyms_population, picos.synonyms_intervention, allowed_designs
    string -> [string], None -> []
    """
    if not isinstance(obj, dict):
        return obj
    p = obj.get("picos")
    if isinstance(p, dict):
        for key in ("outcomes", "study_design", "languages", "synonyms_population", "synonyms_intervention"):
            v = p.get(key, [])
            if v is None:
                p[key] = []
            elif isinstance(v, str):
                p[key] = [v] if v.strip() else []
            elif not isinstance(v, list):
                p[key] = [v]
    if "allowed_designs" in obj:
        v = obj.get("allowed_designs")
        if v is None:
            obj["allowed_designs"] = []
        elif isinstance(v, str):
            obj["allowed_designs"] = [v] if v.strip() else []
        elif not isinstance(v, list):
            obj["allowed_designs"] = [v]
    return obj

# -----------------------------------------------------------------------------
# Public: chat_json (repair-aware)
# -----------------------------------------------------------------------------

def chat_json(
    system: str,
    user: str,
    schema_model: Optional[type[BaseModel]] = None,
    *,
    temperature: float = 0.0,
    max_tokens: int = 900,
    template_for_repair: Optional[str] = None,
) -> dict | BaseModel:
    """
    Robust JSON:
      1) ask LM (temp=0),
      2) extract fenced JSON, sanitize, parse,
      3) coerce known arrays,
      4) pydantic-validate,
      5) if validation fails AND template provided → one repair attempt via LM, then repeat 2–4.
    """
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    raw = _post_chat(messages, temperature=temperature, max_tokens=max_tokens)

    def _parse_then_validate(txt: str):
        js = _sanitize_json(_extract_fenced_json(txt))
        obj = json.loads(js)
        obj = _coerce_arrays(obj)
        if schema_model is None:
            return obj
        return schema_model.model_validate(obj)

    try:
        return _parse_then_validate(raw)
    except Exception as first_err:
        # if no template, surface the first error immediately
        if not template_for_repair:
            raise
        # one repair attempt with template
        rep_user = repair_user(template_for_repair, raw)
        repaired = _post_chat(
            [{"role": "system", "content": REPAIR_SYSTEM}, {"role": "user", "content": rep_user}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        try:
            return _parse_then_validate(repaired)
        except Exception as second_err:
            # surface the original failure with context
            raise RuntimeError(f"JSON/schema failure; first={first_err}; after-repair={second_err}")

# -----------------------------------------------------------------------------
# Embeddings
# -----------------------------------------------------------------------------

def embed_texts(texts: List[str], *, model: Optional[str] = None, retries: int = 2) -> List[List[float]]:
    """
    LM Studio /v1/embeddings wrapper.
    Returns a list of embedding vectors (list[float]) in the same order as inputs.
    """
    url = f"{LMSTUDIO_BASE.rstrip('/')}/v1/embeddings"
    body = {"model": model or LMSTUDIO_EMB, "input": texts}
    backoff = 0.8
    last_err = None
    for _ in range(retries + 1):
        try:
            r = requests.post(url, headers=HEADERS, json=body, timeout=HTTP_TIMEOUT)
            r.raise_for_status()
            data = r.json()["data"]
            return [d["embedding"] for d in data]
        except Exception as e:
            last_err = e
            time.sleep(backoff)
            backoff *= 1.7
    raise RuntimeError(f"embedding call failed: {last_err}")
