from __future__ import annotations
import json, re, requests
from typing import Optional
from src.config.defaults import LMSTUDIO_BASE, LMSTUDIO_CHAT, HTTP_TIMEOUT, USER_AGENT

HEADERS = {"Content-Type":"application/json","User-Agent":USER_AGENT}

_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9]*\s*|\s*```\s*$", re.M)

def _strip_md_fences(s: str) -> str:
    # remove ```json fences anywhere
    return _FENCE_RE.sub("", s).strip()

def _extract_json_block(text: str) -> str:
    # grab the first {...} block greedily; fallback to [...] if needed
    m = re.search(r"\{[\s\S]*\}", text)
    if m: return m.group(0)
    m = re.search(r"\[[\s\S]*\]", text)
    if m: return m.group(0)
    # if nothing obvious, assume whole text is meant to be JSON
    return text.strip()

def _quick_sanitize(js: str) -> str:
    s = js
    # normalize smart quotes → plain
    s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    # remove trailing commas before } or ]
    s = re.sub(r",\s*(\}|\])", r"\1", s)
    # remove stray backticks just in case
    s = s.replace("```", "")
    return s

def _repair_json_via_llm(bad: str) -> str:
    url = f"{LMSTUDIO_BASE.rstrip('/')}/v1/chat/completions"
    system = (
        "You are a JSON fixer. Convert the given content into STRICT, VALID JSON.\n"
        "Rules:\n"
        "- Use only double quotes for keys and string values.\n"
        "- Escape ALL internal double quotes inside string values (e.g., boolean query strings with phrases).\n"
        "- Do NOT include markdown fences. No comments. No trailing commas.\n"
        "- Keep the same keys/structure if present: picos, inclusion_criteria, exclusion_criteria, reason_taxonomy, boolean_queries.\n"
        "Return JSON ONLY."
    )
    user = f"BAD_JSON:\n{bad}"
    body = {
        "model": LMSTUDIO_CHAT,
        "messages": [{"role":"system","content":system},{"role":"user","content":user}],
        "temperature": 0.0,
        "max_tokens": 900,
        "stream": False
    }
    r = requests.post(url, headers=HEADERS, json=body, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    content = _strip_md_fences(content)
    repaired = _extract_json_block(content)
    repaired = _quick_sanitize(repaired)
    return repaired

def chat_json(system: str, user: str, temperature: float = 0.1, max_tokens: int = 700) -> dict:
    url = f"{LMSTUDIO_BASE.rstrip('/')}/v1/chat/completions"
    body = {
        "model": LMSTUDIO_CHAT,
        "messages": [{"role":"system","content":system},{"role":"user","content":user}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    r = requests.post(url, headers=HEADERS, json=body, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]

    # 1) strip fences, 2) extract JSON-ish block, 3) sanitize, 4) parse
    raw = _strip_md_fences(content)
    js = _extract_json_block(raw)
    js = _quick_sanitize(js)

    try:
        return json.loads(js)
    except Exception:
        # One repair attempt via LLM
        try:
            repaired = _repair_json_via_llm(js)
            return json.loads(repaired)
        except Exception as e:
            # Print a short snippet to help debug
            snippet = js[:500]
            raise ValueError(f"Could not parse LLM JSON after repair. Snippet:\n{snippet}\nError: {e}")
