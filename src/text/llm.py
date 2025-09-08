from __future__ import annotations
import json, re, requests
from typing import Optional
from src.config.defaults import LMSTUDIO_BASE, LMSTUDIO_CHAT, HTTP_TIMEOUT, USER_AGENT

HEADERS = {"Content-Type":"application/json","User-Agent":USER_AGENT}

def _extract_json(text: str) -> str:
    m = re.search(r'\{.*\}', text, re.S)
    if m:
        return m.group(0)
    m = re.search(r'\[.*\]', text, re.S)
    if m:
        return m.group(0)
    raise ValueError("No JSON block found")

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
    js = _extract_json(content)
    try:
        return json.loads(js)
    except Exception:
        js2 = re.sub(r',\s*([}\]])', r'\1', js)
        return json.loads(js2)
