from __future__ import annotations
import json, re, requests
from typing import Optional, Dict, Any
from src.config.defaults import LMSTUDIO_BASE, LMSTUDIO_CHAT, HTTP_TIMEOUT, USER_AGENT

HEADERS = {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": USER_AGENT}
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9]*\s*|\s*```\s*$", re.M)

def _strip_md_fences(s: str) -> str:
    return _FENCE_RE.sub("", s).strip()

def _extract_json_block(text: str) -> str:
    # Prefer object; fallback to array; else raw
    m = re.search(r"\{[\s\S]*\}", text)
    if m: return m.group(0)
    m = re.search(r"\[[\s\S]*\]", text)
    if m: return m.group(0)
    return text.strip()

def _quick_sanitize(js: str) -> str:
    s = js
    # smart quotes → ascii
    s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    # trailing commas
    s = re.sub(r",\s*(\}|\])", r"\1", s)
    # strip fenced code if any sneaks in
    s = s.replace("```", "")
    return s

def _post_chat(body: dict) -> str:
    url = f"{LMSTUDIO_BASE.rstrip('/')}/v1/chat/completions"
    r = requests.post(url, headers=HEADERS, json=body, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    js = r.json()
    # LM Studio returns {"choices":[{"message":{"content": ...}}]}
    return js["choices"][0]["message"]["content"]

def _mk_schema_response_format(schema: dict) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.get("name", "payload"),
            "schema": schema["schema"],
            "strict": True
        }
    }

def _try_request(messages, temperature, max_tokens, response_format: Optional[dict]) -> str:
    body = {
        "model": LMSTUDIO_CHAT,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": False,
    }
    if response_format is not None:
        body["response_format"] = response_format
    return _post_chat(body)

def chat_json(
    system: str,
    user: str,
    temperature: float = 0.1,
    max_tokens: int = 900,
    schema: Optional[dict] = None,
) -> dict:
    """
    Robust JSON chat:
      1) If schema provided, try once WITH response_format=json_schema.
      2) If that 4xx/5xx or parse fails, retry WITHOUT response_format (plain).
      3) If still not JSON, try to extract and sanitize braces.
      4) If all fails, raise a clear Exception.
    """
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    # Attempt A: schema format (some LM Studio builds support this)
    if schema is not None:
        try:
            content = _try_request(messages, temperature, max_tokens, response_format=_mk_schema_response_format(schema))
            raw = _quick_sanitize(_extract_json_block(_strip_md_fences(content)))
            return json.loads(raw)
        except Exception:
            # fallthrough to plain mode
            pass

    # Attempt B: plain request (no response_format at all)
    try:
        content = _try_request(messages, temperature, max_tokens, response_format=None)
        raw = _quick_sanitize(_extract_json_block(_strip_md_fences(content)))
        return json.loads(raw)
    except Exception as e:
        # We tried; produce an actionable error with the best we can show
        raise RuntimeError(f"chat_json failed to parse JSON from LM Studio response: {e}")
