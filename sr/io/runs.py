# sr/io/runs.py
from __future__ import annotations
import pathlib, json, time
from sr.config.defaults import RUNS_DIR

class Runs:
    def __init__(self, out_dir: str = ""):
        self.root = pathlib.Path(out_dir) if out_dir else pathlib.Path(RUNS_DIR) / f"run_{int(time.time())}"
        self.root.mkdir(parents=True, exist_ok=True)
    def path(self, name: str) -> pathlib.Path:
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    def save_json(self, name: str, obj):
        p = self.path(name)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
