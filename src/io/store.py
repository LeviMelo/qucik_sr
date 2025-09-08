from __future__ import annotations
import os, pathlib, json, csv
from typing import Any, List, Dict
from src.config.defaults import RUNS_DIR

def ensure_run_dir(out_dir: str) -> pathlib.Path:
    p = pathlib.Path(out_dir) if out_dir else pathlib.Path(RUNS_DIR) / "run"
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_json(path: pathlib.Path, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def write_tsv(path: pathlib.Path, rows: List[Dict[str,Any]], fieldnames: List[str]):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
