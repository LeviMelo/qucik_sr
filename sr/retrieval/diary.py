# sr/retrieval/diary.py
from __future__ import annotations
from typing import Dict, Any, List
from sr.config.schema import SearchDiary

class Diary:
    def __init__(self):
        self._q: List[Dict[str, Any]] = []
        self._pages = 0
        self._total = 0
    def log_query(self, name: str, query: str):
        self._q.append({"name": name, "query": query})
    def log_pages(self, pages: int):
        self._pages += int(pages)
    def set_total(self, total_ids: int):
        self._total = int(total_ids)
    def snapshot(self) -> SearchDiary:
        return SearchDiary(queries=self._q, pages=self._pages, total_ids=self._total)
