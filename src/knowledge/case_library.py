import json
import os


class CaseLibrary:
    """案例库 - JSON 文件 + 关键词检索"""

    def __init__(self, data_dir: str = "data/cases"):
        self._data_dir = data_dir
        self._cases: list[dict] = []
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self._data_dir):
            return
        for filename in os.listdir(self._data_dir):
            if filename.endswith(".json"):
                path = os.path.join(self._data_dir, filename)
                with open(path, "r", encoding="utf-8") as f:
                    self._cases.extend(json.load(f))

    def search(self, query: str, limit: int = 5) -> list[dict]:
        results = []
        query_lower = query.lower()
        for case in self._cases:
            keywords = case.get("keywords", [])
            text = case.get("text", "")
            if any(kw in query_lower or query_lower in kw for kw in keywords):
                results.append(case)
            elif query_lower in text.lower():
                results.append(case)
        return results[:limit]

    def get_by_type(self, case_type: str) -> list[dict]:
        return [c for c in self._cases if c.get("type") == case_type]

    def get_by_subtype(self, subtype: str) -> list[dict]:
        return [c for c in self._cases if c.get("subtype") == subtype]

    def get_all(self) -> list[dict]:
        return self._cases

    def count(self) -> int:
        return len(self._cases)
