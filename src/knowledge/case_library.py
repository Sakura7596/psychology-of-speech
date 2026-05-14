import json
import os
import re
from collections import defaultdict


# 全局单例缓存：同一 data_dir 只加载一次
_instances: dict[str, "CaseLibrary"] = {}


class CaseLibrary:
    """案例库 - JSON 文件 + 关键词倒排索引 + 语义检索"""

    def __init__(self, data_dir: str = "data/cases"):
        self._data_dir = data_dir
        self._cases: list[dict] = []
        self._keyword_index: dict[str, list[int]] = defaultdict(list)  # keyword -> case indices
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not os.path.exists(self._data_dir):
            return
        for filename in os.listdir(self._data_dir):
            if filename.endswith(".json"):
                path = os.path.join(self._data_dir, filename)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        cases = json.load(f)
                    for case in cases:
                        idx = len(self._cases)
                        self._cases.append(case)
                        # 建倒排索引
                        for kw in case.get("keywords", []):
                            self._keyword_index[kw.lower()].append(idx)
                        for concept in case.get("key_concepts", []):
                            self._keyword_index[concept.lower()].append(idx)
                except (json.JSONDecodeError, OSError):
                    continue

    @classmethod
    def get_instance(cls, data_dir: str = "data/cases") -> "CaseLibrary":
        """获取单例实例，同一 data_dir 只加载一次"""
        if data_dir not in _instances:
            _instances[data_dir] = cls(data_dir)
        return _instances[data_dir]

    def _tokenize(self, text: str) -> set[str]:
        """简单的中文分词（基于字符和标点）"""
        tokens = set()
        chinese_words = re.findall(r'[一-鿿]+', text)
        for word in chinese_words:
            for char in word:
                tokens.add(char)
            for i in range(len(word) - 1):
                tokens.add(word[i:i+2])
            for i in range(len(word) - 2):
                tokens.add(word[i:i+3])
        return tokens

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """增强搜索：倒排索引 + 关键词匹配 + 语义相似度"""
        self._ensure_loaded()
        query_lower = query.lower()
        query_tokens = self._tokenize(query)

        # 用倒排索引快速筛选候选
        candidate_indices: set[int] = set()
        for kw in self._keyword_index:
            if kw in query_lower or query_lower in kw:
                candidate_indices.update(self._keyword_index[kw])

        # 如果倒排索引没命中，退化为全量扫描
        if not candidate_indices:
            candidate_indices = set(range(len(self._cases)))

        scored_cases = []
        for idx in candidate_indices:
            case = self._cases[idx]
            score = 0
            keywords = case.get("keywords", [])
            text = case.get("text", "")
            name = case.get("name", "")
            description = case.get("description", "")
            key_concepts = case.get("key_concepts", [])
            analysis = case.get("analysis", "")

            # 1. 关键词精确匹配（高权重）
            for kw in keywords:
                if kw in query_lower or query_lower in kw:
                    score += 3
                elif any(k in query_lower for k in kw if len(k) > 1):
                    score += 1

            # 2. 文本内容匹配
            searchable = f"{text} {name} {description} {analysis} {' '.join(key_concepts)}".lower()
            if query_lower in searchable:
                score += 2

            # 3. 语义相似度（基于 token 重叠）
            case_tokens = self._tokenize(searchable)
            if query_tokens and case_tokens:
                overlap = len(query_tokens & case_tokens)
                score += overlap * 0.5

            if score > 0:
                scored_cases.append((score, case))

        scored_cases.sort(key=lambda x: x[0], reverse=True)
        return [case for _, case in scored_cases[:limit]]

    def get_by_type(self, case_type: str) -> list[dict]:
        self._ensure_loaded()
        return [c for c in self._cases if c.get("type") == case_type]

    def get_by_subtype(self, subtype: str) -> list[dict]:
        self._ensure_loaded()
        return [c for c in self._cases if c.get("subtype") == subtype]

    def get_all(self) -> list[dict]:
        self._ensure_loaded()
        return self._cases

    def count(self) -> int:
        self._ensure_loaded()
        return len(self._cases)
