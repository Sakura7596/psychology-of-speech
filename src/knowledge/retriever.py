"""统一检索接口 - 融合向量、图谱、案例三种检索"""

import logging

from src.knowledge.vector_store import VectorStore
from src.knowledge.knowledge_graph import KnowledgeGraph
from src.knowledge.case_library import CaseLibrary

logger = logging.getLogger(__name__)


class KnowledgeRetriever:
    """统一检索接口 - 融合向量、图谱、案例三种检索"""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
        case_library: CaseLibrary | None = None,
    ):
        self._vector_store = vector_store
        self._knowledge_graph = knowledge_graph
        self._case_library = case_library

    def _extract_entities(self, text: str) -> list[str]:
        """从文本中提取可能的实体名称（匹配知识图谱中的节点）"""
        if not self._knowledge_graph:
            return []
        graph = self._knowledge_graph._graph
        # 按节点名称长度降序排列，优先匹配长实体
        nodes = sorted(graph.nodes, key=len, reverse=True)
        return [node for node in nodes if node in text]

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        include_vector: bool = True,
        include_graph: bool = True,
        include_cases: bool = True,
    ) -> dict:
        """融合检索：同时查向量库、知识图谱、案例库"""
        results = {"vector_results": [], "graph_results": [], "case_results": []}

        if include_vector and self._vector_store:
            try:
                results["vector_results"] = self._vector_store.query(
                    query, n_results=n_results
                )
            except Exception as e:
                logger.warning(f"向量检索失败: {e}")

        if include_graph and self._knowledge_graph:
            try:
                entities = self._extract_entities(query)
                seen = set()
                for entity in entities:
                    neighbors = self._knowledge_graph.get_neighbors(entity)
                    for target, data in neighbors:
                        key = (entity, target, data.get("relation", ""))
                        if key not in seen:
                            seen.add(key)
                            results["graph_results"].append({
                                "source": entity,
                                "entity": target,
                                "relation": data.get("relation", ""),
                            })
                # 如果没有匹配到实体，尝试直接查询
                if not entities:
                    neighbors = self._knowledge_graph.get_neighbors(query)
                    results["graph_results"] = [
                        {"entity": n[0], "relation": n[1].get("relation", "")}
                        for n in neighbors
                    ]
            except Exception as e:
                logger.warning(f"知识图谱检索失败: {e}")

        if include_cases and self._case_library:
            try:
                results["case_results"] = self._case_library.search(
                    query, limit=n_results
                )
            except Exception as e:
                logger.warning(f"案例库检索失败: {e}")

        return results

    def get_context_string(self, query: str, n_results: int = 3) -> str:
        """生成可直接喂给LLM的上下文字符串"""
        results = self.retrieve(query, n_results=n_results)

        parts = []
        if results["vector_results"]:
            parts.append("## 相关知识")
            for r in results["vector_results"][:n_results]:
                parts.append(f"- {r.get('text', '')}")

        if results["case_results"]:
            parts.append("\n## 相关案例")
            for c in results["case_results"][:n_results]:
                parts.append(
                    f"- [{c.get('subtype', '')}] {c.get('text', '')}: "
                    f"{c.get('analysis', '')}"
                )

        if results["graph_results"]:
            parts.append("\n## 相关概念")
            for g in results["graph_results"][:n_results]:
                source = g.get("source", "")
                if source:
                    parts.append(f"- {source} →[{g['relation']}]→ {g['entity']}")
                else:
                    parts.append(f"- {g['entity']} ({g['relation']})")

        return "\n".join(parts) if parts else ""
