"""统一检索接口 - 融合向量、图谱、案例三种检索"""

from src.knowledge.vector_store import VectorStore
from src.knowledge.knowledge_graph import KnowledgeGraph
from src.knowledge.case_library import CaseLibrary


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
            except Exception:
                pass

        if include_graph and self._knowledge_graph:
            try:
                neighbors = self._knowledge_graph.get_neighbors(query)
                results["graph_results"] = [
                    {"entity": n[0], "relation": n[1].get("relation", "")}
                    for n in neighbors
                ]
            except Exception:
                pass

        if include_cases and self._case_library:
            try:
                results["case_results"] = self._case_library.search(
                    query, limit=n_results
                )
            except Exception:
                pass

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
                parts.append(f"- {g['entity']} ({g['relation']})")

        return "\n".join(parts) if parts else ""
