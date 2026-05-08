import json
import networkx as nx


class KnowledgeGraph:
    """知识图谱 - NetworkX 实现"""

    def __init__(self):
        self._graph = nx.DiGraph()

    def add_entity(self, name: str, entity_type: str, properties: dict | None = None) -> None:
        self._graph.add_node(name, type=entity_type, **(properties or {}))

    def add_relation(self, source: str, target: str, relation_type: str, properties: dict | None = None) -> None:
        self._graph.add_edge(source, target, relation=relation_type, **(properties or {}))

    def has_entity(self, name: str) -> bool:
        return self._graph.has_node(name)

    def get_entity(self, name: str) -> dict | None:
        if not self.has_entity(name):
            return None
        return {"name": name, **self._graph.nodes[name]}

    def get_neighbors(self, name: str, relation: str | None = None) -> list[tuple[str, dict]]:
        if not self.has_entity(name):
            return []
        neighbors = []
        for _, target, data in self._graph.out_edges(name, data=True):
            if relation is None or data.get("relation") == relation:
                neighbors.append((target, data))
        return neighbors

    def get_entities_by_type(self, entity_type: str) -> list[dict]:
        return [
            {"name": node, **data}
            for node, data in self._graph.nodes(data=True)
            if data.get("type") == entity_type
        ]

    def find_path(self, source: str, target: str) -> list[str]:
        try:
            return nx.shortest_path(self._graph, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def save(self, path: str) -> None:
        data = {
            "nodes": [{"name": n, **a} for n, a in self._graph.nodes(data=True)],
            "edges": [{"source": u, "target": v, **a} for u, v, a in self._graph.edges(data=True)],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._graph.clear()
        for node in data["nodes"]:
            name = node.pop("name")
            self.add_entity(name, node.pop("type", "unknown"), node)
        for edge in data["edges"]:
            self.add_relation(edge["source"], edge["target"], edge.pop("relation", "related"), edge)
