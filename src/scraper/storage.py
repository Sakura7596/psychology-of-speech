import json
import os
import logging

logger = logging.getLogger(__name__)


class StorageManager:
    TYPE_TO_FILE = {
        "romantic_breakup": "scraped_romantic_breakup_cases.json",
        "romantic_conflict": "scraped_romantic_conflict_cases.json",
        "romantic_chase": "scraped_romantic_chase_cases.json",
        "romantic": "scraped_romantic_cases.json",
        "dating": "scraped_dating_cases.json",
        "dating_app": "scraped_dating_app_cases.json",
        "attachment": "scraped_attachment_cases.json",
        "emotional": "scraped_emotional_cases.json",
        "psychology": "scraped_psychology_cases.json",
        "speech_act": "scraped_speech_act_cases.json",
        "rhetoric": "scraped_rhetoric_cases.json",
    }

    def __init__(
        self,
        cases_dir: str = "data/cases",
        graph_path: str = "data/graph/psychology_graph.json",
    ):
        self._cases_dir = cases_dir
        self._graph_path = graph_path

    def save_case(self, case: dict, target_file: str | None = None) -> str:
        case_type = case.get("type", "unknown")
        if target_file is None:
            target_file = self.TYPE_TO_FILE.get(case_type, f"scraped_{case_type}_cases.json")

        path = os.path.join(self._cases_dir, target_file)

        existing = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)

        existing_ids = {c.get("id", "") for c in existing}
        case_id = case.get("id") or self._generate_id(case_type, existing_ids)
        case["id"] = case_id

        clean_case = {k: v for k, v in case.items() if not k.startswith("_")}
        existing.append(clean_case)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved case {case_id} to {target_file}")
        return case_id

    def save_cases_batch(self, cases: list[dict]) -> list[str]:
        ids = []
        for case in cases:
            case_id = self.save_case(case)
            ids.append(case_id)
        return ids

    def update_graph(self, case: dict) -> None:
        from src.knowledge.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph()
        if os.path.exists(self._graph_path):
            kg.load(self._graph_path)

        changed = False
        for theory_name in case.get("theories", []):
            if not kg.has_entity(theory_name):
                kg.add_entity(theory_name, "theory")
                changed = True

        subtype = case.get("subtype", "")
        if subtype and not kg.has_entity(subtype):
            kg.add_entity(subtype, "romantic_behavior")
            changed = True
            for theory_name in case.get("theories", []):
                kg.add_relation(subtype, theory_name, "analyzed_by")

        if changed:
            kg.save(self._graph_path)
            logger.info(f"Updated knowledge graph with entities from case")

    def _generate_id(self, case_type: str, existing_ids: set[str]) -> str:
        prefix_map = {
            "romantic_breakup": "rb",
            "dating": "dt",
            "romantic": "rc",
            "romantic_conflict": "rct",
            "romantic_chase": "rch",
            "attachment": "at",
            "emotional": "em",
            "psychology": "ps",
            "dating_app": "da",
            "speech_act": "sa",
            "rhetoric": "rh",
        }
        prefix = prefix_map.get(case_type, "sc")
        counter = 1
        while f"{prefix}_{counter:03d}" in existing_ids:
            counter += 1
        return f"{prefix}_{counter:03d}"
