from dataclasses import dataclass


class SyntaxAnalyzer:
    """句法分析器 - HanLP 依存句法 + 语义角色标注"""

    def __init__(self):
        self._dep_parser = None
        self._srl = None
        self._tokenizer = None

    def _get_tokenizer(self):
        if self._tokenizer is None:
            import hanlp
            self._tokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
        return self._tokenizer

    def _get_dep_parser(self):
        if self._dep_parser is None:
            import hanlp
            self._dep_parser = hanlp.load(hanlp.pretrained.dep.CTB9_DEP_ELECTRA_SMALL)
        return self._dep_parser

    def _get_srl(self):
        if self._srl is None:
            import hanlp
            self._srl = hanlp.load(hanlp.pretrained.srl.SRL_ELECTRA_SMALL_ZH)
        return self._srl

    def parse_dependencies(self, text: str) -> list[dict]:
        """依存句法分析"""
        tokenizer = self._get_tokenizer()
        tokens = tokenizer(text)
        parser = self._get_dep_parser()
        arcs = parser(tokens)

        result = []
        for i, (token, arc) in enumerate(zip(tokens, arcs)):
            if isinstance(arc, dict):
                result.append({
                    "id": i,
                    "word": token,
                    "head": arc.get("head", 0),
                    "relation": arc.get("dep", "unknown"),
                })
            else:
                result.append({
                    "id": i,
                    "word": token,
                    "head": 0,
                    "relation": str(arc),
                })
        return result

    def semantic_role_labeling(self, text: str) -> list[dict]:
        """语义角色标注"""
        tokenizer = self._get_tokenizer()
        tokens = tokenizer(text)
        srl = self._get_srl()
        frames = srl(tokens)

        result = []
        if isinstance(frames, list):
            for frame in frames:
                if isinstance(frame, dict):
                    result.append({
                        "predicate": frame.get("predicate", ""),
                        "arguments": frame.get("arguments", []),
                    })
        return result
