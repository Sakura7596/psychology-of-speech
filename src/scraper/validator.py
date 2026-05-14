import re


class ContentValidator:
    MIN_TEXT_LENGTH = 10
    MAX_TEXT_LENGTH = 500
    MIN_KEYWORDS = 2
    MIN_ANALYSIS_LENGTH = 30

    VALID_TYPES = {
        "romantic", "dating", "romantic_breakup", "romantic_conflict",
        "romantic_chase", "attachment", "emotional", "psychology",
        "speech_act", "rhetoric", "dating_app",
    }

    def __init__(self, existing_cases: list[dict] | None = None):
        self._existing_texts: set[str] = set()
        if existing_cases:
            for case in existing_cases:
                self._existing_texts.add(case.get("text", ""))

    def validate_case(self, case: dict) -> tuple[bool, list[str]]:
        issues = []

        required = {"type", "subtype", "text", "keywords", "analysis", "psychological_state", "theories"}
        missing = required - set(case.keys())
        if missing:
            issues.append(f"缺少字段: {missing}")
            return False, issues

        if case["type"] not in self.VALID_TYPES:
            issues.append(f"无效类型: {case['type']}")

        text = case.get("text", "")
        if len(text) < self.MIN_TEXT_LENGTH:
            issues.append(f"文本过短: {len(text)} 字符")
        if len(text) > self.MAX_TEXT_LENGTH:
            issues.append(f"文本过长: {len(text)} 字符")

        keywords = case.get("keywords", [])
        if len(keywords) < self.MIN_KEYWORDS:
            issues.append(f"关键词过少: {len(keywords)}")

        analysis = case.get("analysis", "")
        if len(analysis) < self.MIN_ANALYSIS_LENGTH:
            issues.append(f"分析过短: {len(analysis)} 字符")

        if text in self._existing_texts:
            issues.append("重复文本")

        theories = case.get("theories", [])
        if not theories:
            issues.append("缺少理论引用")

        return len(issues) == 0, issues

    def generate_case_id(self, case_type: str, source: str, existing_ids: set[str]) -> str:
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

    def _tokenize(self, text: str) -> set[str]:
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
