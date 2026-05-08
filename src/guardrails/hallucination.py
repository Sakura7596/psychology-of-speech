from src.agents.base import RELIABILITY_THRESHOLD


class HallucinationGuard:
    def __init__(self, min_sources: int = 1, min_confidence: float = 0.3):
        self._min_sources = min_sources
        self._min_confidence = min_confidence

    def check(self, analysis: dict) -> dict:
        issues = []
        warnings = []
        sources = analysis.get("sources", [])
        if len(sources) < self._min_sources:
            issues.append(f"分析缺少来源引用（需要至少 {self._min_sources} 个来源）")
        confidence = analysis.get("confidence", 0.5)
        if confidence < self._min_confidence:
            warnings.append(f"置信度过低（{confidence}），结果为推测性分析")
        if analysis.get("parse_error"):
            issues.append("LLM 响应解析失败")
        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "warning": "; ".join(warnings) if warnings else "",
            "reason": "; ".join(issues) if issues else "",
        }
