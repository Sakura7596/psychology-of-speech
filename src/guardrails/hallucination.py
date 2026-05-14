import logging
from src.agents.base import RELIABILITY_THRESHOLD

logger = logging.getLogger(__name__)


class HallucinationGuard:
    def __init__(self, min_sources: int = 1, min_confidence: float = 0.3):
        self._min_sources = min_sources
        self._min_confidence = min_confidence

    def check(self, analysis: dict) -> dict:
        issues = []
        warnings = []

        # 1. 来源检查
        sources = analysis.get("sources", [])
        if len(sources) < self._min_sources:
            issues.append(f"分析缺少来源引用（需要至少 {self._min_sources} 个来源）")

        # 2. 置信度检查
        confidence = analysis.get("confidence", 0.5)
        if confidence < self._min_confidence:
            warnings.append(f"置信度过低（{confidence}），结果为推测性分析")

        # 3. 解析错误检查
        if analysis.get("parse_error"):
            issues.append("LLM 响应解析失败")

        # 4. 内容一致性检查
        content_checks = self._check_content_consistency(analysis)
        issues.extend(content_checks.get("issues", []))
        warnings.extend(content_checks.get("warnings", []))

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "warning": "; ".join(warnings) if warnings else "",
            "reason": "; ".join(issues) if issues else "",
        }

    def _check_content_consistency(self, analysis: dict) -> dict:
        """检查 LLM 输出内容的内部一致性"""
        issues = []
        warnings = []

        # 检查置信度字段是否合理
        declared_conf = analysis.get("confidence")
        if declared_conf is not None:
            if declared_conf > 0.95:
                warnings.append("声明置信度过高（>0.95），可能存在过度自信")
            if declared_conf < 0.0:
                issues.append("置信度值异常（<0.0）")

        # 检查分析结果是否为空
        inner_analysis = analysis.get("analysis", {})
        if isinstance(inner_analysis, dict):
            # 检查是否有实质性内容
            non_empty_keys = [
                k for k, v in inner_analysis.items()
                if v and v != [] and v != {} and k not in ("error", "parse_error")
            ]
            if not non_empty_keys:
                warnings.append("分析结果缺乏实质性内容")

        # 检查是否有错误标记
        if inner_analysis.get("error"):
            warnings.append(f"分析过程出现错误: {inner_analysis['error']}")

        return {"issues": issues, "warnings": warnings}

    def cross_validate(
        self, analysis_text: str, source_texts: list[str], knowledge_context: str = ""
    ) -> dict:
        """交叉验证分析结论与源文本/知识库的一致性"""
        import re as _re
        warnings = []

        if knowledge_context:
            # 提取中文词组（2-4字）做 token 级重叠检查
            analysis_tokens = set(_re.findall(r'[一-鿿]{2,4}', analysis_text))
            context_tokens = set(_re.findall(r'[一-鿿]{2,4}', knowledge_context))
            if analysis_tokens and context_tokens:
                overlap = len(analysis_tokens & context_tokens) / len(analysis_tokens)
                if overlap < 0.05:
                    warnings.append("分析内容与知识库关联度较低")

        if source_texts:
            # 检查分析结论是否能在源文本中找到依据
            source_combined = " ".join(source_texts)
            analysis_tokens = set(_re.findall(r'[一-鿿]{2,4}', analysis_text))
            source_tokens = set(_re.findall(r'[一-鿿]{2,4}', source_combined))
            if analysis_tokens and source_tokens:
                overlap = len(analysis_tokens & source_tokens) / len(analysis_tokens)
                if overlap < 0.03:
                    warnings.append("分析结论与源文本关联度较低")

        return {
            "cross_validated": True,
            "warnings": warnings,
        }
