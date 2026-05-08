import re

DISCLAIMER = "\n---\n**免责声明**：本分析基于语言学特征的辅助分析，仅供参考，不构成专业心理咨询、诊断或治疗建议。如需心理帮助，请咨询专业心理咨询师或医疗机构。\n"


class EthicsGuard:
    DIAGNOSTIC_PATTERNS = [
        r'患有.{1,10}[症病]', r'诊断.{0,5}为', r'确诊',
        r'精神.{0,3}[疾病障碍]', r'心理.{0,3}[疾病障碍变态]',
        r'人格障碍', r'需要.{0,5}治疗', r'建议.{0,5}[吃药服药就医住院]',
    ]

    def inject_disclaimer(self, report: str) -> str:
        if "免责声明" not in report and "仅供参考" not in report:
            return report + DISCLAIMER
        return report

    def check_diagnostic_language(self, text: str) -> dict:
        found = []
        for pattern in self.DIAGNOSTIC_PATTERNS:
            found.extend(re.findall(pattern, text))
        return {"has_diagnostic": len(found) > 0, "found_terms": found}

    def sanitize_output(self, report: str) -> str:
        diag = self.check_diagnostic_language(report)
        if diag["has_diagnostic"]:
            for term in diag["found_terms"]:
                report = report.replace(term, "[已移除诊断术语]")
        return self.inject_disclaimer(report)
