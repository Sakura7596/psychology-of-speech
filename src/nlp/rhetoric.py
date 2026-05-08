import re
from dataclasses import dataclass


class RhetoricDetector:
    """修辞手法检测器 - 规则 + 模式匹配"""

    METAPHOR_PATTERNS = [
        (r'像.{1,10}一样', 'simile', '明喻'),
        (r'如.{1,10}一般', 'simile', '明喻'),
        (r'仿佛.{1,10}', 'simile', '明喻'),
    ]

    RHETORICAL_QUESTION_PATTERNS = [
        (r'难道.{0,20}[吗？?]', 'rhetorical_question', '反问'),
        (r'怎么.{0,15}[呢？?]', 'rhetorical_question', '反问'),
        (r'岂.{0,15}[？?]', 'rhetorical_question', '反问'),
    ]

    EXAGGERATION_PATTERNS = [
        (r'[一二三四五六七八九十百千万亿]+[年月日天秒分钟]', 'exaggeration', '夸张'),
        (r'极[了]', 'exaggeration', '夸张'),
        (r'万分', 'exaggeration', '夸张'),
    ]

    def detect(self, text: str) -> list[dict]:
        """检测修辞手法"""
        results = []

        for pattern, rtype, desc in self.METAPHOR_PATTERNS:
            for m in re.finditer(pattern, text):
                results.append({
                    "type": rtype,
                    "text_span": m.group(),
                    "confidence": 0.7,
                    "description": desc,
                })

        for pattern, rtype, desc in self.RHETORICAL_QUESTION_PATTERNS:
            for m in re.finditer(pattern, text):
                results.append({
                    "type": rtype,
                    "text_span": m.group(),
                    "confidence": 0.8,
                    "description": desc,
                })

        for pattern, rtype, desc in self.EXAGGERATION_PATTERNS:
            for m in re.finditer(pattern, text):
                results.append({
                    "type": rtype,
                    "text_span": m.group(),
                    "confidence": 0.6,
                    "description": desc,
                })

        results.extend(self._detect_parallelism(text))
        return results

    def _detect_parallelism(self, text: str) -> list[dict]:
        """检测排比"""
        results = []
        clauses = re.split(r'[，；,;]', text)
        if len(clauses) >= 3:
            prefixes = [c.strip()[:4] for c in clauses if len(c.strip()) >= 4]
            if len(prefixes) >= 3:
                from collections import Counter
                prefix_counts = Counter(prefixes)
                most_common, count = prefix_counts.most_common(1)[0]
                if count >= 3:
                    results.append({
                        "type": "parallelism",
                        "text_span": text[:50],
                        "confidence": min(0.5 + count * 0.1, 0.9),
                        "description": f"排比（重复句式 '{most_common}'）",
                    })
        return results
