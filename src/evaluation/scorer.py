# src/evaluation/scorer.py
class TextAnalysisScorer:
    """文本分析评测评分器"""

    def score_item(self, predicted: dict, expected: dict) -> float:
        """评分单个分析结果"""
        total = 0
        correct = 0
        for key, expected_value in expected.items():
            total += 1
            predicted_value = predicted.get(key)
            if isinstance(predicted_value, dict):
                if predicted_value.get("label") == expected_value:
                    correct += 1
            elif predicted_value == expected_value:
                correct += 1
        return correct / max(total, 1)

    def score_batch(self, results: list[dict]) -> float:
        """批量评分"""
        if not results:
            return 0.0
        scores = [self.score_item(r["predicted"], r["expected"]) for r in results]
        return sum(scores) / len(scores)
