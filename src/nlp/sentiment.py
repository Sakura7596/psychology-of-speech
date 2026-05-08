class SentimentAnalyzer:
    """中文情感分析 - 基于 transformers 预训练模型"""

    def __init__(self, model_name: str = "uer/roberta-base-finetuned-chinanews-chinese"):
        self._model_name = model_name
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            from transformers import pipeline as hf_pipeline
            self._pipeline = hf_pipeline(
                "sentiment-analysis",
                model=self._model_name,
                top_k=None,
            )
        return self._pipeline

    def analyze(self, text: str) -> dict:
        """基础情感分析"""
        pipe = self._get_pipeline()
        results = pipe(text)
        if isinstance(results, list) and len(results) > 0:
            if isinstance(results[0], list):
                results = results[0]
            top = max(results, key=lambda x: x["score"])
            return {
                "label": top["label"].lower(),
                "score": round(top["score"], 4),
            }
        return {"label": "neutral", "score": 0.5}

    def analyze_detail(self, text: str) -> dict:
        """细粒度情感分析"""
        pipe = self._get_pipeline()
        results = pipe(text)
        if isinstance(results, list) and len(results) > 0:
            if isinstance(results[0], list):
                results = results[0]

        scores = {r["label"].lower(): round(r["score"], 4) for r in results}
        positive_score = scores.get("positive", 0.0)
        negative_score = scores.get("negative", 0.0)

        return {
            "positive_score": positive_score,
            "negative_score": negative_score,
            "neutral_score": scores.get("neutral", 0.0),
            "dominant_emotion": "positive" if positive_score > negative_score else "negative",
            "all_scores": scores,
        }
