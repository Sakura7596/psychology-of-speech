class SentimentAnalyzer:
    """中文情感分析 - transformers 模型优先，规则 fallback"""

    # 中文情感关键词（规则 fallback 用）
    POSITIVE_WORDS = {
        "开心", "高兴", "快乐", "幸福", "喜欢", "爱", "感谢", "谢谢", "棒", "好",
        "优秀", "厉害", "赞", "完美", "精彩", "满意", "舒服", "温暖", "感动", "期待",
        "希望", "相信", "支持", "鼓励", "同意", "不错", "太好了", "真好", "很棒", "加油",
        "beautiful", "happy", "great", "love", "good", "wonderful", "excellent", "amazing",
    }
    NEGATIVE_WORDS = {
        "难过", "伤心", "痛苦", "生气", "愤怒", "讨厌", "恨", "失望", "担心", "焦虑",
        "害怕", "恐惧", "抱歉", "对不起", "后悔", "孤独", "无聊", "烦", "累", "崩溃",
        "糟糕", "可怕", "恶心", "无语", "难受", "不开心", "不高兴", "不满", "委屈", "生气",
        "sad", "angry", "hate", "terrible", "bad", "awful", "disgusting", "disappointed",
    }
    NEGATION_WORDS = {"不", "没", "没有", "别", "未", "非", "无", "莫", "勿"}

    def __init__(self, model_name: str = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"):
        self._model_name = model_name
        self._pipeline = None
        self._use_transformers = True

    def _get_pipeline(self):
        if self._pipeline is None:
            from transformers import pipeline as hf_pipeline
            self._pipeline = hf_pipeline(
                "sentiment-analysis",
                model=self._model_name,
                top_k=None,
            )
        return self._pipeline

    def _rule_based_analyze(self, text: str) -> dict:
        """基于关键词的规则情感分析（fallback）"""
        pos_count = 0
        neg_count = 0

        for word in self.POSITIVE_WORDS:
            count = text.count(word)
            if count > 0:
                # 检查否定词（简单：词前面是否有否定）
                idx = text.find(word)
                prefix = text[max(0, idx - 2):idx]
                if any(neg in prefix for neg in self.NEGATION_WORDS):
                    neg_count += count
                else:
                    pos_count += count

        for word in self.NEGATIVE_WORDS:
            count = text.count(word)
            if count > 0:
                idx = text.find(word)
                prefix = text[max(0, idx - 2):idx]
                if any(neg in prefix for neg in self.NEGATION_WORDS):
                    pos_count += count
                else:
                    neg_count += count

        total = pos_count + neg_count
        if total == 0:
            return {"label": "neutral", "score": 0.5, "method": "rule"}

        pos_ratio = pos_count / total
        neg_ratio = neg_count / total

        if pos_ratio > neg_ratio:
            return {"label": "positive", "score": round(0.5 + pos_ratio * 0.4, 4), "method": "rule"}
        elif neg_ratio > pos_ratio:
            return {"label": "negative", "score": round(0.5 + neg_ratio * 0.4, 4), "method": "rule"}
        else:
            return {"label": "neutral", "score": 0.5, "method": "rule"}

    def analyze(self, text: str) -> dict:
        """基础情感分析"""
        if self._use_transformers:
            try:
                pipe = self._get_pipeline()
                results = pipe(text)
                if isinstance(results, list) and len(results) > 0:
                    if isinstance(results[0], list):
                        results = results[0]
                    top = max(results, key=lambda x: x["score"])
                    return {
                        "label": top["label"].lower(),
                        "score": round(top["score"], 4),
                        "method": "transformers",
                    }
            except (ImportError, Exception):
                self._use_transformers = False

        return self._rule_based_analyze(text)

    def analyze_detail(self, text: str) -> dict:
        """细粒度情感分析"""
        if self._use_transformers:
            try:
                pipe = self._get_pipeline()
                results = pipe(text)
                if isinstance(results, list) and len(results) > 0:
                    if isinstance(results[0], list):
                        results = results[0]

                scores = {r["label"].lower(): round(r["score"], 4) for r in results}
                positive_score = scores.get("positive", 0.0)
                negative_score = scores.get("negative", 0.0)

                if positive_score > negative_score:
                    dominant = "positive"
                elif negative_score > positive_score:
                    dominant = "negative"
                else:
                    dominant = "neutral"

                return {
                    "positive_score": positive_score,
                    "negative_score": negative_score,
                    "neutral_score": scores.get("neutral", 0.0),
                    "dominant_emotion": dominant,
                    "all_scores": scores,
                    "method": "transformers",
                }
            except (ImportError, Exception):
                self._use_transformers = False

        result = self._rule_based_analyze(text)
        positive_score = result["score"] if result["label"] == "positive" else 0.0
        negative_score = result["score"] if result["label"] == "negative" else 0.0
        neutral_score = result["score"] if result["label"] == "neutral" else 0.0

        return {
            "positive_score": positive_score,
            "negative_score": negative_score,
            "neutral_score": neutral_score,
            "dominant_emotion": result["label"],
            "all_scores": {"positive": positive_score, "negative": negative_score, "neutral": neutral_score},
            "method": "rule",
        }
