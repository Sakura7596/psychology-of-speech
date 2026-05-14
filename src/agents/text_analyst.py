# src/agents/text_analyst.py
from src.agents.base import BaseAgent, AnalysisContext, AgentResult, AnalysisDepth
from src.nlp.tokenizer import Tokenizer
from src.nlp.syntax import SyntaxAnalyzer
from src.nlp.sentiment import SentimentAnalyzer
from src.nlp.rhetoric import RhetoricDetector


class TextAnalystAgent(BaseAgent):
    """文本解析 Agent - 语言学层面的结构化分析"""

    def __init__(self):
        self._tokenizer = Tokenizer()
        self._syntax = SyntaxAnalyzer()
        self._sentiment = SentimentAnalyzer()
        self._rhetoric = RhetoricDetector()

    @property
    def name(self) -> str:
        return "text_analyst"

    @property
    def description(self) -> str:
        return "语言学层面的深度解析：句法、词汇、语气、修辞、话语标记"

    async def analyze(self, context: AnalysisContext) -> AgentResult:
        """执行文本分析"""
        text = context.text
        depth = context.depth

        analysis = {}

        # 1. 句法结构
        analysis["sentence_types"] = self._analyze_sentence_types(text)
        try:
            analysis["dependency_parse"] = self._syntax.parse_dependencies(text)
        except Exception:
            analysis["dependency_parse"] = []

        # 2. 词汇分析
        tokens = self._tokenizer.tokenize(text)
        pos_tags = self._tokenizer.tokenize_with_pos(text)
        analysis["token_count"] = len(tokens)
        analysis["vocabulary"] = self._analyze_vocabulary(tokens, pos_tags)

        # 3. 语气与情态
        analysis["modality"] = self._analyze_modality(text)

        # 4. 修辞手法
        if depth in (AnalysisDepth.STANDARD, AnalysisDepth.DEEP):
            analysis["rhetorical_devices"] = self._rhetoric.detect(text)

        # 5. 话语标记
        analysis["discourse_markers"] = self._detect_discourse_markers(text)

        # 情感
        try:
            analysis["sentiment"] = self._sentiment.analyze(text)
        except Exception:
            analysis["sentiment"] = {"label": "unknown", "score": 0.0, "error": "sentiment module unavailable"}

        # 置信度
        confidence = self._calculate_confidence(analysis)

        return AgentResult(
            agent_name=self.name,
            analysis=analysis,
            confidence=confidence,
            sources=["HanLP", "jieba", "transformers"],
        )

    def _analyze_sentence_types(self, text: str) -> dict:
        sentences = self._tokenizer.split_sentences(text)
        types = {"declarative": 0, "interrogative": 0, "imperative": 0, "exclamatory": 0}
        for s in sentences:
            if "？" in s or "?" in s:
                types["interrogative"] += 1
            elif "！" in s or "!" in s:
                types["exclamatory"] += 1
            else:
                types["declarative"] += 1
        return {"count": len(sentences), "types": types}

    def _analyze_vocabulary(self, tokens: list[str], pos_tags: list[tuple[str, str]]) -> dict:
        vague_words = ["大概", "可能", "也许", "或许", "基本上", "差不多", "似乎"]
        found_vague = [t for t in tokens if t in vague_words]
        formal_pos = {"nr", "ns", "nt", "nz", "vn", "an"}
        formal_count = sum(1 for _, tag in pos_tags if tag in formal_pos)
        formality = formal_count / max(len(pos_tags), 1)
        return {
            "unique_tokens": len(set(tokens)),
            "vague_words": found_vague,
            "formality_score": round(formality, 3),
        }

    def _analyze_modality(self, text: str) -> dict:
        modal_verbs = {
            "必须": "necessity", "应该": "obligation", "可以": "permission",
            "能": "ability", "会": "possibility", "需要": "necessity", "想要": "desire",
        }
        found = [{"word": w, "type": t} for w, t in modal_verbs.items() if w in text]
        return {"modal_verbs": found, "modal_count": len(found)}

    def _detect_discourse_markers(self, text: str) -> list[dict]:
        markers = {
            "但是": "转折", "可是": "转折", "然而": "转折", "不过": "转折",
            "其实": "强调", "事实上": "强调", "说实话": "强调",
            "总之": "总结", "综上": "总结",
            "另外": "补充", "而且": "递进", "并且": "递进",
            "因为": "因果", "所以": "因果", "因此": "因果",
            "虽然": "让步", "尽管": "让步",
        }
        return [{"marker": m, "category": c} for m, c in markers.items() if m in text]

    def _calculate_confidence(self, analysis: dict) -> float:
        score = 0.5
        if analysis.get("dependency_parse"):
            score += 0.1
        if analysis.get("vocabulary", {}).get("vague_words") is not None:
            score += 0.1
        if analysis.get("modality"):
            score += 0.1
        if analysis.get("rhetorical_devices") is not None:
            score += 0.1
        if analysis.get("discourse_markers"):
            score += 0.1
        return min(score, 1.0)
