# tests/test_text_analyst.py
import pytest
from unittest.mock import patch, MagicMock
from src.agents.base import AnalysisContext, AnalysisDepth
from src.agents.text_analyst import TextAnalystAgent


def test_text_analyst_name():
    """测试 Agent 名称"""
    agent = TextAnalystAgent()
    assert agent.name == "text_analyst"


def test_text_analyst_description():
    """测试 Agent 描述"""
    agent = TextAnalystAgent()
    assert len(agent.description) > 10


def test_analyze_sentence_structure():
    """测试句法结构分析"""
    agent = TextAnalystAgent()
    # Mock NLP components to avoid model downloads
    agent._tokenizer = MagicMock()
    agent._tokenizer.tokenize.return_value = ["小明", "吃", "了", "一个", "苹果"]
    agent._tokenizer.tokenize_with_pos.return_value = [
        ("小明", "nr"), ("吃", "v"), ("了", "ul"), ("一个", "m"), ("苹果", "n")
    ]
    agent._tokenizer.split_sentences.return_value = ["小明吃了一个苹果"]

    agent._syntax = MagicMock()
    agent._syntax.parse_dependencies.return_value = [
        {"id": 0, "word": "小明", "head": 1, "relation": "nsubj"},
    ]

    agent._sentiment = MagicMock()
    agent._sentiment.analyze.return_value = {"label": "neutral", "score": 0.6}

    agent._rhetoric = MagicMock()
    agent._rhetoric.detect.return_value = []

    ctx = AnalysisContext(text="小明吃了一个苹果。", depth=AnalysisDepth.STANDARD)
    result = agent.analyze(ctx)
    assert result.agent_name == "text_analyst"
    assert "sentence_types" in result.analysis
    assert result.confidence > 0


def test_analyze_discourse_markers():
    """测试话语标记检测"""
    agent = TextAnalystAgent()
    agent._tokenizer = MagicMock()
    agent._tokenizer.tokenize.return_value = ["虽然", "他", "很", "努力"]
    agent._tokenizer.tokenize_with_pos.return_value = [("虽然", "c"), ("他", "r")]
    agent._tokenizer.split_sentences.return_value = ["虽然他很努力，但是结果并不理想"]

    agent._syntax = MagicMock()
    agent._syntax.parse_dependencies.return_value = []

    agent._sentiment = MagicMock()
    agent._sentiment.analyze.return_value = {"label": "negative", "score": 0.7}

    agent._rhetoric = MagicMock()
    agent._rhetoric.detect.return_value = []

    ctx = AnalysisContext(text="虽然他很努力，但是结果并不理想。", depth=AnalysisDepth.STANDARD)
    result = agent.analyze(ctx)
    markers = result.analysis.get("discourse_markers", [])
    marker_texts = [m["marker"] for m in markers]
    assert "虽然" in marker_texts or "但是" in marker_texts


def test_analyze_modality():
    """测试语气分析"""
    agent = TextAnalystAgent()
    agent._tokenizer = MagicMock()
    agent._tokenizer.tokenize.return_value = ["你", "必须", "完成"]
    agent._tokenizer.tokenize_with_pos.return_value = [("你", "r"), ("必须", "d"), ("完成", "v")]
    agent._tokenizer.split_sentences.return_value = ["你必须完成这个任务"]

    agent._syntax = MagicMock()
    agent._syntax.parse_dependencies.return_value = []

    agent._sentiment = MagicMock()
    agent._sentiment.analyze.return_value = {"label": "neutral", "score": 0.5}

    agent._rhetoric = MagicMock()
    agent._rhetoric.detect.return_value = []

    ctx = AnalysisContext(text="你必须完成这个任务", depth=AnalysisDepth.STANDARD)
    result = agent.analyze(ctx)
    modality = result.analysis.get("modality", {})
    modal_words = [m["word"] for m in modality.get("modal_verbs", [])]
    assert "必须" in modal_words


def test_analyze_with_rhetoric():
    """测试修辞识别集成"""
    agent = TextAnalystAgent()
    agent._tokenizer = MagicMock()
    agent._tokenizer.tokenize.return_value = ["他", "的", "心", "像", "冰"]
    agent._tokenizer.tokenize_with_pos.return_value = [("他", "r"), ("心", "n")]
    agent._tokenizer.split_sentences.return_value = ["他的心像冰一样冷"]

    agent._syntax = MagicMock()
    agent._syntax.parse_dependencies.return_value = []

    agent._sentiment = MagicMock()
    agent._sentiment.analyze.return_value = {"label": "negative", "score": 0.8}

    agent._rhetoric = MagicMock()
    agent._rhetoric.detect.return_value = [
        {"type": "simile", "text_span": "像冰一样", "confidence": 0.7, "description": "明喻"}
    ]

    ctx = AnalysisContext(text="他的心像冰一样冷", depth=AnalysisDepth.DEEP)
    result = agent.analyze(ctx)
    assert len(result.analysis.get("rhetorical_devices", [])) > 0
