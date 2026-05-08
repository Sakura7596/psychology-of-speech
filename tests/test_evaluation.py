# tests/test_evaluation.py
import pytest
import json
import tempfile
import os
from src.evaluation.schema import BenchmarkItem, validate_benchmark, load_benchmark
from src.evaluation.scorer import TextAnalysisScorer


def test_benchmark_item_creation():
    """测试评测项创建"""
    item = BenchmarkItem(
        id="test_001",
        text="今天天气真好",
        expected={"sentiment": "positive"},
    )
    assert item.id == "test_001"
    assert item.text == "今天天气真好"


def test_validate_benchmark():
    """测试基准数据集验证"""
    items = [
        BenchmarkItem(id="t1", text="你好", expected={"sentiment": "neutral"}),
        BenchmarkItem(id="t2", text="太棒了", expected={"sentiment": "positive"}),
    ]
    assert validate_benchmark(items) is True


def test_validate_benchmark_duplicate_id():
    """测试重复 ID 验证"""
    items = [
        BenchmarkItem(id="t1", text="你好", expected={"sentiment": "neutral"}),
        BenchmarkItem(id="t1", text="太棒了", expected={"sentiment": "positive"}),
    ]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_benchmark(items)


def test_load_benchmark():
    """测试加载基准数据集"""
    items = load_benchmark("data/golden/text_analysis_benchmark.json")
    assert len(items) > 0
    assert all(isinstance(item, BenchmarkItem) for item in items)


def test_scorer_basic():
    """测试基础评分"""
    scorer = TextAnalysisScorer()
    predicted = {"sentiment": {"label": "positive"}}
    expected = {"sentiment": "positive"}
    score = scorer.score_item(predicted, expected)
    assert 0.0 <= score <= 1.0
    assert score == 1.0


def test_scorer_wrong():
    """测试错误预测评分"""
    scorer = TextAnalysisScorer()
    predicted = {"sentiment": {"label": "negative"}}
    expected = {"sentiment": "positive"}
    score = scorer.score_item(predicted, expected)
    assert score == 0.0


def test_scorer_batch():
    """测试批量评分"""
    scorer = TextAnalysisScorer()
    results = [
        {"predicted": {"sentiment": {"label": "positive"}}, "expected": {"sentiment": "positive"}},
        {"predicted": {"sentiment": {"label": "negative"}}, "expected": {"sentiment": "positive"}},
    ]
    avg_score = scorer.score_batch(results)
    assert avg_score == 0.5
