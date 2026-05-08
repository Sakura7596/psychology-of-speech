import pytest
from unittest.mock import patch, MagicMock


def test_tokenizer_basic():
    """测试基本分词"""
    from src.nlp.tokenizer import Tokenizer
    tok = Tokenizer(use_hanlp=False)  # Force jieba for testing
    tokens = tok.tokenize("今天天气真不错")
    assert len(tokens) > 0
    assert isinstance(tokens, list)
    assert all(isinstance(t, str) for t in tokens)


def test_tokenizer_sentence_split():
    """测试断句"""
    from src.nlp.tokenizer import Tokenizer
    tok = Tokenizer()
    sentences = tok.split_sentences("你好。今天天气不错！我们去散步吧？")
    assert len(sentences) == 3
    assert "你好" in sentences[0]


def test_tokenizer_pos_tags():
    """测试词性标注（jieba 模式）"""
    from src.nlp.tokenizer import Tokenizer
    tok = Tokenizer(use_hanlp=False)
    result = tok.tokenize_with_pos("我喜欢苹果")
    assert len(result) > 0
    assert all(isinstance(item, tuple) and len(item) == 2 for item in result)


def test_syntax_dependency_mock():
    """测试依存句法分析（mock HanLP）"""
    from src.nlp.syntax import SyntaxAnalyzer
    analyzer = SyntaxAnalyzer()

    # Mock HanLP components
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = ["小明", "吃", "苹果"]
    analyzer._tokenizer = mock_tokenizer

    mock_parser = MagicMock()
    mock_parser.return_value = [
        {"head": 1, "dep": "nsubj"},
        {"head": 0, "dep": "root"},
        {"head": 1, "dep": "dobj"},
    ]
    analyzer._dep_parser = mock_parser

    deps = analyzer.parse_dependencies("小明吃苹果")
    assert len(deps) == 3
    assert deps[0]["word"] == "小明"
    assert deps[0]["relation"] == "nsubj"


def test_syntax_srl_mock():
    """测试语义角色标注（mock HanLP）"""
    from src.nlp.syntax import SyntaxAnalyzer
    analyzer = SyntaxAnalyzer()

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = ["小明", "在", "公园", "里", "吃", "苹果"]
    analyzer._tokenizer = mock_tokenizer

    mock_srl = MagicMock()
    mock_srl.return_value = [
        {"predicate": "吃", "arguments": [{"role": "A0", "text": "小明"}, {"role": "A1", "text": "苹果"}]}
    ]
    analyzer._srl = mock_srl

    srl = analyzer.semantic_role_labeling("小明在公园里吃苹果")
    assert isinstance(srl, list)
    assert len(srl) > 0
    assert srl[0]["predicate"] == "吃"


def test_tokenizer_empty_text():
    """测试空文本"""
    from src.nlp.tokenizer import Tokenizer
    tok = Tokenizer(use_hanlp=False)
    tokens = tok.tokenize("")
    assert tokens == []
