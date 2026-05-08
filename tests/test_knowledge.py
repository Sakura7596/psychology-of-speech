import pytest
import numpy as np
from src.knowledge.embedding import EmbeddingModel


def test_embedding_encode_single():
    """测试单文本编码"""
    model = EmbeddingModel()
    model._model = type("MockModel", (), {
        "encode": lambda self, texts, **kwargs: [np.zeros(768).tolist() for _ in texts]
    })()

    vec = model.encode("今天天气真好")
    assert len(vec) == 768
    assert isinstance(vec, list)


def test_embedding_encode_batch():
    """测试批量编码"""
    model = EmbeddingModel()
    model._model = type("MockModel", (), {
        "encode": lambda self, texts, **kwargs: [np.zeros(768).tolist() for _ in texts]
    })()

    vecs = model.encode_batch(["你好", "世界"])
    assert len(vecs) == 2
    assert all(len(v) == 768 for v in vecs)


def test_embedding_dimension():
    """测试向量维度"""
    model = EmbeddingModel()
    assert model.dimension == 768
