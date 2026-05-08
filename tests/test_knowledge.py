import pytest
import numpy as np
from src.knowledge.embedding import EmbeddingModel
from src.knowledge.vector_store import VectorStore
import tempfile


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


def test_vector_store_add_and_query():
    """测试添加和查询"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(persist_dir=tmpdir, collection_name="test")
        store._embedding = type("MockEmbedding", (), {
            "encode": lambda self, text: [0.1] * 768,
            "encode_batch": lambda self, texts: [[0.1] * 768 for _ in texts],
        })()

        store.add("test_1", "今天天气真好", {"category": "sentiment"})
        results = store.query("天气", n_results=1)

        assert len(results) > 0
        assert results[0]["id"] == "test_1"
        store.close()


def test_vector_store_metadata_filter():
    """测试元数据过滤"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(persist_dir=tmpdir, collection_name="test_meta")
        store._embedding = type("MockEmbedding", (), {
            "encode": lambda self, text: [0.1] * 768,
            "encode_batch": lambda self, texts: [[0.1] * 768 for _ in texts],
        })()

        store.add("t1", "文本1", {"category": "sentiment"})
        store.add("t2", "文本2", {"category": "rhetoric"})

        results = store.query("文本", n_results=10, where={"category": "sentiment"})
        assert len(results) == 1
        assert results[0]["metadata"]["category"] == "sentiment"
        store.close()


def test_vector_store_delete():
    """测试删除"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(persist_dir=tmpdir, collection_name="test_del")
        store._embedding = type("MockEmbedding", (), {
            "encode": lambda self, text: [0.1] * 768,
            "encode_batch": lambda self, texts: [[0.1] * 768 for _ in texts],
        })()

        store.add("t1", "文本1")
        store.delete("t1")
        results = store.query("文本", n_results=1)
        assert len(results) == 0
        store.close()


def test_vector_store_count():
    """测试文档计数"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(persist_dir=tmpdir, collection_name="test_count")
        store._embedding = type("MockEmbedding", (), {
            "encode": lambda self, text: [0.1] * 768,
            "encode_batch": lambda self, texts: [[0.1] * 768 for _ in texts],
        })()

        assert store.count() == 0
        store.add("t1", "文本1")
        assert store.count() == 1
        store.close()
