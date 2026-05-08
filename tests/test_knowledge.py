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


from src.knowledge.knowledge_graph import KnowledgeGraph


def test_kg_add_entity():
    """测试添加实体"""
    kg = KnowledgeGraph()
    kg.add_entity("言语行为理论", "theory", {"founder": "Austin"})
    assert kg.has_entity("言语行为理论")


def test_kg_add_relation():
    """测试添加关系"""
    kg = KnowledgeGraph()
    kg.add_entity("言语行为理论", "theory")
    kg.add_entity("断言", "speech_act")
    kg.add_relation("言语行为理论", "断言", "defines")

    neighbors = kg.get_neighbors("言语行为理论")
    assert "断言" in [n[0] for n in neighbors]


def test_kg_find_path():
    """测试路径查找"""
    kg = KnowledgeGraph()
    kg.add_entity("A", "theory")
    kg.add_entity("B", "concept")
    kg.add_entity("C", "feature")
    kg.add_relation("A", "B", "applies_to")
    kg.add_relation("B", "C", "indicates")

    path = kg.find_path("A", "C")
    assert len(path) >= 2
    assert path[0] == "A"
    assert path[-1] == "C"


def test_kg_save_load():
    """测试序列化和反序列化"""
    import tempfile, os

    kg = KnowledgeGraph()
    kg.add_entity("test", "type")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        path = f.name
    try:
        kg.save(path)
        kg2 = KnowledgeGraph()
        kg2.load(path)
        assert kg2.has_entity("test")
    finally:
        os.unlink(path)


def test_kg_query_by_type():
    """测试按类型查询"""
    kg = KnowledgeGraph()
    kg.add_entity("言语行为理论", "theory")
    kg.add_entity("会话含义", "theory")
    kg.add_entity("断言", "speech_act")

    theories = kg.get_entities_by_type("theory")
    assert len(theories) == 2


from src.knowledge.case_library import CaseLibrary


def test_case_library_load():
    """测试加载案例库"""
    lib = CaseLibrary("data/cases")
    assert lib.count() > 0


def test_case_library_search():
    """测试搜索案例"""
    lib = CaseLibrary("data/cases")
    results = lib.search("比喻")
    assert len(results) > 0


def test_case_library_filter_by_type():
    """测试按类型过滤"""
    lib = CaseLibrary("data/cases")
    rhetoric_cases = lib.get_by_type("rhetoric")
    assert len(rhetoric_cases) > 0
    assert all(c["type"] == "rhetoric" for c in rhetoric_cases)


def test_theory_library_load():
    """测试加载理论库"""
    lib = CaseLibrary("data/literature")
    assert lib.count() > 0


def test_theory_search_speech_act():
    """测试搜索言语行为理论"""
    lib = CaseLibrary("data/literature")
    results = lib.search("言语行为")
    assert len(results) > 0


def test_knowledge_graph_load_data():
    """测试加载知识图谱数据"""
    kg = KnowledgeGraph()
    kg.load("data/graph/psychology_graph.json")
    assert kg.has_entity("言语行为理论")
    assert kg.has_entity("会话含义理论")


def test_knowledge_graph_relations():
    """测试图谱关系"""
    kg = KnowledgeGraph()
    kg.load("data/graph/psychology_graph.json")
    neighbors = kg.get_neighbors("言语行为理论")
    assert len(neighbors) > 0


from src.knowledge.retriever import KnowledgeRetriever
from unittest.mock import MagicMock


def test_retriever_initialization():
    """测试检索器初始化"""
    retriever = KnowledgeRetriever(
        vector_store=None,
        knowledge_graph=None,
        case_library=None,
    )
    assert retriever is not None


def test_retriever_with_mock():
    """测试融合检索（mock）"""
    mock_vs = MagicMock()
    mock_vs.query.return_value = [
        {"id": "v1", "text": "向量结果", "metadata": {"source": "vector"}, "distance": 0.1}
    ]

    mock_kg = MagicMock()
    mock_kg.get_neighbors.return_value = [("相关实体", {"relation": "related"})]
    mock_kg.find_path.return_value = ["A", "B"]

    mock_cl = MagicMock()
    mock_cl.search.return_value = [
        {"id": "c1", "text": "案例结果", "type": "rhetoric"}
    ]

    retriever = KnowledgeRetriever(
        vector_store=mock_vs,
        knowledge_graph=mock_kg,
        case_library=mock_cl,
    )

    results = retriever.retrieve("比喻修辞")
    assert "vector_results" in results
    assert "graph_results" in results
    assert "case_results" in results


def test_retriever_context_string():
    """测试上下文字符串生成"""
    mock_vs = MagicMock()
    mock_vs.query.return_value = [{"text": "相关知识"}]

    mock_cl = MagicMock()
    mock_cl.search.return_value = [{"subtype": "simile", "text": "像冰一样", "analysis": "明喻"}]

    retriever = KnowledgeRetriever(vector_store=mock_vs, case_library=mock_cl)
    context = retriever.get_context_string("比喻")
    assert "相关知识" in context or "明喻" in context
