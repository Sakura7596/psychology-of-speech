# 阶段三：RAG 知识引擎实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建 RAG 知识引擎，包含向量检索（ChromaDB + text2vec）、知识图谱（NetworkX）、理论文献库、案例库，以及统一检索接口

**Architecture:** 三层知识存储：ChromaDB 向量库（语义检索）+ NetworkX 知识图谱（关系检索）+ JSON 案例库（精确匹配）。统一检索接口融合三种检索结果，为 Agent 提供增强上下文。

**Tech Stack:** ChromaDB, sentence-transformers (text2vec-base-chinese), NetworkX, pytest

**依赖：** 阶段一、二已完成

---

## 文件结构

```
src/
├── knowledge/
│   ├── __init__.py          # 知识引擎统一导出
│   ├── vector_store.py      # ChromaDB 向量库封装
│   ├── embedding.py         # text2vec Embedding 模型
│   ├── knowledge_graph.py   # NetworkX 知识图谱
│   ├── case_library.py      # 案例库（JSON + 向量化）
│   └── retriever.py         # 统一检索接口（融合检索）
data/
├── literature/
│   └── theories.json        # 理论文献库（核心理论条目）
├── cases/
│   ├── rhetoric_cases.json  # 修辞案例
│   └── fallacy_cases.json   # 逻辑谬误案例
├── graph/
│   └── psychology_graph.json # 知识图谱序列化
└── embeddings/              # ChromaDB 持久化目录
tests/
└── test_knowledge.py        # 知识引擎测试
```

---

## Task 16: Embedding 模型封装

**Files:**
- Create: `src/knowledge/embedding.py`
- Create: `tests/test_knowledge.py`

- [ ] **Step 1: 编写 Embedding 测试**

```python
# tests/test_knowledge.py
import pytest
import numpy as np
from src.knowledge.embedding import EmbeddingModel


def test_embedding_encode_single():
    """测试单文本编码"""
    model = EmbeddingModel()
    # Mock the actual model to avoid download
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
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_knowledge.py -v -k "embedding"
```

- [ ] **Step 3: 实现 Embedding 封装**

```python
# src/knowledge/embedding.py
import numpy as np


class EmbeddingModel:
    """中文 Embedding 模型 - text2vec-base-chinese"""

    def __init__(self, model_name: str = "shibing624/text2vec-base-chinese"):
        self._model_name = model_name
        self._model = None
        self._dimension = 768

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def encode(self, text: str) -> list[float]:
        """编码单个文本"""
        model = self._get_model()
        vec = model.encode(text)
        return vec.tolist()

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """批量编码"""
        model = self._get_model()
        vecs = model.encode(texts)
        return [v.tolist() for v in vecs]
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_knowledge.py -v -k "embedding"
```

- [ ] **Step 5: 提交**

```bash
git add src/knowledge/embedding.py src/knowledge/__init__.py tests/test_knowledge.py
git commit -m "feat: add text2vec-base-chinese embedding model wrapper"
```

---

## Task 17: ChromaDB 向量库

**Files:**
- Create: `src/knowledge/vector_store.py`
- Modify: `tests/test_knowledge.py`

- [ ] **Step 1: 编写向量库测试**

在 `tests/test_knowledge.py` 中追加：

```python
from src.knowledge.vector_store import VectorStore
import tempfile
import os


def test_vector_store_add_and_query():
    """测试添加和查询"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(persist_dir=tmpdir, collection_name="test")
        
        # Mock embedding
        store._embedding = type("MockEmbedding", (), {
            "encode": lambda self, text: [0.1] * 768,
            "encode_batch": lambda self, texts: [[0.1] * 768 for _ in texts],
        })()
        
        store.add("test_1", "今天天气真好", {"category": "sentiment"})
        results = store.query("天气", n_results=1)
        
        assert len(results) > 0
        assert results[0]["id"] == "test_1"


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
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_knowledge.py -v -k "vector_store"
```

- [ ] **Step 3: 实现向量库**

```python
# src/knowledge/vector_store.py
from src.knowledge.embedding import EmbeddingModel


class VectorStore:
    """ChromaDB 向量库封装"""

    def __init__(self, persist_dir: str = "./data/embeddings", collection_name: str = "default"):
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._client = None
        self._collection = None
        self._embedding = EmbeddingModel()

    def _get_collection(self):
        if self._collection is None:
            import chromadb
            self._client = chromadb.PersistentClient(path=self._persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add(self, doc_id: str, text: str, metadata: dict | None = None) -> None:
        """添加文档"""
        collection = self._get_collection()
        embedding = self._embedding.encode(text)
        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata or {}],
        )

    def add_batch(self, ids: list[str], texts: list[str], metadatas: list[dict] | None = None) -> None:
        """批量添加"""
        collection = self._get_collection()
        embeddings = self._embedding.encode_batch(texts)
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas or [{}] * len(ids),
        )

    def query(self, text: str, n_results: int = 5, where: dict | None = None) -> list[dict]:
        """查询相似文档"""
        collection = self._get_collection()
        embedding = self._embedding.encode(text)
        
        kwargs = {
            "query_embeddings": [embedding],
            "n_results": min(n_results, collection.count()),
        }
        if where:
            kwargs["where"] = where

        results = collection.query(**kwargs)
        
        items = []
        if results and results["ids"]:
            for i in range(len(results["ids"][0])):
                items.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i] if results.get("documents") else "",
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                    "distance": results["distances"][0][i] if results.get("distances") else 0.0,
                })
        return items

    def delete(self, doc_id: str) -> None:
        """删除文档"""
        collection = self._get_collection()
        collection.delete(ids=[doc_id])

    def count(self) -> int:
        """文档数量"""
        collection = self._get_collection()
        return collection.count()
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_knowledge.py -v -k "vector_store"
```

- [ ] **Step 5: 提交**

```bash
git add src/knowledge/vector_store.py tests/test_knowledge.py
git commit -m "feat: add ChromaDB vector store with metadata filtering"
```

---

## Task 18: 知识图谱（NetworkX）

**Files:**
- Create: `src/knowledge/knowledge_graph.py`
- Modify: `tests/test_knowledge.py`

- [ ] **Step 1: 编写知识图谱测试**

在 `tests/test_knowledge.py` 中追加：

```python
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
    import tempfile
    import os
    
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
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_knowledge.py -v -k "kg"
```

- [ ] **Step 3: 实现知识图谱**

```python
# src/knowledge/knowledge_graph.py
import json
import networkx as nx


class KnowledgeGraph:
    """知识图谱 - NetworkX 实现"""

    def __init__(self):
        self._graph = nx.DiGraph()

    def add_entity(self, name: str, entity_type: str, properties: dict | None = None) -> None:
        """添加实体"""
        self._graph.add_node(name, type=entity_type, **(properties or {}))

    def add_relation(self, source: str, target: str, relation_type: str, properties: dict | None = None) -> None:
        """添加关系"""
        self._graph.add_edge(source, target, relation=relation_type, **(properties or {}))

    def has_entity(self, name: str) -> bool:
        """检查实体是否存在"""
        return self._graph.has_node(name)

    def get_entity(self, name: str) -> dict | None:
        """获取实体信息"""
        if not self.has_entity(name):
            return None
        data = self._graph.nodes[name]
        return {"name": name, **data}

    def get_neighbors(self, name: str, relation: str | None = None) -> list[tuple[str, dict]]:
        """获取邻居节点"""
        if not self.has_entity(name):
            return []
        neighbors = []
        for _, target, data in self._graph.out_edges(name, data=True):
            if relation is None or data.get("relation") == relation:
                neighbors.append((target, data))
        return neighbors

    def get_entities_by_type(self, entity_type: str) -> list[dict]:
        """按类型查询实体"""
        return [
            {"name": node, **data}
            for node, data in self._graph.nodes(data=True)
            if data.get("type") == entity_type
        ]

    def find_path(self, source: str, target: str) -> list[str]:
        """查找路径"""
        try:
            return nx.shortest_path(self._graph, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def save(self, path: str) -> None:
        """序列化到 JSON"""
        data = {
            "nodes": [
                {"name": node, **attrs}
                for node, attrs in self._graph.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v, **attrs}
                for u, v, attrs in self._graph.edges(data=True)
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        """从 JSON 加载"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self._graph.clear()
        for node in data["nodes"]:
            name = node.pop("name")
            self.add_entity(name, node.pop("type", "unknown"), node)
        for edge in data["edges"]:
            self.add_relation(
                edge["source"], edge["target"],
                edge.pop("relation", "related"),
                edge,
            )
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_knowledge.py -v -k "kg"
```

- [ ] **Step 5: 提交**

```bash
git add src/knowledge/knowledge_graph.py tests/test_knowledge.py
git commit -m "feat: add NetworkX knowledge graph with save/load"
```

---

## Task 19: 案例库

**Files:**
- Create: `src/knowledge/case_library.py`
- Create: `data/cases/rhetoric_cases.json`
- Create: `data/cases/fallacy_cases.json`
- Modify: `tests/test_knowledge.py`

- [ ] **Step 1: 编写案例库测试**

在 `tests/test_knowledge.py` 中追加：

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_knowledge.py -v -k "case_library"
```

- [ ] **Step 3: 创建案例数据**

```json
// data/cases/rhetoric_cases.json
[
  {
    "id": "rhet_001",
    "type": "rhetoric",
    "subtype": "simile",
    "text": "他的心像冰一样冷",
    "analysis": "明喻，将'心'比作'冰'，表达冷漠无情",
    "keywords": ["比喻", "明喻", "冰", "冷"]
  },
  {
    "id": "rhet_002",
    "type": "rhetoric",
    "subtype": "rhetorical_question",
    "text": "这难道不是显而易见的吗？",
    "analysis": "反问，用疑问形式表达确定观点，加强语气",
    "keywords": ["反问", "难道", "显而易见"]
  },
  {
    "id": "rhet_003",
    "type": "rhetoric",
    "subtype": "exaggeration",
    "text": "我等了一万年",
    "analysis": "夸张，用极大数字表达等待时间之长",
    "keywords": ["夸张", "一万年", "等待"]
  },
  {
    "id": "rhet_004",
    "type": "rhetoric",
    "subtype": "parallelism",
    "text": "学习使人进步，学习使人聪明，学习使人强大",
    "analysis": "排比，三个相同句式并列，增强表达气势",
    "keywords": ["排比", "学习", "使人"]
  }
]
```

```json
// data/cases/fallacy_cases.json
[
  {
    "id": "fall_001",
    "type": "fallacy",
    "subtype": "straw_man",
    "text": "你支持环保？那你是不是想让我们都回到原始社会？",
    "analysis": "稻草人谬误，歪曲对方立场后攻击",
    "keywords": ["稻草人", "歪曲", "环保"]
  },
  {
    "id": "fall_002",
    "type": "fallacy",
    "subtype": "slippery_slope",
    "text": "如果允许学生用手机，他们就会沉迷游戏，然后成绩下降，最后毁了一生",
    "analysis": "滑坡谬误，无根据地推导出极端后果",
    "keywords": ["滑坡", "极端", "连锁"]
  },
  {
    "id": "fall_003",
    "type": "fallacy",
    "subtype": "appeal_to_authority",
    "text": "爱因斯坦说过想象力比知识重要，所以不需要学习知识",
    "analysis": "诉诸权威谬误，断章取义名人名言",
    "keywords": ["诉诸权威", "爱因斯坦", "断章取义"]
  },
  {
    "id": "fall_004",
    "type": "fallacy",
    "subtype": "ad_hominem",
    "text": "你一个没结过婚的人，有什么资格谈婚姻观？",
    "analysis": "人身攻击谬误，攻击说话者而非论点",
    "keywords": ["人身攻击", "资格", "婚姻"]
  }
]
```

- [ ] **Step 4: 实现案例库**

```python
# src/knowledge/case_library.py
import json
import os


class CaseLibrary:
    """案例库 - JSON 文件 + 关键词检索"""

    def __init__(self, data_dir: str = "data/cases"):
        self._data_dir = data_dir
        self._cases: list[dict] = []
        self._load()

    def _load(self) -> None:
        """加载所有案例文件"""
        if not os.path.exists(self._data_dir):
            return
        for filename in os.listdir(self._data_dir):
            if filename.endswith(".json"):
                path = os.path.join(self._data_dir, filename)
                with open(path, "r", encoding="utf-8") as f:
                    cases = json.load(f)
                    self._cases.extend(cases)

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """关键词搜索"""
        results = []
        query_lower = query.lower()
        for case in self._cases:
            keywords = case.get("keywords", [])
            text = case.get("text", "")
            if any(kw in query_lower or query_lower in kw for kw in keywords):
                results.append(case)
            elif query_lower in text.lower():
                results.append(case)
        return results[:limit]

    def get_by_type(self, case_type: str) -> list[dict]:
        """按类型获取"""
        return [c for c in self._cases if c.get("type") == case_type]

    def get_by_subtype(self, subtype: str) -> list[dict]:
        """按子类型获取"""
        return [c for c in self._cases if c.get("subtype") == subtype]

    def get_all(self) -> list[dict]:
        """获取所有案例"""
        return self._cases

    def count(self) -> int:
        """案例数量"""
        return len(self._cases)
```

- [ ] **Step 5: 运行测试确认通过**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_knowledge.py -v -k "case_library"
```

- [ ] **Step 6: 提交**

```bash
git add src/knowledge/case_library.py data/cases/ tests/test_knowledge.py
git commit -m "feat: add case library with rhetoric and fallacy cases"
```

---

## Task 20: 理论文献库

**Files:**
- Create: `data/literature/theories.json`
- Modify: `tests/test_knowledge.py`

- [ ] **Step 1: 编写理论库测试**

在 `tests/test_knowledge.py` 中追加：

```python
from src.knowledge.case_library import CaseLibrary


def test_theory_library_load():
    """测试加载理论库"""
    lib = CaseLibrary("data/literature")
    assert lib.count() > 0


def test_theory_search_speech_act():
    """测试搜索言语行为理论"""
    lib = CaseLibrary("data/literature")
    results = lib.search("言语行为")
    assert len(results) > 0
    assert any("言语行为" in r.get("name", "") for r in results)
```

- [ ] **Step 2: 创建理论数据**

```json
// data/literature/theories.json
[
  {
    "id": "theory_001",
    "type": "theory",
    "name": "言语行为理论",
    "founders": ["Austin", "Searle"],
    "year": 1962,
    "description": "语言不仅描述世界，还执行行为。分为言内行为、言外行为、言后行为。",
    "keywords": ["言语行为", "言外行为", "Austin", "Searle", "施为句"],
    "applications": ["识别说话者的言外之意", "分析命令、请求、承诺等"]
  },
  {
    "id": "theory_002",
    "type": "theory",
    "name": "会话含义理论",
    "founders": ["Grice"],
    "year": 1975,
    "description": "通过合作原则及其准则（量、质、关系、方式）推导隐含意义。",
    "keywords": ["会话含义", "合作原则", "Grice", "隐含意义", "准则"],
    "applications": ["分析违反准则时的隐含意义", "识别讽刺、暗示"]
  },
  {
    "id": "theory_003",
    "type": "theory",
    "name": "礼貌策略",
    "founders": ["Brown", " Levinson"],
    "year": 1987,
    "description": "面子理论，分析威胁面子的行为和补偿策略。",
    "keywords": ["礼貌", "面子", "Brown", "Levinson", "面子威胁"],
    "applications": ["分析礼貌策略", "识别面子威胁行为"]
  },
  {
    "id": "theory_004",
    "type": "theory",
    "name": "儒家面子理论",
    "founders": ["胡先缙", "黄光国"],
    "year": 1944,
    "description": "中国本土面子观：脸（道德面子）vs 面子（社会面子）。",
    "keywords": ["面子", "脸", "胡先缙", "黄光国", "儒家", "中国"],
    "applications": ["分析中国社会的面子博弈", "高语境沟通分析"]
  },
  {
    "id": "theory_005",
    "type": "theory",
    "name": "高语境沟通",
    "founders": ["Hall"],
    "year": 1976,
    "description": "中国文化语境下，大量信息通过语境而非语言本身传递。",
    "keywords": ["高语境", "Hall", "中国文化", "隐含信息"],
    "applications": ["分析中国式含蓄表达", "解读言外之意"]
  },
  {
    "id": "theory_006",
    "type": "theory",
    "name": "权力距离",
    "founders": ["Hofstede"],
    "year": 1980,
    "description": "语言中的敬语、称呼策略反映权力关系认知。",
    "keywords": ["权力距离", "Hofstede", "敬语", "称呼"],
    "applications": ["分析语言中的权力关系", "敬语使用分析"]
  },
  {
    "id": "theory_007",
    "type": "theory",
    "name": "批判性话语分析",
    "founders": ["van Dijk", "Fairclough"],
    "year": 1993,
    "description": "分析话语中的权力关系、意识形态和社会结构。",
    "keywords": ["批判性话语", "van Dijk", "Fairclough", "权力", "意识形态"],
    "applications": ["媒体话语分析", "政治话语分析"]
  },
  {
    "id": "theory_008",
    "type": "theory",
    "name": "系统功能语言学",
    "founders": ["Halliday"],
    "year": 1994,
    "description": "语言的三大元功能：概念功能、人际功能、语篇功能。",
    "keywords": ["系统功能", "Halliday", "元功能", "概念", "人际"],
    "applications": ["分析语言的人际功能", "语篇分析"]
  }
]
```

- [ ] **Step 3: 运行测试确认通过**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_knowledge.py -v -k "theory"
```

- [ ] **Step 4: 提交**

```bash
git add data/literature/ tests/test_knowledge.py
git commit -m "feat: add theory literature library with 8 core theories"
```

---

## Task 21: 知识图谱数据填充

**Files:**
- Create: `data/graph/psychology_graph.json`
- Modify: `tests/test_knowledge.py`

- [ ] **Step 1: 编写图谱数据测试**

在 `tests/test_knowledge.py` 中追加：

```python
def test_knowledge_graph_load_data():
    """测试加载知识图谱数据"""
    from src.knowledge.knowledge_graph import KnowledgeGraph
    kg = KnowledgeGraph()
    kg.load("data/graph/psychology_graph.json")
    assert kg.has_entity("言语行为理论")
    assert kg.has_entity("会话含义理论")


def test_knowledge_graph_relations():
    """测试图谱关系"""
    from src.knowledge.knowledge_graph import KnowledgeGraph
    kg = KnowledgeGraph()
    kg.load("data/graph/psychology_graph.json")
    
    neighbors = kg.get_neighbors("言语行为理论")
    assert len(neighbors) > 0
```

- [ ] **Step 2: 创建图谱数据**

```json
// data/graph/psychology_graph.json
{
  "nodes": [
    {"name": "言语行为理论", "type": "theory", "founders": "Austin/Searle"},
    {"name": "会话含义理论", "type": "theory", "founders": "Grice"},
    {"name": "礼貌策略", "type": "theory", "founders": "Brown/Levinson"},
    {"name": "儒家面子理论", "type": "theory", "founders": "胡先缙/黄光国"},
    {"name": "高语境沟通", "type": "theory", "founders": "Hall"},
    {"name": "权力距离", "type": "theory", "founders": "Hofstede"},
    {"name": "断言", "type": "speech_act"},
    {"name": "指令", "type": "speech_act"},
    {"name": "承诺", "type": "speech_act"},
    {"name": "表达", "type": "speech_act"},
    {"name": "比喻", "type": "rhetoric"},
    {"name": "反问", "type": "rhetoric"},
    {"name": "夸张", "type": "rhetoric"},
    {"name": "排比", "type": "rhetoric"},
    {"name": "面子威胁", "type": "concept"},
    {"name": "礼貌补偿", "type": "concept"},
    {"name": "隐含意义", "type": "concept"},
    {"name": "讽刺", "type": "concept"}
  ],
  "edges": [
    {"source": "言语行为理论", "target": "断言", "relation": "defines"},
    {"source": "言语行为理论", "target": "指令", "relation": "defines"},
    {"source": "言语行为理论", "target": "承诺", "relation": "defines"},
    {"source": "言语行为理论", "target": "表达", "relation": "defines"},
    {"source": "会话含义理论", "target": "隐含意义", "relation": "explains"},
    {"source": "会话含义理论", "target": "讽刺", "relation": "explains"},
    {"source": "礼貌策略", "target": "面子威胁", "relation": "defines"},
    {"source": "礼貌策略", "target": "礼貌补偿", "relation": "defines"},
    {"source": "儒家面子理论", "target": "面子威胁", "relation": "extends"},
    {"source": "高语境沟通", "target": "隐含意义", "relation": "related"},
    {"source": "权力距离", "target": "礼貌策略", "relation": "related"}
  ]
}
```

- [ ] **Step 3: 运行测试确认通过**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_knowledge.py -v -k "knowledge_graph"
```

- [ ] **Step 4: 提交**

```bash
git add data/graph/ tests/test_knowledge.py
git commit -m "feat: add knowledge graph data with theories and relations"
```

---

## Task 22: 统一检索接口

**Files:**
- Create: `src/knowledge/retriever.py`
- Modify: `tests/test_knowledge.py`

- [ ] **Step 1: 编写统一检索测试**

在 `tests/test_knowledge.py` 中追加：

```python
from src.knowledge.retriever import KnowledgeRetriever


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
    from unittest.mock import MagicMock
    
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
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_knowledge.py -v -k "retriever"
```

- [ ] **Step 3: 实现统一检索接口**

```python
# src/knowledge/retriever.py
from src.knowledge.vector_store import VectorStore
from src.knowledge.knowledge_graph import KnowledgeGraph
from src.knowledge.case_library import CaseLibrary


class KnowledgeRetriever:
    """统一检索接口 - 融合向量、图谱、案例三种检索"""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
        case_library: CaseLibrary | None = None,
    ):
        self._vector_store = vector_store
        self._knowledge_graph = knowledge_graph
        self._case_library = case_library

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        include_vector: bool = True,
        include_graph: bool = True,
        include_cases: bool = True,
    ) -> dict:
        """融合检索"""
        results = {
            "vector_results": [],
            "graph_results": [],
            "case_results": [],
        }

        if include_vector and self._vector_store:
            try:
                results["vector_results"] = self._vector_store.query(query, n_results=n_results)
            except Exception:
                pass

        if include_graph and self._knowledge_graph:
            try:
                # 尝试用查询词作为实体名查找邻居
                neighbors = self._knowledge_graph.get_neighbors(query)
                results["graph_results"] = [
                    {"entity": n[0], "relation": n[1].get("relation", "")}
                    for n in neighbors
                ]
            except Exception:
                pass

        if include_cases and self._case_library:
            try:
                results["case_results"] = self._case_library.search(query, limit=n_results)
            except Exception:
                pass

        return results

    def get_context_string(self, query: str, n_results: int = 3) -> str:
        """获取格式化的知识上下文（供 Agent 使用）"""
        results = self.retrieve(query, n_results=n_results)
        
        parts = []
        
        if results["vector_results"]:
            parts.append("## 相关知识")
            for r in results["vector_results"][:n_results]:
                parts.append(f"- {r.get('text', '')}")
        
        if results["case_results"]:
            parts.append("\n## 相关案例")
            for c in results["case_results"][:n_results]:
                parts.append(f"- [{c.get('subtype', '')}] {c.get('text', '')}: {c.get('analysis', '')}")
        
        if results["graph_results"]:
            parts.append("\n## 相关概念")
            for g in results["graph_results"][:n_results]:
                parts.append(f"- {g['entity']} ({g['relation']})")
        
        return "\n".join(parts) if parts else ""
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_knowledge.py -v -k "retriever"
```

- [ ] **Step 5: 提交**

```bash
git add src/knowledge/retriever.py tests/test_knowledge.py
git commit -m "feat: add unified knowledge retriever (vector + graph + case fusion)"
```

---

## Task 23: 阶段三总结

- [ ] **Step 1: 运行全部测试**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/ -v --tb=short
```

预期：全部 PASS

- [ ] **Step 2: 更新知识模块 __init__.py**

```python
# src/knowledge/__init__.py
from src.knowledge.embedding import EmbeddingModel
from src.knowledge.vector_store import VectorStore
from src.knowledge.knowledge_graph import KnowledgeGraph
from src.knowledge.case_library import CaseLibrary
from src.knowledge.retriever import KnowledgeRetriever

__all__ = [
    "EmbeddingModel",
    "VectorStore",
    "KnowledgeGraph",
    "CaseLibrary",
    "KnowledgeRetriever",
]
```

- [ ] **Step 3: 更新 README.md**

将阶段三标记为已完成，更新项目结构。

- [ ] **Step 4: 最终提交**

```bash
git add src/knowledge/__init__.py README.md
git commit -m "docs: update progress - Phase 3 complete"
```
