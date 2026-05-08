# 阶段二：NLP 管道 + 文本解析 Agent 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建中文 NLP 处理管道（HanLP + jieba + transformers），实现文本解析 Agent 的 5 个分析维度，并建立评测基准数据集

**Architecture:** NLP 管道采用分层设计：HanLP 为主力（深度学习句法依存、语义角色标注），jieba 为快速后备（分词 + 自定义词典），transformers 提供细粒度情感分析。文本解析 Agent 调用 NLP 管道，从 5 个维度（句法、词汇、语气、修辞、话语标记）输出结构化分析。

**Tech Stack:** HanLP v2/v3, jieba, transformers (chinese-roberta-wwm-ext), pytest

**依赖：** 阶段一已完成（BaseAgent, LLM 客户端, 配置, 协调器）

---

## 文件结构

```
src/
├── nlp/
│   ├── __init__.py              # NLP 管道统一导出
│   ├── tokenizer.py             # 分词（HanLP + jieba 后备）
│   ├── syntax.py                # 句法分析（HanLP 依存句法）
│   ├── sentiment.py             # 情感分析（transformers 中文模型）
│   └── rhetoric.py              # 修辞识别（规则 + LLM 辅助）
├── agents/
│   └── text_analyst.py          # 文本解析 Agent
├── evaluation/
│   ├── __init__.py
│   ├── schema.py                # 评测数据集 JSON Schema
│   └── scorer.py                # 评测评分脚本
data/
└── golden/
    └── text_analysis_benchmark.json  # 评测基准数据集
tests/
├── test_nlp.py                  # NLP 管道测试
├── test_text_analyst.py         # 文本解析 Agent 测试
└── test_evaluation.py           # 评测框架测试
```

---

## Task 9: HanLP 集成（分词、依存句法、语义角色标注）

**Files:**
- Create: `src/nlp/tokenizer.py`
- Create: `src/nlp/syntax.py`
- Create: `tests/test_nlp.py`

- [ ] **Step 1: 编写分词测试**

```python
# tests/test_nlp.py
import pytest
from src.nlp.tokenizer import Tokenizer


def test_tokenizer_basic():
    """测试基本分词"""
    tok = Tokenizer()
    tokens = tok.tokenize("今天天气真不错")
    assert len(tokens) > 0
    assert isinstance(tokens, list)
    assert all(isinstance(t, str) for t in tokens)


def test_tokenizer_sentence_split():
    """测试断句"""
    tok = Tokenizer()
    sentences = tok.split_sentences("你好。今天天气不错！我们去散步吧？")
    assert len(sentences) == 3
    assert "你好" in sentences[0]


def test_tokenizer_pos_tags():
    """测试词性标注"""
    tok = Tokenizer()
    result = tok.tokenize_with_pos("我喜欢苹果")
    assert len(result) > 0
    assert all(isinstance(item, tuple) and len(item) == 2 for item in result)
    # 应该有名词和动词
    pos_tags = [tag for _, tag in result]
    assert any("n" in tag for tag in pos_tags)  # 名词


def test_syntax_dependency():
    """测试依存句法分析"""
    from src.nlp.syntax import SyntaxAnalyzer
    analyzer = SyntaxAnalyzer()
    deps = analyzer.parse_dependencies("小明吃苹果")
    assert len(deps) > 0
    assert all("head" in d and "relation" in d for d in deps)


def test_syntax_srl():
    """测试语义角色标注"""
    from src.nlp.syntax import SyntaxAnalyzer
    analyzer = SyntaxAnalyzer()
    srl = analyzer.semantic_role_labeling("小明在公园里吃苹果")
    assert isinstance(srl, list)
    # 应该能识别施事者和受事者
    assert len(srl) > 0
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_nlp.py -v
```

预期：FAIL（ImportError）

- [ ] **Step 3: 实现分词模块**

```python
# src/nlp/tokenizer.py
import re
from dataclasses import dataclass

import hanlp
import jieba


class Tokenizer:
    """分词器 - HanLP 主力 + jieba 后备"""

    def __init__(self, use_hanlp: bool = True):
        self._use_hanlp = use_hanlp
        self._hanlp_tokenizer = None
        self._hanlp_pos = None

    def _get_hanlp_tokenizer(self):
        if self._hanlp_tokenizer is None:
            self._hanlp_tokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
        return self._hanlp_tokenizer

    def _get_hanlp_pos(self):
        if self._hanlp_pos is None:
            self._hanlp_pos = hanlp.load(hanlp.pretrained.pos.CTB9_POS_ELECTRA_SMALL)
        return self._hanlp_pos

    def tokenize(self, text: str) -> list[str]:
        """分词"""
        if self._use_hanlp:
            try:
                tokenizer = self._get_hanlp_tokenizer()
                return tokenizer(text)
            except Exception:
                pass
        return list(jieba.cut(text))

    def tokenize_with_pos(self, text: str) -> list[tuple[str, str]]:
        """分词 + 词性标注"""
        if self._use_hanlp:
            try:
                tokens = self.tokenize(text)
                pos_tagger = self._get_hanlp_pos()
                pos_tags = pos_tagger(tokens)
                return list(zip(tokens, pos_tags))
            except Exception:
                pass
        return list(jieba.posseg.cut(text))

    def split_sentences(self, text: str) -> list[str]:
        """断句"""
        # 中文标点断句
        sentences = re.split(r'[。！？!?]', text)
        return [s.strip() for s in sentences if s.strip()]
```

- [ ] **Step 4: 实现句法分析模块**

```python
# src/nlp/syntax.py
from dataclasses import dataclass

import hanlp


@dataclass
class DependencyArc:
    """依存弧"""
    id: int
    word: str
    head: int
    relation: str


@dataclass
class SRLFrame:
    """语义角色标注帧"""
    predicate: str
    arguments: list[dict]


class SyntaxAnalyzer:
    """句法分析器 - HanLP 依存句法 + 语义角色标注"""

    def __init__(self):
        self._dep_parser = None
        self._srl = None
        self._tokenizer = None

    def _get_tokenizer(self):
        if self._tokenizer is None:
            import hanlp
            self._tokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
        return self._tokenizer

    def _get_dep_parser(self):
        if self._dep_parser is None:
            self._dep_parser = hanlp.load(hanlp.pretrained.dep.CTB9_DEP_ELECTRA_SMALL)
        return self._dep_parser

    def _get_srl(self):
        if self._srl is None:
            self._srl = hanlp.load(hanlp.pretrained.srl.SRL_ELECTRA_SMALL_ZH)
        return self._srl

    def parse_dependencies(self, text: str) -> list[dict]:
        """依存句法分析"""
        tokenizer = self._get_tokenizer()
        tokens = tokenizer(text)
        parser = self._get_dep_parser()
        arcs = parser(tokens)

        result = []
        for i, (token, arc) in enumerate(zip(tokens, arcs)):
            result.append({
                "id": i,
                "word": token,
                "head": arc["head"] if isinstance(arc, dict) else 0,
                "relation": arc["dep"] if isinstance(arc, dict) else str(arc),
            })
        return result

    def semantic_role_labeling(self, text: str) -> list[dict]:
        """语义角色标注"""
        tokenizer = self._get_tokenizer()
        tokens = tokenizer(text)
        srl = self._get_srl()
        frames = srl(tokens)

        result = []
        if isinstance(frames, list):
            for frame in frames:
                if isinstance(frame, dict):
                    result.append({
                        "predicate": frame.get("predicate", ""),
                        "arguments": frame.get("arguments", []),
                    })
                elif isinstance(frame, (list, tuple)) and len(frame) >= 2:
                    result.append({
                        "predicate": str(frame[0]),
                        "arguments": [{"role": str(a[0]), "text": str(a[1])} if isinstance(a, (list, tuple)) else {"text": str(a)} for a in frame[1:]],
                    })
        return result
```

- [ ] **Step 5: 运行测试确认通过**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_nlp.py -v -k "tokenizer or syntax"
```

预期：全部 PASS（首次运行会下载模型，需要网络）

- [ ] **Step 6: 提交**

```bash
git add src/nlp/tokenizer.py src/nlp/syntax.py src/nlp/__init__.py tests/test_nlp.py
git commit -m "feat: add HanLP integration for tokenization and syntax analysis"
```

---

## Task 10: jieba 后备分词 + 自定义词典

**Files:**
- Modify: `src/nlp/tokenizer.py`
- Modify: `tests/test_nlp.py`

- [ ] **Step 1: 编写自定义词典测试**

在 `tests/test_nlp.py` 中追加：

```python
def test_tokenizer_custom_dict():
    """测试自定义词典"""
    tok = Tokenizer(use_hanlp=False)  # 强制使用 jieba
    # 添加自定义词
    tok.add_custom_words(["语言心理学", "话语分析", "多智能体"])
    tokens = tok.tokenize("语言心理学是一门有趣的学科")
    assert "语言心理学" in tokens


def test_tokenizer_fallback():
    """测试 HanLP 不可用时的 jieba 后备"""
    tok = Tokenizer(use_hanlp=False)
    tokens = tokenize_with_fallback(tok, "今天天气不错")
    assert len(tokens) > 0


def tokenize_with_fallback(tok, text):
    """辅助函数：模拟后备逻辑"""
    return tok.tokenize(text)


def test_tokenizer_empty_text():
    """测试空文本"""
    tok = Tokenizer(use_hanlp=False)
    tokens = tok.tokenize("")
    assert tokens == []


def test_tokenizer_punctuation_handling():
    """测试标点处理"""
    tok = Tokenizer(use_hanlp=False)
    tokens = tok.tokenize("你好，世界！")
    assert len(tokens) > 0
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_nlp.py -v -k "custom_dict or fallback or empty or punctuation"
```

预期：FAIL（add_custom_words 方法不存在）

- [ ] **Step 3: 实现自定义词典功能**

在 `src/nlp/tokenizer.py` 的 `Tokenizer` 类中追加：

```python
    def add_custom_words(self, words: list[str]) -> None:
        """添加自定义词到 jieba 词典"""
        for word in words:
            jieba.add_word(word)

    def load_custom_dict(self, dict_path: str) -> None:
        """加载自定义词典文件"""
        jieba.load_userdict(dict_path)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_nlp.py -v
```

预期：全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/nlp/tokenizer.py tests/test_nlp.py
git commit -m "feat: add jieba fallback and custom dictionary support"
```

---

## Task 11: transformers 中文情感分析模型

**Files:**
- Create: `src/nlp/sentiment.py`
- Modify: `tests/test_nlp.py`

- [ ] **Step 1: 编写情感分析测试**

在 `tests/test_nlp.py` 中追加：

```python
from src.nlp.sentiment import SentimentAnalyzer


def test_sentiment_positive():
    """测试正面情感"""
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze("今天心情非常好，一切都很顺利")
    assert result["label"] in ["positive", "POS"]
    assert result["score"] > 0.5


def test_sentiment_negative():
    """测试负面情感"""
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze("这件事让我非常失望和难过")
    assert result["label"] in ["negative", "NEG"]
    assert result["score"] > 0.5


def test_sentiment_neutral():
    """测试中性情感"""
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze("今天是星期三")
    assert "label" in result
    assert "score" in result


def test_sentiment_detail():
    """测试细粒度情感分析"""
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze_detail("虽然有点累，但是很开心能完成这个项目")
    assert "positive_score" in result
    assert "negative_score" in result
    assert "dominant_emotion" in result
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_nlp.py -v -k "sentiment"
```

预期：FAIL（ImportError）

- [ ] **Step 3: 实现情感分析模块**

```python
# src/nlp/sentiment.py
from transformers import pipeline


class SentimentAnalyzer:
    """中文情感分析 - 基于 transformers 预训练模型"""

    def __init__(self, model_name: str = "uer/roberta-base-finetuned-chinanews-chinese"):
        self._model_name = model_name
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            self._pipeline = pipeline(
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
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_nlp.py -v -k "sentiment"
```

预期：全部 PASS（首次运行会下载模型）

- [ ] **Step 5: 提交**

```bash
git add src/nlp/sentiment.py tests/test_nlp.py
git commit -m "feat: add transformers-based Chinese sentiment analysis"
```

---

## Task 12: 修辞识别模块

**Files:**
- Create: `src/nlp/rhetoric.py`
- Modify: `tests/test_nlp.py`

- [ ] **Step 1: 编写修辞识别测试**

在 `tests/test_nlp.py` 中追加：

```python
from src.nlp.rhetoric import RhetoricDetector


def test_detect_metaphor():
    """测试比喻识别"""
    detector = RhetoricDetector()
    results = detector.detect("他的心像冰一样冷")
    types = [r["type"] for r in results]
    assert "metaphor" in types or "simile" in types


def test_detect_rhetorical_question():
    """测试反问识别"""
    detector = RhetoricDetector()
    results = detector.detect("这难道不是显而易见的吗？")
    types = [r["type"] for r in results]
    assert "rhetorical_question" in types


def test_detect_exaggeration():
    """测试夸张识别"""
    detector = RhetoricDetector()
    results = detector.detect("我等了一万年")
    types = [r["type"] for r in results]
    assert "exaggeration" in types


def test_detect_parallelism():
    """测试排比识别"""
    detector = RhetoricDetector()
    text = "学习使人进步，学习使人聪明，学习使人强大"
    results = detector.detect(text)
    types = [r["type"] for r in results]
    assert "parallelism" in types


def test_no_rhetoric():
    """测试无修辞的普通文本"""
    detector = RhetoricDetector()
    results = detector.detect("今天是2026年5月8日")
    assert len(results) == 0 or all(r["confidence"] < 0.5 for r in results)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_nlp.py -v -k "rhetoric or metaphor or rhetorical or exaggeration or parallelism or no_rhetoric"
```

预期：FAIL（ImportError）

- [ ] **Step 3: 实现修辞识别模块**

```python
# src/nlp/rhetoric.py
import re
from dataclasses import dataclass


@dataclass
class RhetoricResult:
    """修辞识别结果"""
    type: str
    text_span: str
    confidence: float
    description: str


class RhetoricDetector:
    """修辞手法检测器 - 规则 + 模式匹配"""

    # 比喻模式
    METAPHOR_PATTERNS = [
        (r'像.{1,10}一样', 'simile', '明喻'),
        (r'如.{1,10}一般', 'simile', '明喻'),
        (r'仿佛.{1,10}', 'simile', '明喻'),
        (r'是.{0,5}的.{0,5}', 'metaphor', '暗喻'),
    ]

    # 反问模式
    RHETORICAL_QUESTION_PATTERNS = [
        (r'难道.{0,20}[吗？?]', 'rhetorical_question', '反问'),
        (r'怎么.{0,15}[呢？?]', 'rhetorical_question', '反问'),
        (r'岂.{0,15}[？?]', 'rhetorical_question', '反问'),
    ]

    # 夸张模式
    EXAGGERATION_PATTERNS = [
        (r'[一二三四五六七八九十百千万亿]+[年月日天秒分钟]', 'exaggeration', '夸张'),
        (r'极[了]', 'exaggeration', '夸张'),
        (r'万分', 'exaggeration', '夸张'),
    ]

    # 排比检测关键词
    PARALLELISM_MARKERS = ['，', '；']

    def detect(self, text: str) -> list[dict]:
        """检测修辞手法"""
        results = []

        # 比喻检测
        for pattern, rtype, desc in self.METAPHOR_PATTERNS:
            matches = re.finditer(pattern, text)
            for m in matches:
                results.append({
                    "type": rtype,
                    "text_span": m.group(),
                    "confidence": 0.7,
                    "description": desc,
                })

        # 反问检测
        for pattern, rtype, desc in self.RHETORICAL_QUESTION_PATTERNS:
            matches = re.finditer(pattern, text)
            for m in matches:
                results.append({
                    "type": rtype,
                    "text_span": m.group(),
                    "confidence": 0.8,
                    "description": desc,
                })

        # 夸张检测
        for pattern, rtype, desc in self.EXAGGERATION_PATTERNS:
            matches = re.finditer(pattern, text)
            for m in matches:
                results.append({
                    "type": rtype,
                    "text_span": m.group(),
                    "confidence": 0.6,
                    "description": desc,
                })

        # 排比检测（简单：重复句式）
        results.extend(self._detect_parallelism(text))

        return results

    def _detect_parallelism(self, text: str) -> list[dict]:
        """检测排比（基于重复句式）"""
        results = []
        # 用逗号或分号分割
        clauses = re.split(r'[，；,;]', text)
        if len(clauses) >= 3:
            # 检查是否有相似的句式开头
            prefixes = [c.strip()[:3] for c in clauses if len(c.strip()) >= 3]
            if len(prefixes) >= 3:
                from collections import Counter
                prefix_counts = Counter(prefixes)
                most_common_prefix, count = prefix_counts.most_common(1)[0]
                if count >= 3:
                    results.append({
                        "type": "parallelism",
                        "text_span": text[:50],
                        "confidence": min(0.5 + count * 0.1, 0.9),
                        "description": f"排比（重复句式 '{most_common_prefix}' 出现 {count} 次）",
                    })
        return results
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_nlp.py -v -k "rhetoric or metaphor or rhetorical or exaggeration or parallelism or no_rhetoric"
```

预期：全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/nlp/rhetoric.py tests/test_nlp.py
git commit -m "feat: add rhetoric detection module (metaphor, question, exaggeration, parallelism)"
```

---

## Task 13: 文本解析 Agent 完整实现

**Files:**
- Create: `src/agents/text_analyst.py`
- Create: `tests/test_text_analyst.py`

- [ ] **Step 1: 编写文本解析 Agent 测试**

```python
# tests/test_text_analyst.py
import pytest
from unittest.mock import patch, MagicMock
from src.agents.base import AnalysisContext, AnalysisDepth, TextFeatures
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
    ctx = AnalysisContext(
        text="小明吃了一个苹果。",
        depth=AnalysisDepth.STANDARD,
    )
    result = agent.analyze(ctx)
    assert result.agent_name == "text_analyst"
    assert "sentence_types" in result.analysis
    assert result.confidence > 0


def test_analyze_vocabulary():
    """测试词汇分析"""
    agent = TextAnalystAgent()
    ctx = AnalysisContext(
        text="这个问题基本上大概可能需要进一步研究",
        depth=AnalysisDepth.STANDARD,
    )
    result = agent.analyze(ctx)
    assert "formality_score" in result.analysis or "vague_words" in result.analysis


def test_analyze_modality():
    """测试语气分析"""
    agent = TextAnalystAgent()
    ctx = AnalysisContext(
        text="你必须完成这个任务，应该尽快提交",
        depth=AnalysisDepth.STANDARD,
    )
    result = agent.analyze(ctx)
    assert "modality" in result.analysis or "modal_verbs" in result.analysis


def test_analyze_rhetoric():
    """测试修辞识别"""
    agent = TextAnalystAgent()
    ctx = AnalysisContext(
        text="他的心像冰一样冷，时间仿佛停止了",
        depth=AnalysisDepth.DEEP,
    )
    result = agent.analyze(ctx)
    assert "rhetorical_devices" in result.analysis


def test_analyze_discourse_markers():
    """测试话语标记"""
    agent = TextAnalystAgent()
    ctx = AnalysisContext(
        text="虽然他很努力，但是结果并不理想。其实问题出在方法上",
        depth=AnalysisDepth.STANDARD,
    )
    result = agent.analyze(ctx)
    assert "discourse_markers" in result.analysis
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_text_analyst.py -v
```

预期：FAIL（ImportError）

- [ ] **Step 3: 实现文本解析 Agent**

```python
# src/agents/text_analyst.py
from src.agents.base import BaseAgent, AnalysisContext, AgentResult, AnalysisDepth, TextFeatures
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

    def analyze(self, context: AnalysisContext) -> AgentResult:
        """执行文本分析"""
        text = context.text
        depth = context.depth

        analysis = {}

        # 1. 句法结构分析
        analysis["sentence_types"] = self._analyze_sentence_types(text)
        analysis["dependency_parse"] = self._syntax.parse_dependencies(text)

        # 2. 词汇分析
        tokens = self._tokenizer.tokenize(text)
        pos_tags = self._tokenizer.tokenize_with_pos(text)
        analysis["token_count"] = len(tokens)
        analysis["vocabulary"] = self._analyze_vocabulary(tokens, pos_tags)

        # 3. 语气与情态分析
        analysis["modality"] = self._analyze_modality(text, pos_tags)

        # 4. 修辞手法（标准和深度模式）
        if depth in (AnalysisDepth.STANDARD, AnalysisDepth.DEEP):
            analysis["rhetorical_devices"] = self._rhetoric.detect(text)

        # 5. 话语标记
        analysis["discourse_markers"] = self._detect_discourse_markers(text)

        # 情感分析
        sentiment = self._sentiment.analyze(text)
        analysis["sentiment"] = sentiment

        # 计算置信度
        confidence = self._calculate_confidence(analysis)

        return AgentResult(
            agent_name=self.name,
            analysis=analysis,
            confidence=confidence,
            sources=["HanLP", "jieba", "transformers"],
        )

    def _analyze_sentence_types(self, text: str) -> dict:
        """分析句子类型"""
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
        """词汇分析"""
        # 模糊语检测
        vague_words = ["大概", "可能", "也许", "或许", "基本上", "差不多", "似乎"]
        found_vague = [t for t in tokens if t in vague_words]

        # 正式程度（基于词性分布）
        formal_pos = {"nr", "ns", "nt", "nz", "vn", "an"}
        formal_count = sum(1 for _, tag in pos_tags if tag in formal_pos)
        formality = formal_count / max(len(pos_tags), 1)

        return {
            "unique_tokens": len(set(tokens)),
            "vague_words": found_vague,
            "formality_score": round(formality, 3),
        }

    def _analyze_modality(self, text: str, pos_tags: list[tuple[str, str]]) -> dict:
        """语气与情态分析"""
        modal_verbs = {
            "必须": "necessity",
            "应该": "obligation",
            "可以": "permission",
            "能": "ability",
            "会": "possibility",
            "需要": "necessity",
            "想要": "desire",
        }
        found_modals = []
        for word, modal_type in modal_verbs.items():
            if word in text:
                found_modals.append({"word": word, "type": modal_type})

        return {
            "modal_verbs": found_modals,
            "modal_count": len(found_modals),
        }

    def _detect_discourse_markers(self, text: str) -> list[dict]:
        """检测话语标记"""
        markers = {
            "但是": "转折", "可是": "转折", "然而": "转折", "不过": "转折",
            "其实": "强调", "事实上": "强调", "说实话": "强调",
            "总之": "总结", "综上": "总结",
            "另外": "补充", "而且": "递进", "并且": "递进",
            "因为": "因果", "所以": "因果", "因此": "因果",
            "虽然": "让步", "尽管": "让步",
        }
        found = []
        for marker, category in markers.items():
            if marker in text:
                found.append({"marker": marker, "category": category})
        return found

    def _calculate_confidence(self, analysis: dict) -> float:
        """计算分析置信度"""
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
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_text_analyst.py -v
```

预期：全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/agents/text_analyst.py tests/test_text_analyst.py
git commit -m "feat: add TextAnalystAgent with 5 analysis dimensions"
```

---

## Task 14: 评测数据集结构与评分脚本

**Files:**
- Create: `src/evaluation/schema.py`
- Create: `src/evaluation/scorer.py`
- Create: `data/golden/text_analysis_benchmark.json`
- Create: `tests/test_evaluation.py`

- [ ] **Step 1: 编写评测框架测试**

```python
# tests/test_evaluation.py
import pytest
from src.evaluation.schema import BenchmarkItem, validate_benchmark
from src.evaluation.scorer import TextAnalysisScorer


def test_benchmark_item_creation():
    """测试评测项创建"""
    item = BenchmarkItem(
        id="test_001",
        text="今天天气真好",
        expected={
            "sentiment": "positive",
            "sentence_type": "declarative",
        },
    )
    assert item.id == "test_001"
    assert item.text == "今天天气真好"


def test_validate_benchmark():
    """测试基准数据集验证"""
    items = [
        BenchmarkItem(
            id="t1",
            text="你好",
            expected={"sentiment": "neutral"},
        ),
        BenchmarkItem(
            id="t2",
            text="这太棒了！",
            expected={"sentiment": "positive"},
        ),
    ]
    assert validate_benchmark(items) is True


def test_scorer_basic():
    """测试基础评分"""
    scorer = TextAnalysisScorer()
    predicted = {"sentiment": {"label": "positive"}}
    expected = {"sentiment": "positive"}
    score = scorer.score_item(predicted, expected)
    assert 0.0 <= score <= 1.0


def test_scorer_with_benchmark():
    """测试基准数据集评分"""
    scorer = TextAnalysisScorer()
    results = [
        {"predicted": {"sentiment": {"label": "positive"}}, "expected": {"sentiment": "positive"}},
        {"predicted": {"sentiment": {"label": "negative"}}, "expected": {"sentiment": "positive"}},
    ]
    avg_score = scorer.score_batch(results)
    assert 0.0 <= avg_score <= 1.0
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_evaluation.py -v
```

预期：FAIL（ImportError）

- [ ] **Step 3: 实现评测框架**

```python
# src/evaluation/schema.py
from dataclasses import dataclass, field
import json


@dataclass
class BenchmarkItem:
    """评测基准项"""
    id: str
    text: str
    expected: dict
    category: str = "general"
    difficulty: str = "medium"
    metadata: dict = field(default_factory=dict)


def validate_benchmark(items: list[BenchmarkItem]) -> bool:
    """验证基准数据集的有效性"""
    ids = set()
    for item in items:
        if not item.id:
            raise ValueError("Item ID cannot be empty")
        if item.id in ids:
            raise ValueError(f"Duplicate ID: {item.id}")
        ids.add(item.id)
        if not item.text:
            raise ValueError(f"Empty text for item {item.id}")
        if not item.expected:
            raise ValueError(f"Empty expected for item {item.id}")
    return True


def load_benchmark(path: str) -> list[BenchmarkItem]:
    """加载基准数据集"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [BenchmarkItem(**item) for item in data]
```

```python
# src/evaluation/scorer.py
class TextAnalysisScorer:
    """文本分析评测评分器"""

    def score_item(self, predicted: dict, expected: dict) -> float:
        """评分单个分析结果"""
        total = 0
        correct = 0

        for key, expected_value in expected.items():
            total += 1
            predicted_value = predicted.get(key)

            if isinstance(predicted_value, dict):
                # 嵌套结构（如 sentiment.label）
                if predicted_value.get("label") == expected_value:
                    correct += 1
            elif predicted_value == expected_value:
                correct += 1

        return correct / max(total, 1)

    def score_batch(self, results: list[dict]) -> float:
        """批量评分"""
        if not results:
            return 0.0
        scores = [
            self.score_item(r["predicted"], r["expected"])
            for r in results
        ]
        return sum(scores) / len(scores)
```

- [ ] **Step 4: 创建小型基准数据集**

```json
// data/golden/text_analysis_benchmark.json
[
  {
    "id": "sent_001",
    "text": "今天天气真好，心情非常愉快！",
    "expected": {"sentiment": "positive", "sentence_type": "exclamatory"},
    "category": "sentiment",
    "difficulty": "easy"
  },
  {
    "id": "sent_002",
    "text": "这件事让我非常失望，简直是浪费时间。",
    "expected": {"sentiment": "negative", "sentence_type": "declarative"},
    "category": "sentiment",
    "difficulty": "easy"
  },
  {
    "id": "rhet_001",
    "text": "他的心像冰一样冷。",
    "expected": {"rhetoric_type": "simile"},
    "category": "rhetoric",
    "difficulty": "medium"
  },
  {
    "id": "rhet_002",
    "text": "这难道不是显而易见的吗？",
    "expected": {"rhetoric_type": "rhetorical_question"},
    "category": "rhetoric",
    "difficulty": "medium"
  },
  {
    "id": "modal_001",
    "text": "你必须在明天之前完成这个任务。",
    "expected": {"modal_type": "necessity"},
    "category": "modality",
    "difficulty": "easy"
  },
  {
    "id": "disc_001",
    "text": "虽然他很努力，但是结果并不理想。",
    "expected": {"discourse_marker": "转折"},
    "category": "discourse",
    "difficulty": "easy"
  },
  {
    "id": "vague_001",
    "text": "这个问题基本上大概可能需要进一步研究。",
    "expected": {"has_vague_words": true},
    "category": "vocabulary",
    "difficulty": "medium"
  },
  {
    "id": "complex_001",
    "text": "说实话，虽然这个方案看起来不错，但是实际上存在很多问题，比如成本太高、周期太长，而且风险也很大。",
    "expected": {"discourse_marker": "转折", "has_vague_words": false},
    "category": "complex",
    "difficulty": "hard"
  }
]
```

- [ ] **Step 5: 运行测试确认通过**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_evaluation.py -v
```

预期：全部 PASS

- [ ] **Step 6: 提交**

```bash
git add src/evaluation/schema.py src/evaluation/scorer.py src/evaluation/__init__.py data/golden/text_analysis_benchmark.json tests/test_evaluation.py
git commit -m "feat: add evaluation framework with benchmark dataset and scorer"
```

---

## Task 15: 阶段二总结

- [ ] **Step 1: 运行全部测试**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/ -v --tb=short
```

预期：全部 PASS

- [ ] **Step 2: 更新 NLP __init__.py 导出**

```python
# src/nlp/__init__.py
from src.nlp.tokenizer import Tokenizer
from src.nlp.syntax import SyntaxAnalyzer
from src.nlp.sentiment import SentimentAnalyzer
from src.nlp.rhetoric import RhetoricDetector

__all__ = ["Tokenizer", "SyntaxAnalyzer", "SentimentAnalyzer", "RhetoricDetector"]
```

- [ ] **Step 3: 更新 README.md 开发进度**

在 README.md 中将阶段二标记为已完成。

- [ ] **Step 4: 最终提交**

```bash
git add src/nlp/__init__.py README.md
git commit -m "docs: update progress - Phase 2 complete"
```
