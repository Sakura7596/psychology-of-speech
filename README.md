# 语言心理学话语分析系统

基于语言心理学的多智能体话语分析系统，结合 RAG 技术和大语言模型，对自然语言文本进行深度分析。

## 快速开始

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 配置 API Key：
   ```bash
   cp .env.example .env
   # 编辑 .env 文件，填入 DEEPSEEK_API_KEY
   ```

3. 运行分析：
   ```bash
   python -m src.main "要分析的文本"
   python -m src.main "要分析的文本" deep    # 深度分析
   python -m src.main "要分析的文本" quick   # 快速分析
   ```

4. 运行测试：
   ```bash
   DEEPSEEK_API_KEY=test-key python -m pytest tests/ -v
   ```

## 项目结构

```
src/
├── main.py              # CLI 入口
├── config.py            # 配置管理（Pydantic Settings v2）
├── agents/
│   ├── base.py          # BaseAgent 抽象类 + 数据模型
│   └── orchestrator.py  # 总协调器（动态路由 + 置信度融合）
├── llm/
│   ├── client.py        # LLM 统一调用接口
│   ├── deepseek.py      # DeepSeek API 适配器
│   ├── prompts.py       # Prompt 模板管理
│   └── exceptions.py    # LLM 异常类
├── nlp/
│   ├── tokenizer.py     # 分词（HanLP + jieba）
│   ├── syntax.py        # 句法分析（依存句法 + SRL）
│   ├── sentiment.py     # 情感分析（transformers）
│   └── rhetoric.py      # 修辞识别
├── agents/
│   └── text_analyst.py  # 文本解析 Agent（5 维分析）
├── knowledge/
│   ├── embedding.py     # text2vec Embedding 模型
│   ├── vector_store.py  # ChromaDB 向量库
│   ├── knowledge_graph.py # NetworkX 知识图谱
│   ├── case_library.py  # 案例库（JSON + 关键词检索）
│   └── retriever.py     # 统一检索接口
├── guardrails/          # 安全/伦理防护（待实现）
├── evaluation/
│   ├── schema.py        # 评测数据集结构
│   └── scorer.py        # 评测评分脚本
├── api/                 # FastAPI Web 接口（待实现）
└── utils/               # 工具函数（待实现）
```

## 开发进度

- [x] 阶段一：基础框架（配置、Agent 接口、LLM 客户端、协调器骨架、CLI 入口）
- [x] 阶段二：NLP 管道 + 文本解析 Agent（HanLP、jieba、transformers、修辞识别、评测框架）
- [x] 阶段三：RAG 知识引擎（ChromaDB、知识图谱、理论文献库、案例库、统一检索）
- [ ] 阶段四：心理分析 + 逻辑推理 Agent
- [ ] 阶段五：报告生成 + 协调器完善
- [ ] 阶段六：Web 接口 + 可观测性

## 技术栈

- Python 3.11+, LangGraph, FastAPI
- HanLP, jieba, transformers
- ChromaDB, NetworkX
- DeepSeek API, Pydantic Settings v2, pytest
