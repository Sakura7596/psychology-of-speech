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
   # CLI 模式
   python -m src.main "要分析的文本"
   python -m src.main "要分析的文本" deep    # 深度分析
   python -m src.main "要分析的文本" quick   # 快速分析

   # Web 服务模式（后端 API）
   python -m src.main serve                   # 启动 http://localhost:8000

   # 前端界面（需要同时启动后端）
   cd frontend && npm install && npm run dev  # 启动 http://localhost:5173
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
│   ├── text_analyst.py      # 文本解析 Agent（5 维分析）
│   ├── psychology_analyst.py # 心理分析 Agent（中西方理论）
│   └── logic_analyst.py     # 逻辑推理 Agent（谬误检测）
├── knowledge/
│   ├── embedding.py     # text2vec Embedding 模型
│   ├── vector_store.py  # ChromaDB 向量库
│   ├── knowledge_graph.py # NetworkX 知识图谱
│   ├── case_library.py  # 案例库（JSON + 关键词检索）
│   └── retriever.py     # 统一检索接口
├── guardrails/
│   ├── hallucination.py # 幻觉防御
│   ├── ethics.py        # 伦理守则 + 免责声明
│   └── privacy.py       # 隐私保护（PII 脱敏）
├── evaluation/
│   ├── schema.py        # 评测数据集结构
│   └── scorer.py        # 评测评分脚本
├── api/
│   ├── app.py           # FastAPI 应用工厂 + 日志中间件
│   ├── routes.py        # 路由（/analyze, /health）
│   └── models.py        # 请求/响应 Pydantic 模型
└── utils/               # 工具函数（待实现）
```

## 开发进度

- [x] 阶段一：基础框架（配置、Agent 接口、LLM 客户端、协调器骨架、CLI 入口）
- [x] 阶段二：NLP 管道 + 文本解析 Agent（HanLP、jieba、transformers、修辞识别、评测框架）
- [x] 阶段三：RAG 知识引擎（ChromaDB、知识图谱、理论文献库、案例库、统一检索）
- [x] 阶段四：心理分析 + 逻辑推理 Agent + Guardrails（心理分析、逻辑推理、幻觉防御、伦理守则、隐私保护）
- [x] 阶段五：报告生成 + 协调器完善（报告生成 Agent、管道编排、端到端集成测试）
- [x] 阶段六：Web 接口 + 可观测性（FastAPI、/analyze、/health、日志中间件）

## API 接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/analyze` | POST | 文本分析（body: `{text, depth?, output_format?}`） |

## 前端界面

Vue 3 + Vite + Tailwind CSS 构建，位于 `frontend/` 目录。

```bash
cd frontend
npm install
npm run dev          # 开发服务器 http://localhost:5173
npm run build        # 生产构建
```

设计风格：克制学术风 + 趣味点缀（Agent 图标动画、置信度温度计、修辞标记）

## 项目统计

- 109 个测试全部通过
- 6 个阶段全部完成 + 前端界面
- 5 个 Agent（协调器、文本解析、心理分析、逻辑推理、报告生成）
- 3 层 Guardrails（幻觉防御、伦理守则、隐私保护）
- RAG 知识引擎（向量库 + 知识图谱 + 案例库）

## 技术栈

- Python 3.11+, LangGraph, FastAPI
- HanLP, jieba, transformers
- ChromaDB, NetworkX
- DeepSeek API, Pydantic Settings v2, pytest
