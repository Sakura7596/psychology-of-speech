# 语言心理学话语分析系统

> 基于多 Agent 协作 + RAG 知识引擎的中文话语深度分析系统。不只是情感打分——从语言心理学视角解读说话者的真实意图、权力动态和认知模式。

## 项目简介

日常沟通中，大量信息隐藏在字面意义之下。"我再考虑考虑"可能是委婉拒绝，"领导说得对"可能暗含不满。现有 NLP 工具只能做浅层情感打分，无法解读这类高语境沟通中的言外之意。

本系统通过**多 Agent 并行推理 + RAG 知识检索**解决这一问题：

1. **Orchestrator** 根据文本特征动态规划分析策略
2. 三个专业 Agent **并行执行**深度推理：
   - **TextAnalyst** — 句法解析、词汇分析、修辞识别、话语标记、情感分析
   - **PsychologyAnalyst** — 基于 20+ 语言心理学理论（言语行为、面子理论、权力距离、认知失调等）和 40+ 真实案例进行多理论交叉推理
   - **LogicAnalyst** — 论证结构提取、20 种逻辑谬误检测
3. **ReportGenerator** 整合三路结果，生成包含理论溯源和置信度标注的结构化报告
4. 全程包含三层安全 Guardrails：幻觉检测、伦理合规、PII 脱敏

### 核心特性

- **多 Agent 并行推理** — Orchestrator 动态规划 + 3 Agent 并行分析 + 报告融合
- **RAG 知识引擎** — ChromaDB 向量检索 + NetworkX 知识图谱 + 40+ 案例库关键词检索
- **语言心理学理论支撑** — 20+ 理论（Austin/Searle、Brown & Levinson、Hofstede、Hall、Goffman 等）
- **SSE 流式输出** — 前端实时展示 Agent 协作过程
- **Web 爬取管线** — 自动从知乎、豆瓣、小红书等平台扩充案例库
- **三层安全 Guardrails** — 幻觉检测、伦理合规（自动注入免责声明）、PII 脱敏

### 分析示例

输入："领导说：'这个方案还需要再完善一下。'小王回答：'好的，我会尽快修改。'但他心里想：'每次都是这样，要求不明确就让人改。'"

系统会从以下维度分析：
- **语言学**：句式结构、话语标记（转折/让步）、情态动词、修辞手法
- **心理学**：面子策略（消极礼貌）、权力距离（高权力距离下的服从策略）、防御机制（合理化）
- **逻辑学**：论证结构、隐含前提、可能的逻辑谬误

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/Sakura7596/psychology-of-speech.git
cd psychology-of-speech
```

### 2. 安装后端依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API Key

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的 [DeepSeek API Key](https://platform.deepseek.com/api_keys)：

```
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

### 4. 运行分析

```bash
# CLI 模式
python -m src.main "要分析的文本"
python -m src.main "要分析的文本" deep    # 深度分析
python -m src.main "要分析的文本" quick   # 快速分析

# Web 服务模式（后端 API）
python -m src.main serve                   # 启动 http://localhost:8000
```

### 5. 启动前端界面（可选）

需要同时运行后端服务：

```bash
cd frontend
npm install
npm run dev    # 启动 http://localhost:5173
```

### 6. 运行测试

```bash
DEEPSEEK_API_KEY=test-key python -m pytest tests/ -v
```

## 架构

```
用户输入
   │
   ▼
Orchestrator（动态规划分析策略）
   │
   ├── TextAnalyst ──────── 句法 / 词汇 / 修辞 / 话语标记 / 情感
   ├── PsychologyAnalyst ── RAG 检索 20+ 理论 + 40+ 案例 → 多理论交叉推理
   └── LogicAnalyst ─────── 论证结构 / 20 种逻辑谬误检测
   │
   ▼（并行执行，置信度加权融合）
ReportGenerator → 结构化报告（Markdown/JSON/HTML）
   │
   ▼
EthicsGuard → 诊断术语审查 + 免责声明注入
   │
   ▼
用户输出（SSE 流式 / 同步）
```

## 项目结构

```
src/
├── main.py                    # CLI 入口 + Web 服务启动
├── config.py                  # Pydantic Settings v2 配置管理
├── agents/
│   ├── base.py                # BaseAgent 抽象类 + 数据模型
│   ├── orchestrator.py        # 总协调器（LLM/规则双模式规划 + 并行管道）
│   ├── text_analyst.py        # 文本解析 Agent（句法、词汇、修辞、情感）
│   ├── psychology_analyst.py  # 心理分析 Agent（20+ 理论 + RAG）
│   ├── logic_analyst.py       # 逻辑推理 Agent（20 种谬误检测）
│   └── report_generator.py    # 报告生成 Agent（多格式输出）
├── llm/
│   ├── client.py              # LLM 统一客户端（带缓存 + 重试）
│   ├── deepseek.py            # DeepSeek/MiMo API 适配器（流式 + 指数退避）
│   ├── prompts.py             # Prompt 模板（含理论知识和示例）
│   └── exceptions.py          # 自定义异常
├── nlp/
│   ├── tokenizer.py           # 分词（HanLP 主力 + jieba fallback）
│   ├── syntax.py              # 依存句法 + 语义角色标注
│   ├── sentiment.py           # 情感分析（transformers + 规则 fallback）
│   └── rhetoric.py            # 修辞识别（比喻、反问、排比、夸张）
├── knowledge/
│   ├── embedding.py           # text2vec Embedding 模型
│   ├── vector_store.py        # ChromaDB 向量库（余弦相似度）
│   ├── knowledge_graph.py     # NetworkX 有向知识图谱
│   ├── case_library.py        # 案例库（单例 + 倒排索引）
│   └── retriever.py           # 统一检索接口（向量 + 图谱 + 案例融合）
├── guardrails/
│   ├── hallucination.py       # 幻觉防御（来源检查 + 交叉验证）
│   ├── ethics.py              # 伦理守则（诊断术语审查 + 免责声明）
│   └── privacy.py             # 隐私保护（手机/邮箱/身份证脱敏）
├── scraper/                   # 知识库爬取管线
│   ├── pipeline.py            # 爬取 → 清洗 → 分析 → 验证 → 存储
│   ├── sources/               # 知乎、豆瓣、小红书、心理学博客
│   └── ...
├── api/
│   ├── app.py                 # FastAPI 工厂（CORS + 限流 + 日志中间件）
│   ├── routes.py              # 路由（/analyze, /analyze/stream, /scrape）
│   └── models.py              # Pydantic 请求/响应模型
└── evaluation/                # 评测框架
frontend/                      # Vue 3 + Tailwind CSS 前端
data/
├── cases/                     # 40+ 案例库（JSON）
├── graph/                     # 知识图谱
└── literature/                # 理论文献（20+ 理论）
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
| `/analyze/stream` | POST | SSE 流式分析（实时返回 Agent 进度和报告内容） |
| `/scrape` | POST | 触发知识库爬取（body: `{query, sources?, max_items_per_source?}`） |
| `/scrape/sources` | GET | 列出可用爬取源 |

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

- 144 个测试全部通过
- 6 个阶段全部完成 + 前端界面
- 5 个 Agent（协调器、文本解析、心理分析、逻辑推理、报告生成）
- 3 层 Guardrails（幻觉防御、伦理守则、隐私保护）
- RAG 知识引擎（向量库 + 知识图谱 + 案例库）
- 40+ 案例库文件（涵盖职场、情感、家庭、社交等场景）

## 技术栈

- Python 3.11+, LangGraph, FastAPI
- HanLP, jieba, transformers
- ChromaDB, NetworkX
- DeepSeek API, Pydantic Settings v2, pytest
