<div align="center">

# 📚 PaperMind

**面向科研阅读与学术问答的论文知识库系统**

上传 PDF，自动完成版面解析、多粒度分块与混合检索索引。  
结合三层论文地图、阅读工具与 LangGraph 工作流，让回答中的引用可回溯至原始 PDF 页码与坐标。

通过研究课题、研究者画像与跨会话记忆，持续整理阅读进度、研究目标。

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-1C3C3C)](https://www.langchain.com/langgraph)
[![ParadeDB](https://img.shields.io/badge/Vector_%26_BM25-ParadeDB_%2F_pgvector-336791?logo=postgresql&logoColor=white)](https://www.paradedb.com/)
[![Docker Compose](https://img.shields.io/badge/Deploy-Docker_Compose-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/compose/)

[核心特性](#-核心特性) • [界面预览](#-界面预览) • [系统架构](#-系统架构与工作流) • [快速开始](#-快速开始) • [配置说明](#️-配置说明) • [开发与测试](#-开发与测试)

</div>

---

## 🖼️ 界面预览

<div align="center">
  <img src="images/image.png" alt="PaperMind 问答工作台" width="100%" />
  <p><em>PaperMind 科研问答工作台：支持 Fast / Agentic 双模式切换、多轮追问、引用原文悬停预览、点击跳转 PDF 坐标高亮，以及右侧实时检索与校验进度面板。</em></p>
</div>

---

## ✨ 核心特性

- 🗂️ **多用户私人论文库**
  - 支持邮箱验证码注册与找回密码、多知识库分组管理、PDF 拖拽上传与版本原子切换。
  - 自动融合 PDF 属性、MinerU 解析原文、Crossref / arXiv 接口与快速模型，完成论文元数据（标题、作者、年份、场刊、DOI、摘要）提取与在线编辑。
  - 支持失败重试、复用解析结果的低成本**重新分块（Rechunk）**，以及重新云端解析（Reparse）。

- 📖 **PDF 与结构化正文并排精读**
  - 左侧支持切换**解析 Markdown 视图**（支持 GFM 表格、KaTeX 公式、同源授权图片渲染与严格 XSS 清洗）与**分块检查器（Chunk Inspector）**。
  - 右侧内嵌原始 PDF 阅读器，支持点击任意分块或引用徽章自动翻页，并在页内绘制归一化 `bbox` 坐标高亮框。

- 🧠 **三层论文地图（L0 / L1 / L2）与原生阅读工具**
  - **L0 论文地图**：挂载论文标题、完整摘要、两级目录树及全量图表（Table / Figure）Caption 索引，辅助全局导航。
  - **L1 章节全文**：按原生学术章节（`method`、`experiments`、`ablation`、`limitations` 等）保留完整父块上下文。
  - **L2 多模态分块**：结合带上下文头的段落块、保留完整 Markdown / HTML 结构的表格块（附带趋势与极值摘要）及上下文关联公式块。
  - **三大阅读工具**：模型按需原生调用 `search_paper_chunks`（多探针并行混合检索）、`read_paper_section`（整节无损调阅）与 `inspect_element`（按 `Table 2` / `Equation (4)` 精确调阅图表公式及正文引用分析）。

- 🔬 **Fast / Agentic 双模式科研问答**
  - **Fast 快速问答**：记忆加载 → 问题改写 → 混合检索与工具调用 → 流式生成 → 引用规则校验。
  - **Agentic 深度核验**：在 Fast 基础上增加**候选来源相关性评分**、**检索纠错改写**与**LLM 事实依据核验（Verify Grounding）**；未通过校验时触发有界重生成，达到重试上限后返回安全回退回答。
  - **断点续读 SSE**：基于 PostgreSQL 事件表与 Redis Stream 双写镜像，支持网络断线或刷新页面后通过 `Last-Event-ID` 无缝恢复流式输出与实时节点时间线。

- 🧩 **研究项目与可追溯记忆**
  - **研究记忆中心**：统一管理研究者画像、项目核心备忘与按用户 / 项目 / 论文 / 会话隔离的档案记忆，支持自动写入与人工审核两种模式。
  - **跨会话延续**：检索历史问答原文，保存结构化研究状态、回答与引用，以及与论文版本绑定的阅读位置。
  - **可追溯治理**：支持来源有效性检查、筛选、批量确认 / 固定 / 迁移 / 删除、变更记录与撤销；工作台展示本轮参考和写入的记忆。

- 📊 **问答追踪与检索诊断**
  - 以稳定的 trace ID 串联问答执行、恢复与结构化日志，记录工具调用、各检索阶段、模型调用和降级原因。
  - 向 Langfuse 上报回答状态、引用与依据核验结果、重试次数及用量；默认关闭原文采集。

- 🛡️ **租户隔离、配额预算与管理控制台**
  - 检索 SQL 通道在 Top-K 召回前强制校验用户归属与活跃版本，杜绝跨租户数据泄漏。
  - 内置原子预占与按节点（Embedding、Rerank、改写、生成、核验、摘要）精细化结算的每日页数 / Token 预算控制。
  - 提供个人回答偏好设置（语言 / 详略）、分节点用量统计面板，以及管理员专属的全局监控、用户启停、配额覆盖与评测报告控制台。

---

## 🏗️ 系统架构与工作流

```mermaid
flowchart LR
    subgraph Client["前端工作台 (React 19 + PDF.js)"]
        UI["知识库 / 并排阅读器 / 问答 / 研究记忆中心"]
    end

    subgraph Backend["后端服务 (FastAPI + ARQ Workers)"]
        API["FastAPI 接口 & SSE 流"]
        Ingest["入库队列 Worker"]
        Chat["问答队列 Worker (LangGraph)"]
        Mail["邮件队列 Worker"]
    end

    subgraph Storage["存储与索引层"]
        PG[("PostgreSQL / ParadeDB\n(pgvector + BM25 + Checkpoints + 研究记忆)")]
        Redis[("Redis\n(ARQ 队列 + SSE Stream)")]
    end

    subgraph Providers["外部模型与解析服务"]
        MinerU["MinerU 云端版面解析"]
        LLM["主模型 / 快速模型 (OpenAI 兼容)"]
        Emb["Embedding & Reranker"]
    end

    UI <-->|"REST / SSE (Cookie + CSRF)"| API
    API <--> PG
    API <--> Redis
    Redis <--> Ingest & Chat & Mail
    Ingest -->|"PDF 解析"| MinerU
    Ingest -->|"向量化 / 元数据抽取"| Emb & LLM
    Ingest -->|"写入 L0/L1/L2"| PG
    Chat <-->|"混合检索 / 论文工具"| PG
    Chat -->|"规划 / 生成 / 事实核验"| LLM & Emb
```

---

## 🛠️ 技术栈

| 层级 | 核心技术与组件 |
| :--- | :--- |
| **前端工作台** | React 19、TypeScript、Vite、Tailwind CSS、React PDF (`pdf.js`)、`react-markdown`、KaTeX |
| **后端 API** | Python 3.12、FastAPI、SQLAlchemy 2.0 (Async)、Alembic、Pydantic v2 |
| **问答编排** | LangGraph、LangChain、原生 Function Calling、PostgreSQL Checkpointer、可续读 SSE |
| **存储与检索** | PostgreSQL 16 (ParadeDB)、`pgvector` 稠密向量检索、`pg_search` BM25 稀疏检索、动态加权 RRF 融合 + Rerank、`pg_trgm` 历史问答检索 |
| **异步任务** | Redis 7、ARQ 异步队列（独立隔离 `mail` / `ingest` / `chat` 队列）+ Transactional Outbox |
| **外部服务** | MinerU VLM 云解析、OpenAI 兼容 LLM（如 DeepSeek / Qwen）、Embedding (`bge-m3`)、Reranker (`bge-reranker-v2-m3`)、SMTP、Langfuse |
| **部署与基建** | Docker Compose、Nginx（同源反向代理与安全 CSP 响应头） |

---

## 🚀 快速开始

### 1. 环境要求

- **Docker** 与 **Docker Compose** `2.24+`
- **Python** `3.12` 与 [**uv**](https://docs.astral.sh/uv/)
- **Node.js** `20.19+`（或 `22+`）与 **npm**
- 可用的外部服务凭据：OpenAI 兼容模型 API、Embedding & Rerank 服务、MinerU Token、SMTP 邮箱配置（以及可选的 Langfuse 追踪密钥）

### 2. 安装与一键启动

```bash
git clone https://github.com/gxiaopang/PaperMind.git
cd PaperMind

# 1. 首次使用时复制配置模板（若已有 .env 请保留）
test -f .env || cp deploy/.env.example .env
```

编辑根目录下的 `.env`，填入模型服务、MinerU 和 SMTP 等必要凭据，随后执行：

默认启用 Langfuse 追踪，需要填写 `LANGFUSE_PUBLIC_KEY` 与 `LANGFUSE_SECRET_KEY`；暂不使用追踪时，请在 `.env` 中设置 `TRACING_ENABLED=false`，否则后端配置校验不会通过。

```bash
# 2. 安装前后端依赖，并自动生成 .env.local（包含随机数据库密码、Fernet 邮件加密密钥与本地连接串）
make bootstrap

# 3. 构建镜像并启动 PostgreSQL、Redis、数据库自动迁移、API、Worker 和 Web 服务
make up
```

启动完成后，即可通过浏览器访问以下入口：

| 服务入口 | 访问地址 | 说明 |
| :--- | :--- | :--- |
| **Web 工作台** | http://localhost:8080 | 注册并验证邮箱后，即可创建知识库、上传 PDF 与发起问答 |
| **API 交互文档** | http://localhost:18000/docs | FastAPI OpenAPI / Swagger 调试界面 |
| **服务就绪检查** | http://localhost:18000/readyz | 检查数据库、Redis 与核心依赖连通状态 |

常用运维指令：

```bash
make logs     # 实时查看 API / Worker / Web 容器日志
make down     # 停止容器服务（保留数据库与论文文件持久卷）
```

> [!NOTE]
> 上传论文进行解析时会将 PDF 发送至 MinerU 云服务；解析、向量化、重排序与多轮问答均会消耗对应外部服务额度。

### 3. 已有部署升级

Docker Compose 部署运行 `make up`，迁移服务会在 API 与 Worker 启动前自动升级数据库。本机调试部署需先运行 `make migrate`，再重启 API 和 Worker。

研究记忆功能依赖 `0012_research_memory` 与 `0013_hermes_memory_harness` 两个迁移，包含研究项目、档案记忆、核心备忘、变更日志和历史消息检索索引。升级前请备份数据库与论文文件持久卷；详细变更见 [研究记忆说明](MEMORY.md)。

---

## ⚙️ 配置说明

完整环境变量模板参见 [`deploy/.env.example`](deploy/.env.example)。系统按以下优先级依次加载并覆盖配置（进程环境变量优先级最高）：

```text
.env  ──►  .env.local (由 make bootstrap 自动生成管理)  ──►  .env.smtp.local (可选独立邮件凭据)
```

| 配置分组 | 关键变量 | 说明 |
| :--- | :--- | :--- |
| **主模型 / 快速模型** | `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`<br>`LLM_FAST_BASE_URL`, `LLM_FAST_API_KEY`, `LLM_FAST_MODEL` | 主模型负责规划、生成与核验；快速模型负责路由、改写、评分、摘要及元数据/表格摘要抽取 |
| **向量与重排** | `EMBEDDING_BASE_URL`, `EMBEDDING_API_KEY`, `EMBEDDING_MODEL`, `EMBEDDING_DIM`<br>`RERANK_BASE_URL`, `RERANK_API_KEY`, `RERANK_MODEL` | 模板默认使用 `BAAI/bge-m3` 与 `BAAI/bge-reranker-v2-m3`；当前数据库索引要求 `EMBEDDING_DIM=1024` |
| **MinerU 解析** | `MINERU_API_BASE`, `MINERU_API_TOKEN`, `MINERU_MODEL_VERSION` | 云端 PDF 解析凭据，默认开启公式与表格识别（`vlm` 模式） |
| **邮件服务** | `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `MAIL_FROM` | 用于发送 6 位注册验证码与密码重置邮件；可单独存放在 `.env.smtp.local` |
| **可观测性** | `TRACING_ENABLED`, `TRACE_CONTENT_ENABLED`, `LANGFUSE_BASE_URL`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` | 默认开启追踪，`TRACE_CONTENT_ENABLED=false` 屏蔽输入输出原文并保留诊断字段；无需追踪时可设 `TRACING_ENABLED=false` |
| **配额与并发** | `USER_DAILY_PARSE_PAGES`, `USER_DAILY_MODEL_TOKENS`, `PROVIDER_CONCURRENCY` | 单用户每日解析页数、每日模型 Token 预算及跨进程供应商并发上限 |
| **问答与记忆预算** | `RAG_MODE`, `RAG_MAX_RETRIEVAL_RETRIES`, `RAG_MAX_GENERATION_RETRIES`<br>`MEMORY_RECENT_TOKEN_BUDGET`, `MEMORY_LONG_TERM_TOKEN_BUDGET`, `MEMORY_SUMMARIZE_AFTER_TURNS` | 默认 Agentic；检索纠错最多 2 次、重生成最多 1 次；近期历史预算 2000 Token、档案记忆预算 1600 Token，默认每 8 轮或出现上下文压力时生成摘要 |

`MEMORY_LONG_TERM_TOKEN_BUDGET` 可按需添加到 `.env`；默认值与校验范围以 [`backend/app/core/config.py`](backend/app/core/config.py) 为准。

> [!CAUTION]
> 请勿将包含真实密钥的 `.env*`、本地论文数据（`data/`）或私有解析产物提交至 Git 仓库。

### 追踪与诊断

问答 trace ID 由用户 ID 与 `run_id` 确定性生成，恢复执行保持同一 trace，每次执行创建独立 attempt span；后台记忆总结使用独立 trace，并保留同一 `run_id` 和会话关联。Worker 的结构化日志包含 `run_id` 与 `trace_id`，可用 `app.core.observability.run_trace_id(user_id, run_id)` 定位问答 trace。

检索链路记录工具调用、query embedding、向量/BM25/元数据召回、RRF、候选扩展、rerank 与上下文组装，各阶段保留耗时、数量和降级信息。关闭内容记录时仍保留白名单诊断 metadata，过滤正文、工具参数及记忆；开启内容记录后，输入输出仍会过滤常见凭据和邮箱，并限制每个内容属性的长度。

问答结束后，从已提交的最终记录生成 `result_status`、`citation_validation`、`grounding`、`degraded`、`retrieval_retries`、`generation_retries` 六项 Langfuse scores，并记录引用数、恢复次数、用量及估算标记。未执行的检查使用 `not_run`，不适用的检查使用 `not_applicable`；相同问答重复投递使用相同 score ID。追踪采用尽力上报，SDK 异常不会使问答失败；进程被强制终止时仍可能缺少上报，额度核算以数据库账本为准。

---

## 💻 开发与测试

### 本机热重载调试

若需在宿主机直接调试前后端代码，可仅通过 Docker 启动基础设施（PostgreSQL 与 Redis）：

```bash
make bootstrap
make infra        # 仅启动 PostgreSQL (15432) 与 Redis (16379)
make migrate      # 执行 Alembic 数据库迁移与 LangGraph Checkpoints 初始化

# 在三个终端分别启动服务（Vite 会自动将 /api 代理至本机 18000 端口）：
make api
make worker
make web
```

### 管理员命令行工具

```bash
# 在 backend/ 目录下执行（或通过 docker compose exec api python -m app.cli ...）：
cd backend
uv run python -m app.cli admin someone@example.com      # 授予管理员权限
uv run python -m app.cli disable someone@example.com    # 停用账户并立即吊销其活跃会话
uv run python -m app.cli enable someone@example.com     # 重新启用账户
```

### 质量检查与自动化测试

普通测试均使用内存替身或独立容器（`testcontainers`），**不会调用付费模型、MinerU 或发送真实邮件**：

```bash
make check            # 代码质量检查：ruff lint/format、mypy 类型检查、前端 eslint 与生产构建
make test             # 运行完整后端测试套件（含独立 PostgreSQL / Redis 集成测试）
make test-parsing     # 运行 MinerU 解析与容错离线测试
make test-ingestion   # 运行分块、批量向量化与入库测试
make test-retrieval   # 运行混合检索、论文工具、租户隔离与降级测试
make test-chat        # 运行 LangGraph 问答图、记忆摘要、SSE 续读与幂等测试
make test-memory      # 无需 Docker 的记忆隔离、召回、来源失效与摘要回归
make eval-memory      # 无模型调用的合成记忆召回排序基准
make e2e              # 启动真实后端与独立数据库，运行 Playwright Chromium 端到端测试
```

记忆界面与追踪诊断也可单独回归：

```bash
cd backend
uv run pytest tests/test_observability.py tests/test_chat_observability.py
cd ../frontend
npx playwright test tests/unit.spec.ts tests/memory.spec.ts
```

如需显式验证外部供应商连通性或运行实模评测（会消耗少量 API 额度）：

```bash
make smoke            # 校验 LLM、Embedding、Rerank、MinerU、SMTP 与 Langfuse 连通性
make eval-retrieval   # 运行 M3 检索黄金集四种消融方案评测
make smoke-chat       # 运行 M4 真实模型工具问答冒烟测试
```

---

## 📂 项目结构与文档

```text
PaperMind/
├── backend/       # FastAPI 应用、LangGraph 问答编排、混合检索、ARQ 异步任务、Alembic 迁移与后端测试
├── frontend/      # React 19 + TypeScript 前端工作台、并排 PDF 阅读器与 Playwright E2E 测试
├── deploy/        # Dockerfile、Docker Compose 编排、Nginx 安全配置与环境变量模板
├── scripts/       # 环境初始化 (bootstrap)、镜像拉取辅助、E2E 启动器与评测脚本
├── eval/          # 检索评测数据集 (v7)、冻结语料与人工审核记录
├── docs/          # 架构设计、阶段开发说明与验收记录
├── reports/       # 脱敏的自动化验证与检索评测报告
├── images/        # 项目界面预览截图
├── MEMORY.md      # 研究记忆架构、使用方式、接口与迁移说明
└── README.md      # 项目介绍、启动、配置与验证入口
```
