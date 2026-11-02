# 研究记忆（Hermes / Letta / mem0 三层架构）

PaperMind 采用受 **Hermes Agent**（Nous Research）、**Letta / MemGPT** 与 **mem0** 启发的三层研究记忆架构，在近期对话、会话研究状态和长期记忆之间分别设置预算。历史、核心记忆、摘要和档案记忆均用于理解用户目标、学术背景与指代；**论文事实仍须从本轮授权来源重新检索并引用**，任何记忆均不会进入证据充分性评分或支撑度校验节点。

## 三层记忆架构

1. **Tier 1 — 核心常驻记忆（Core Memory Snapshot，对标 Hermes `USER.md` / `MEMORY.md`）**
   - **研究者画像（`USER.md`，`<= 1500` 字符）**：跨课题常驻的研究领域、方法偏好、数学推导习惯与回答要求。
   - **项目核心备忘（`MEMORY.md`，`<= 3000` 字符）**：当前研究课题的核心目标、关键假设、实验约束与阶段结论，并与项目描述保持同步。
   - **冻结快照语义**：在每轮会话开始（`load_memory`）时以冻结快照注入规划与生成上下文；支持自动行级去重合并与超限裁剪。
2. **Tier 2 — 结构化档案记忆库（Archival Memory Store，对标 mem0 去重合并机制）**
   - 按 `user / project / paper / conversation` 四级作用域隔离，支持 `goal / constraint / preference / hypothesis / conclusion / focus / correction / note / discussed` 九类研究条目。
   - **写入去重与冲突消解（`smart_save_entry`）**：写入前自动执行归一化精确去重与词项相似度合并（`similarity_score >= 0.80` 原位更新）；当写入类型为 `correction` 时，自动将同课题冲突的旧目标/假设/结论标为 `superseded`。
   - **按需治理取代逐轮噪声**：不再将每轮普通提问机械存为未确认候选；仅在用户明确表达研究目标、限定条件、偏好、待验证假设或纠正时由 `curate_user_turn` 提取，并支持一键清理历史遗留的未确认 `discussed` 候选。
3. **Tier 3 — 跨会话原文检索（Cross-Session Recall，对标 Hermes `session_search`）**
   - 基于 PostgreSQL `pg_trgm` GIN 索引（`ix_messages_content_trgm`）与中英双字词/关键词重合度重排，支持跨会话检索历史问答原文片段，找回摘要压缩前讨论过的公式、实验细节与推导过程。

## 使用与前端治理

1. **研究记忆中心（`#/memory`）**：左侧导航新增独立二级页面“研究记忆中心”，集中提供：
   - **核心常驻记忆双栏编辑器**：带实时字符计量条编辑研究者画像（`1500` 字上限）与当前项目核心备忘（`3000` 字上限）。
   - **写入策略切换**：支持 **自动模式（Hermes 默认）**（自动将高置信度目标/约束/偏好/纠正写入已确认记忆并同步核心备忘）与 **审核模式**（仅生成待确认候选）。
   - **档案记忆筛选与批量治理**：支持按关键词、层级、类型、状态、证据有效性筛选，以及全选、批量确认、批量固定、批量标为已修正、批量迁移项目、批量删除与一键清理历史提问噪声。
   - **跨会话原文检索与变更历史**：支持按关键词检索历史问答片段并一键跳转原会话；所有自动或手动记忆变更均记录在 `memory_events` 表中，支持**一键撤销（Undo）**。
2. **问答工作台透明化反馈**：
   - 助手回答下方展示可折叠的 **`参考了 N 条研究记忆 ▾`** 胶囊，展开即可查看本轮召回的记忆类型与内容。
   - 当本轮自动记住或更新了研究记忆时，回答下方展示 **`已记住：… [撤销]`** 提示条，点击“撤销”即可立即回滚该次写入。
   - 点击回答下方的“保存为研究记忆”打开弹窗时，自动剥离回答中的 `[1][2]` 局部引用角标，引导保存为独立可复用的研究判断。

## 接口

| 接口 | 用途 |
| --- | --- |
| `GET/POST /api/v1/projects` | 列出、创建研究项目 |
| `PATCH/DELETE /api/v1/projects/{id}` | 修改、删除研究项目 |
| `GET/POST /api/v1/memories` | 查看、保存档案记忆；支持 `project_id`、`conversation_id`、`level`、`kind`、`status`、`evidence_status`、`q` 筛选 |
| `PATCH/DELETE /api/v1/memories/{id}` | 修改、确认、固定、标记假设或已修正、软删除（记录可撤销事件） |
| `POST /api/v1/memories/bulk` | 批量确认、固定、标为已修正、迁移项目或删除记忆 |
| `POST /api/v1/memories/cleanup-discussed` | 一键清理历史遗留的未确认 `discussed` 提问候选 |
| `GET/PUT /api/v1/memory/settings` | 查看、切换研究记忆写入策略（`auto` / `review`） |
| `GET/PUT /api/v1/memory/core` | 查看、编辑研究者画像（`level=user`）与项目核心备忘（`level=project`） |
| `GET /api/v1/memory/events` | 查看研究记忆变更审计日志 |
| `POST /api/v1/memory/events/{id}/undo` | 一键撤销指定的记忆新增、更新、替代或删除操作 |
| `GET /api/v1/memory/sessions/search` | 跨会话检索历史问答原文片段（`session_search`） |
| `GET /api/v1/memories/{id}/source` | 查看记忆关联的原始来源消息 |
| `GET/PATCH/DELETE /api/v1/conversations/{id}/memory` | 查看、编辑或清除会话摘要与结构化研究状态 |
| `GET/PUT /api/v1/papers/{id}/reading-position` | 查看、保存与版本绑定的论文阅读位置 |

全部接口沿用登录、CSRF 与严格的用户归属检查。

## 迁移与验证

数据库迁移为 `0012_research_memory` 与 `0013_hermes_memory_harness`（包含 `research_projects`、`memory_entries`、`memory_core_blocks`、`memory_events`、`user_settings.memory_write_mode` 及 `ix_messages_content_trgm` 索引）。部署新代码前执行：

```bash
make migrate
```

无需外部模型或 Docker 的离线验证：

```bash
make test-memory
make eval-memory
```

前端构建与单元/端到端回归：

```bash
cd frontend
npm run lint && npm run build
npx playwright test tests/unit.spec.ts tests/memory.spec.ts
```
