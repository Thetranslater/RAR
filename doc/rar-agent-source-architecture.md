# RAR Agent 源码结构与运行架构

本文说明 `src/rar_agent` 与 `web/src` 当前实现的职责边界。设计规格仍以
`doc/design/2026-09-01-rar-agent-text-core-implementation-spec.md` 为准；本文只描述已经落到源码中的结构。

## 1. 核心概念

- **Project**：一个本地工作目录，也是工具可见文件的安全边界。Project 不等于 Dataset。
- **Chat**：Project 下的一次独立对话。一个 Project 可以有任意数量的 Chat。
- **Dataset**：一次或多次资源提取形成的数据集产物。Dataset 独立于 Chat，也不会显示在当前左侧对话树中。
- **AgentHarness**：处理检查、修改、合并、重新导出等开放式任务的通用模型—工具循环。
- **DatasetBuildWorkflow**：处理文本数据集提取的固定、可恢复流程。

Web UI 将后两条执行路径放在同一个项目对话窗口中，但它们具有不同的控制方式：普通请求由 Agent 自行选择工具；文本提取则由程序控制阶段、并行单元和中间产物。

## 2. 目录总览

```text
src/rar_agent/
├─ agent/                    # 通用 Agent Harness 与工具系统
│  ├─ harness.py
│  └─ tools.py
├─ api/                      # FastAPI 本地服务
│  └─ app.py
├─ domain/                   # Dataset、Plot、Dialogue 等领域模型
│  └─ models.py
├─ export/                   # DatasetBundle 到训练格式的转换
│  └─ sharegpt.py
├─ models/                   # 模型协议、供应商适配、调度和测试模型
│  ├─ base.py
│  ├─ openai_compatible.py
│  ├─ scheduler.py
│  ├─ scripted.py
│  └─ structured.py
├─ prompts/                  # 可替换的提示词模板
│  ├─ loader.py
│  └─ defaults/
├─ storage/                  # Project SQLite、文件产物与数据库迁移
│  ├─ artifacts.py
│  ├─ database.py
│  └─ migrations/
├─ text/                     # 文本提取各确定性处理单元
│  ├─ chunking.py
│  ├─ characters.py
│  ├─ dialogue.py
│  ├─ plot_rebuilder.py
│  └─ tokenizer.py
├─ workflow/
│  └─ dataset_build.py       # 文本数据集固定 Workflow 编排
├─ cli.py                    # `rar-agent serve/extract`
├─ main.py                   # 通过环境变量启动服务
├─ settings.py               # 模型供应商配置
└─ extensions.py             # 后续资源类型扩展入口

web/src/
├─ App.tsx                   # 项目对话树、消息区、授权及提取交互
├─ main.tsx                  # React Router 入口
└─ styles.css                # 响应式界面样式
```

## 3. 通用 Agent 执行路径

普通对话采用后台运行模式：

```text
POST /api/chat
  → 创建或验证 Chat
  → 立即保存用户消息
  → 返回 202 + run_id
  → 后台 asyncio Task 执行 AgentHarness
  → ModelScheduler 获取全局模型名额
  → 模型返回文本或工具调用
  → 工具若有风险则进入 awaiting_approval
  → 用户允许/拒绝后继续同一个逻辑 Run
  → 保存助手消息、工具结果和用量
```

`agent/harness.py` 负责：

- 根据 Chat 历史构造模型上下文；
- 只保留能够完整装入预算的最近对话轮次，不截断半个工具调用链；
- 将 UI 事件排除在模型上下文之外；
- 执行多轮模型—工具循环；
- 在工具需要授权时返回可恢复的 `PendingAgentRun`；
- 达到轮次上限后返回可理解的未完成提示，而不是抛出 HTTP 500。

`agent/tools.py` 负责工具注册、工具定义、参数校验、Project 路径限制、风险属性和执行。同步文件工具通过工作线程执行，避免阻塞 FastAPI 事件循环。

## 4. 授权模型

授权属于一次工具调用的临时状态，不写入 SQLite。运行中的授权信息包含工具、规范化参数、风险影响、token 和过期时间；用户处理后即删除。当前规则为：

- 只读工具直接执行；
- 写入、修改等风险工具进入 `awaiting_approval`；
- 授权请求默认五分钟过期；
- 页面切换或刷新后，只要服务进程未重启，仍可通过 Run 状态取回授权；
- 服务重启后不恢复临时授权，用户可发送“继续”重新规划；
- Project 路径边界始终生效，用户授权不会扩大可访问目录。

## 5. Chat、后台任务与上下文

每个 Chat 是数据库中的逻辑会话，不拥有独立操作系统进程。多个 Chat 的后台任务由同一 Uvicorn 进程中的不同 `asyncio.Task` 执行，因此切换页面不会终止任务。

当前约束：

- 同一个 Chat 同时只允许一个 Agent Run；
- 不同 Chat 可以并行运行；
- `max_active_chat_turns` 限制同时运行的 Chat 数量；
- `max_model_concurrency` 是 Chat 与 Workflow 共用的最外层模型调用上限，默认值为 12；单个 Workflow 在每个模型阶段最多并行 8 个调用，剩余容量可由 Harness 或其他 Workflow 使用；
- Chat 模型调用优先于等待中的 Workflow 调用；
- 多个 Workflow 按 `workload_id` 轮转，避免单个数据集长期占满队列；
- 终态 Run 在内存中保留一小时，供前端轮询；
- 用户可取消正在运行或等待授权的 Run。

消息分为两类：

- `message`：用户、助手和工具消息，可按规则进入模型上下文；
- `event`：任务完成、失败、取消等 UI 时间线信息，持久化展示但不传给模型。

当前上下文按完整用户轮次裁剪。更长期的摘要、事实记忆与检索将由后续 Memory 模块承担。

## 6. Project SQLite

每个 Project 使用 `<project>/.rar/rar.sqlite3`。SQLite 保存运行状态，不替代 Dataset 的 JSON/JSONL 文件。

主要表：

| 表 | 用途 |
|---|---|
| `chat_sessions` | 对话标题、创建/更新时间、归档状态 |
| `messages` | 普通消息和 UI 事件、上下文标记 |
| `tool_calls` | 工具名称、参数、结果、授权影响和执行状态 |
| `workflow_runs` | 文本提取运行与其来源 Chat |
| `model_usage` | 每次模型调用的 token 用量 |
| `events` | 通用程序事件 |
| `artifact_refs` | Workflow 产物路径引用 |

`ProjectDatabase` 初始化时自动运行 Alembic `upgrade head`。首次启动会创建数据库；旧版数据库会补充 Chat 管理所需字段。删除 Chat 时只删除对话消息、工具调用和对应用量；不会删除 Project 文件或 Dataset，相关 Workflow 记录只解除 Chat 关联。

## 7. 文本 Dataset Workflow

`workflow/dataset_build.py` 编排以下固定流程：

```text
InputManifest
  → TextChunk
  → 分 Chunk 剧情提取
  → 候选角色聚合、正式名称选择与档案生成
  → PlotChunk 重建
  → 分 PlotChunk 对话提取
  → DatasetBundle
  → ShareGPT JSONL
```

关键实现分工：

- `text/chunking.py`：先识别卷、章 Section，再在每个 Section 内生成 TextChunk；
- `models/structured.py`：调用模型、解析结构化输出并执行有限重试；
- `text/characters.py`：聚合角色候选、准备单角色提示词输入并根据模型结果重排名称；
- `text/plot_rebuilder.py`：根据前后 Chunk 状态重建 PlotChunk；
- `text/dialogue.py`：规范化环境和对白，并按顺序回查原文位置；
- `storage/artifacts.py`：保存阶段产物、失败占位和恢复依据；
- `export/sharegpt.py`：按角色生成 ShareGPT 样本并标记 loss。

### TextChunk 的两层切分

`TextChunker` 不直接在整份原文上累计句子。它先匹配独占一行的卷标题和章标题，形成带有
`volume`、`chapter` 元信息的 Section，然后仅在当前 Section 内按完整句子累计到目标 Token 数。
默认情况下，标题本身和第一卷之前的前置内容不进入 Chunk 文本；可通过 Workflow 配置保留它们，
也可为不同体裁提供自定义卷、章匹配词。TextChunk 的 `index` 在单个输入文件内连续递增。

### Prompt 的加载和发送

`prompts/loader.py` 保留提示词文件原文，并在发送前识别最后一个独占一行的
`----------`。分隔线之前作为 system message，之后作为 user prefix，与紧随其后的 JSON 输入
共同组成 user message。角色档案提示词也使用相同的 system/user prefix 拆分规则。
默认剧情和对话提示词以 `repo/RLFF_extraction/prompt` 为参考并允许独立迭代；输入字段分别对齐
`input/previous/next` 与 `input/characters`。

角色整理在剧情提取后、剧情重建前执行。程序先按剧情候选中的首个名称聚合观察结果，合并并去重
所有名称和 description；每个聚合角色使用 `character_profile.txt` 调用一次模型。请求严格采用
`{"character":{"names":[...],"description":"..."}}`，响应严格采用
`{"profile":{"name":"...","content":"..."}}`。模型选择的 `name` 必须来自输入 names；程序将其
移动到名称数组首位，其余名称按原顺序保留，并在 DatasetBundle 中分别写为 `name` 和 `aliases`。
模型生成的 `content` 成为角色档案。剧情重建后，程序再根据角色引用补充 `plot_refs`，无需额外模型调用。

恢复完全由程序读取 Dataset 目录中的固定文件完成。某个并行单元失败时保留空 `result` 占位；恢复时按 index 找出缺失结果重做。V1 不检测原始输入是否在中途被修改。

`WorkflowConfig.debug` 用于低成本端到端验证。开启后，Chunk 阶段仍处理并保存全部输入；剧情提取
只处理前 5 个 TextChunk，剧情重建基于这部分结果，对话提取再只处理重建后的前 5 个
PlotChunk。角色整理与档案生成、DatasetBundle 和 ShareGPT 导出照常运行，但只基于上述有限结果。

## 8. Web UI 与路由

当前路由：

- `/`：恢复本地记录的最近 Chat，否则进入新对话；
- `/chats/new`：空白新对话，第一条消息发送时才创建 Chat；
- `/chats/:id`：指定 Chat。

左侧树只展示当前 Project 与其 Chat，不展示 Dataset。界面支持：

- 新建、选择、重命名、归档、恢复和永久删除 Chat；
- 最近 100 条消息及向前分页；
- 每个 Chat 独立的本地草稿；
- 后台运行和未读状态提示；
- 查看并处理工具授权；
- 停止当前 Agent Run；
- 从对话中启动文本提取。

永久删除 Chat 不会删除 Dataset 或用户文件。Dataset 未来通过独立的可视化报告查看。

前端会轮询对话状态，因此 Uvicorn 会记录周期性的 `GET /api/chats`：存在后台任务时约每
1.5 秒一次，空闲时约每 10 秒一次。归档列表只在初始化或用户操作后刷新，消息只在打开对话、
任务状态变化或发送消息后刷新，不再随对话列表轮询重复读取。

## 9. API 概览

| 方法 | 路径 | 作用 |
|---|---|---|
| `GET` | `/api/project` | 当前 Project 信息 |
| `GET/POST` | `/api/chats` | 列出或创建 Chat |
| `GET/PATCH/DELETE` | `/api/chats/{id}` | 查询、修改、归档或删除 Chat |
| `GET` | `/api/chats/{id}/messages` | 分页读取消息 |
| `POST` | `/api/chat` | 创建后台 Agent Run |
| `GET` | `/api/chats/{id}/runs/{run}` | 查询 Run 状态和授权 |
| `POST` | `/api/chats/{id}/runs/{run}/cancel` | 停止 Run |
| `POST` | `/api/approvals/{token}` | 允许或拒绝风险工具 |
| `POST` | `/api/extractions` | 启动文本提取 Workflow |
| `GET` | `/api/extractions/{task}` | 查询提取状态 |
| `POST` | `/api/extractions/{task}/confirm` | 阶段确认模式下继续 |

## 10. 配置与启动

CLI 示例：

```powershell
.venv\Scripts\rar-agent.exe serve `
  --project . `
  --provider deepseek `
  --model deepseek-chat `
  --max-model-concurrency 4 `
  --max-active-chat-turns 4 `
  --max-context-tokens 32768 `
  --reserved-output-tokens 4096
```

`python -m rar_agent.main` 还支持：

- `RAR_PROJECT_ROOT`
- `RAR_MODEL_PROVIDER`
- `RAR_MODEL`
- `RAR_MAX_MODEL_CONCURRENCY`
- `RAR_MAX_ACTIVE_CHAT_TURNS`
- `RAR_MAX_CONTEXT_TOKENS`
- `RAR_RESERVED_OUTPUT_TOKENS`

供应商密钥只从外部环境读取，不保存到项目配置或 SQLite。当前配置尚未进入 Web UI。

## 11. 测试结构

测试覆盖领域模型、Chunk、剧情重建、角色处理、对白定位、产物恢复、模型网关、Agent Harness、API、旧数据库迁移、上下文预算和模型并发调度。

```powershell
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m ruff check src tests
.venv\Scripts\python.exe -m mypy src
pnpm --dir web run typecheck
pnpm --dir web run test
pnpm --dir web run build
```

结构化模型测试使用 `ScriptedModelClient`，不会调用真实 DeepSeek/Qwen API；真实接口仍需在具备密钥时进行单独集成验证。
