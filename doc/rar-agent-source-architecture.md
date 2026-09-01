# RAR Agent 源码结构与模块职责

> 文档范围：`src/rar_agent/` 当前实现  
> 适用版本：RAR Agent 0.1.0 文字核心 V1  
> 更新日期：2026-09-01

## 1. 源码总体结构

RAR 后端采用 Python，实现上分成两条彼此协作但职责不同的执行路径：

1. `DatasetBuildWorkflow`：固定的数据集生产流程。适用于大规模、可并行、需要断点恢复的文字提取任务。
2. `AgentHarness`：通用的多轮模型与工具循环。适用于检查、修改、合并、修复、重新导出等无法提前穷举的任务。

两条路径共用模型接口、模型调度器、文件工具和 Project 范围约束，但不会把普通修改任务实现成新的专用 Workflow。

```text
src/rar_agent/
├── __init__.py
├── main.py                  # 默认 ASGI 应用入口
├── cli.py                   # rar-agent 命令行入口
├── settings.py              # DeepSeek/Qwen 环境配置
├── extensions.py            # 后续扩展协议
├── api/                     # FastAPI 本地服务
├── agent/                   # 通用 AgentHarness 与工具系统
├── domain/                  # 领域 Schema 与持久化契约
├── export/                  # DatasetBundle 派生导出
├── models/                  # 模型接口、适配器、调度与结构化解析
├── prompts/                 # 可替换 Prompt 文件
├── storage/                 # Dataset 文件产物和 SQLite
├── text/                    # 确定性文字处理算法
└── workflow/                # 固定 Dataset 构建流程
```

## 2. 运行时总览

### 2.1 本地服务启动

```text
rar_agent.main
  ├─ 从环境变量读取 Project、Provider 和 Model
  ├─ settings.create_model_client()
  ├─ AppRuntime
  │   ├─ ProjectDatabase
  │   └─ ModelScheduler
  └─ api.create_app()
      └─ FastAPI ASGI 应用
```

`main.py` 在没有 API Key 时仍然可以启动 Web 服务，用户可以浏览本地结果；需要模型的 Chat 和 Extraction 请求会返回明确的配置错误。

### 2.2 Chat 请求

```text
POST /api/chat
  → AgentHarness.run()
  → 从 SQLite 恢复已有对话
  → ModelScheduler
  → ModelClient.complete()
  → 如果模型返回 Tool Call：
      ToolDispatcher.dispatch()
      → 文件工具或 start_text_extraction
      → ToolObservation 返回模型
  → 直到模型返回无 Tool Call 的最终文本
  → 对话和工具记录写入 SQLite
```

### 2.3 数据集提取请求

```text
POST /api/extractions
或 Agent 调用 start_text_extraction
  → 创建后台 Extraction Task
  → DatasetBuildWorkflow.run()
  → 固定 Stage 图
  → Dataset 文件产物
  → ShareGPT 派生导出
```

自动模式连续执行所有 Stage；逐阶段确认模式在每个大 Stage 输出完成后暂停，等待 `/api/extractions/{task}/confirm`。

## 3. 顶层入口与配置

### `src/rar_agent/__init__.py`

- 定义包版本 `__version__`。
- 不承担运行时初始化，避免导入包时产生额外副作用。

### `src/rar_agent/main.py`

- 默认 ASGI 应用入口，可由 Uvicorn 加载。
- 读取以下环境变量：
  - `RAR_PROJECT_ROOT`
  - `RAR_MODEL_PROVIDER`
  - `RAR_MODEL`
- 尝试创建模型 Client；缺少 API Key 时返回 `None`，但不阻止 Web 服务启动。
- 创建 `AppRuntime` 和 FastAPI `app`。
- 使用延迟加载的 `TikTokenTokenizer`，启动服务时不会为了 Token 编码表主动访问网络。

### `src/rar_agent/cli.py`

提供 `rar-agent` 命令：

- `serve`：为指定 Project 启动本地 Web 服务。
- `extract`：从 InputManifest JSON 直接运行或恢复文字 Workflow。

CLI 只负责参数解析和组合已有模块，不复制 Workflow 或模型逻辑。

### `src/rar_agent/settings.py`

- 根据 Provider 创建 `OpenAICompatibleClient`。
- 首批支持 `deepseek` 和 `qwen`。
- API Key 只从函数参数或环境变量读取，不写入配置文件或 SQLite。
- 支持 Provider Base URL 覆盖。

## 4. API 层：`api/`

### `api/app.py`

这是本地 HTTP 边界，核心对象包括：

- `AppRuntime`
  - 当前 Project 根目录。
  - 当前 `ModelClient`。
  - 默认模型名称。
  - Tokenizer。
  - Project 级 `ProjectDatabase`。
  - 所有 Chat 和 Workflow 共享的 `ModelScheduler`。
- `ChatRequest` / `ChatResponse`
  - Chat HTTP Schema。
- `ExtractionRequest`
  - 包含 InputManifest、恢复目录、Chunk 大小、并发数和运行模式。
- `_TaskState`
  - API 进程内的后台提取状态。
  - 保存当前 Stage、状态、Dataset 路径、错误和阶段确认 Event。

当前 API：

| 方法 | 路径 | 职责 |
|---|---|---|
| GET | `/api/health` | 本地服务健康检查 |
| GET | `/api/project` | Project 信息、模型状态和 Dataset 摘要 |
| GET | `/api/datasets` | 列出已生成 Dataset |
| GET | `/api/datasets/{directory}/dataset` | 读取并校验 DatasetBundle |
| POST | `/api/chat` | 执行或继续一次 AgentHarness 对话 |
| GET | `/api/chats/{session}/messages` | 读取持久化聊天记录 |
| POST | `/api/extractions` | 创建后台文字提取任务 |
| GET | `/api/extractions/{task}` | 读取提取状态 |
| POST | `/api/extractions/{task}/confirm` | 允许逐阶段模式继续 |

API 还注册了 `start_text_extraction` Agent 工具，使模型可以在普通 Chat 中构造 InputManifest 并启动固定 Workflow。

生产构建时，API 优先读取 Python 包中的 `web_dist`；源码开发时回退到仓库的 `web/dist`，并为 React Router 风格路径提供前端回退。

## 5. 通用 Agent：`agent/`

### `agent/harness.py`

`AgentHarness` 是开放式任务的最高层入口，职责包括：

1. 创建或继续 Chat Session。
2. 从 SQLite 重建模型消息历史。
3. 选择本轮可见工具并构造 `ModelRequest`。
4. 通过共享 `ModelScheduler` 调用模型。
5. 识别并执行模型返回的 Tool Calls。
6. 将 Tool Observation 作为 `tool` 消息送回下一轮模型。
7. 保存消息、工具调用和模型 Token 使用量。
8. 在模型不再调用工具时返回最终文本。

`AgentHarness` 不理解 Dataset 合并、修正等业务类型。模型通过读取当前文件、执行工具和验证结果完成这些任务。

`AgentRunResult` 返回：

- Chat Session 标识。
- 最终文本内容。
- 本轮执行的工具调用数量。

### `agent/tools.py`

工具系统由以下概念组成：

- `RegisteredTool`：Tool Definition、Effect 和 Handler 的组合。
- `ToolObservation`：统一的成功、失败、不确定性和结果结构。
- `ToolDispatcher`：工具注册、筛选、授权回调和异常归一化入口。
- `build_workspace_tools()`：创建内置 Project 文件工具。

内置工具：

| 工具 | Effect | 功能 |
|---|---|---|
| `read_file` | read | 读取 UTF-8 文件 |
| `list_files` | read | 递归列出 Project 文件 |
| `write_file` | write | 创建或覆盖 UTF-8 文件 |
| `replace_text` | write | 精确替换文件内容 |
| `validate_dataset` | read | 使用 DatasetBundle Schema 校验结果 |
| `delete_file` | delete | 删除 Project 内单个文件 |

所有路径在执行前解析为绝对路径，并验证仍位于 Project 根目录内。API 当前不提供额外 Approval Callback，等价于默认允许 Agent 发出的 Project 内工具操作；`ToolDispatcher` 已保留可插入授权策略的接口。

## 6. 固定文字 Workflow：`workflow/`

### `workflow/dataset_build.py`

`DatasetBuildWorkflow` 是文字提取最高层测试和运行边界。

主要配置由 `WorkflowConfig` 提供：

- 默认模型和 Stage 模型覆盖。
- TextChunk / PlotChunk Token 目标。
- 模型尝试次数。
- 模型并发数。
- 相邻剧情上下文句数。
- 角色筛选描述预算。
- 角色档案分层摘要预算。
- Prompt 文件覆盖。
- ShareGPT System Prompt 模板。

固定流程如下：

1. **InputManifest**
   - 新任务创建 Dataset 目录并写入 `input_manifest.json`。
   - 恢复任务优先读取已经冻结的 InputManifest。
2. **Chunk**
   - 读取有序文字资源。
   - 使用 `TextChunker` 生成 `text_chunks.jsonl`。
3. **Plot extraction**
   - 每个 TextChunk 是独立模型 Unit。
   - 可携带同一章节相邻 Chunk 的少量句子作为边界上下文。
   - 结果写入 `plot_extractions.jsonl`。
4. **Character filter**
   - 汇总全部 CharacterCandidate。
   - 一次全局模型调用筛除模糊名称、合并角色并选择正式名称。
   - 结果写入 `character_filter.json`。
5. **Plot reconstruction**
   - `PlotRebuilder` 根据模型边界恢复原文，而不是保存模型摘要。
   - 直接切分为接近对话预算的 PlotChunks。
   - 结果写入 `plots.json`。
6. **Dialogue extraction**
   - 每个 PlotChunk 是独立模型 Unit。
   - `DialogueAligner` 将对白向前定位到原文。
   - `ConversationAssembler` 确定性地将每个 Plot 组装成一个 Conversation。
   - 原始模型结果写入 `dialogue_extractions.jsonl`。
7. **Character profiles**
   - 按剧情顺序汇总角色 Description 并去除完全重复项。
   - 超过预算时先并行生成局部摘要，再生成最终角色档案。
   - 结果写入 `character_profiles.jsonl`。
8. **DatasetBundle**
   - 组装 Resources、CharacterProfiles 和 Conversations。
   - 写入 `dataset.json` 和角色纯文本文件。
9. **ShareGPT export**
   - 从 DatasetBundle 和 Plots 派生训练样本。
   - 写入 `exports/sharegpt/`。

### Workflow 失败与恢复

模型 Unit 的持久化记录统一包含：

```json
{
  "input": {"path": "...", "index": 0},
  "prompt_file": "plot_extraction.txt",
  "provider": "deepseek",
  "model": "deepseek-chat",
  "result": {}
}
```

- 成功 Unit 的 `result` 是符合固定 Schema 的对象。
- 最终失败 Unit 使用 `result: {}` 占位，不删除所在位置。
- 同一 Stage 的独立 Unit 会继续完成，不因单个 Unit 失败而全部取消。
- Stage 完成后，只要存在空结果，就抛出 `IncompleteStageError` 并停止依赖 Stage。
- 恢复时程序扫描固定文件，只重跑缺失或空结果，不让 LLM 决定恢复点。

`stage_callback` 是自动模式与逐阶段确认模式之间的接口。Workflow 只负责在完成大 Stage 后等待 Callback；是否暂停由 API 或其他调用方决定。

## 7. 领域模型：`domain/`

### `domain/models.py`

该文件定义 RAR 文件产物的 Pydantic Schema，是 Workflow、工具校验、导出和 API 之间的公共契约。

主要模型分组：

#### 输入

- `InputResource`
- `InputManifest`
- `TextChunk`

#### 剧情与角色提取

- `CharacterCandidate`
- `PlotExtractionResult`
- `CharacterFilterRequest`
- `CharacterGroup`
- `CharacterFilterResult`

#### 来源引用

- `PlotRef`
- `CharacterRef`
- `PlotChunkRef`
- `TextChunkSpanRef`

领域关系不使用额外 ID，而是使用工作区相对文件路径和数组 Index。

#### 重建剧情与对话

- `PlotChunk`
- `Plot`
- `PlotsDocument`
- `Utterance`
- `Conversation`

`Utterance` 强制要求至少具有 PlotChunk 引用。`Environment` 内容必须使用非空 `*(...)*` 格式，且不绑定 TextChunk 文字区间。

#### 最终 Dataset

- `CharacterProfile`
- `DatasetResource`
- `DatasetBundle`

所有 Domain Model 默认拒绝未知字段，并进行工作区相对路径、索引范围、空值和领域关系校验。

## 8. 确定性文字处理：`text/`

### `text/tokenizer.py`

- `Tokenizer`：统一 Token 计数协议。
- `CharacterTokenizer`：按字符计数，主要用于完全确定性的测试和显式回退。
- `TikTokenTokenizer`：生产 Tokenizer；延迟加载 tiktoken Encoding。

### `text/chunking.py`

- `split_sentences()`：识别中英文句末标点、换行和 CJK 闭合符号。
- `TextChunker`：按目标 Token 数组合完整句子。
- 当单句超过预算时保留完整句子，不为了严格满足大小而截断。
- TextChunk 保存原始文字、文件、文件内 Index、Token 数和开放 Meta。

### `text/characters.py`

`CharacterResolver` 负责全局角色筛选前后的确定性部分：

- 按清理后的首选名称精确预聚合 Candidate。
- 在描述预算内构建一次性模型请求。
- 验证正式名称和别名必须来自输入 Candidate。
- 验证 Candidate Index 不得跨 Group 重复。
- 对模型结果执行精确别名交集二次合并。

它不执行模糊名称匹配；语义筛选由全局角色 Prompt 和模型完成。

### `text/plot_rebuilder.py`

`PlotRebuilder` 将模型给出的剧情首尾句恢复成原始文本 Plot：

- 支持跨 TextChunk 的开放 Plot。
- 将 `plots: []` 解释为当前 Chunk 剧情连续。
- 不跨不同 Meta 章节合并。
- 根据准确边界截取 TextChunk 原文。
- 映射 CharacterCandidate 到最终 CharacterGroup 引用。
- 生成携带 TextChunk Span 的 PlotChunks。
- 直接按目标大小进行句子感知切分，不经过额外 512 Token 中间层。

### `text/dialogue.py`

- `RawUtterance` / `DialogueExtractionResult`：模型对白输出 Schema。
- `DialogueAligner`：
  - 为每个 PlotChunk 从偏移 0 建立向前游标。
  - 先精确匹配，再对足够长的对白执行保守模糊匹配。
  - 成功后推进游标，避免重复对白定位到前一个位置。
  - Environment 不搜索原文也不移动游标。
  - 定位失败时保留 Utterance，只省略 TextChunk Span。
- `ConversationAssembler`：按 PlotChunk Index 排序 Batch，并为一个 Plot 生成一个 Conversation。

## 9. 模型层：`models/`

### `models/base.py`

定义 Provider 中立契约：

- `ModelMessage`
- `ToolDefinition`
- `ToolCall`
- `ModelUsage`
- `ModelRequest`
- `ModelResponse`
- `ModelClient` Protocol

Workflow 和 AgentHarness 只依赖这些契约，不直接依赖 DeepSeek 或 Qwen SDK。

### `models/openai_compatible.py`

`OpenAICompatibleClient`：

- 调用 `/chat/completions`。
- 支持普通消息和 Function Tool Calls。
- 保留 Provider 特有参数 `provider_options`。
- 解析文本、Tool Calls、Finish Reason 和 Token Usage。
- 可注入 `httpx.AsyncClient`，用于测试或自定义连接行为。

### `models/structured.py`

`StructuredModelGateway` 负责模型结构化输出：

- 统一尝试次数预算。
- 去除 Markdown Code Fence。
- 提取 JSON Object/Array。
- 修复安全的尾随逗号。
- 使用目标 Pydantic Schema 校验。
- 支持附加领域 Validator。
- 最终失败抛出 `StructuredOutputError`。

### `models/scheduler.py`

`ModelScheduler` 是进程级共享并发控制：

- 按 `(provider, model)` 创建独立 Semaphore。
- 支持默认并发上限和模型特定覆盖。
- 被 AgentHarness 和多个 DatasetBuildWorkflow 共用。

因此多个并行 Dataset 不会绕过 Provider/Model 并发限制。

### `models/scripted.py`

`ScriptedModelClient` 是测试模型：

- 按预设队列返回文本、ModelResponse 或异常。
- 保存所有 ModelRequest，供测试验证调用次数和请求结构。
- 不访问真实 API。

## 10. Prompt：`prompts/`

### `prompts/loader.py`

- 定义固定 Prompt Stage 名称。
- `PromptCatalog` 从默认目录或配置覆盖路径读取 Prompt。
- `PromptTemplate.filename` 会写入模型产物的 `prompt_file`。
- Prompt 是可替换文本，不是 Workflow 状态，也不改变固定 Schema。

### `prompts/defaults/`

当前默认 Prompt：

- `plot_extraction.txt`
- `character_filter.txt`
- `dialogue_extraction.txt`
- `character_profile.txt`

这些 Prompt 是可运行占位实现。后续可以独立迭代 Prompt 质量，但替换后仍必须返回对应 Stage 的固定结构。

## 11. 存储：`storage/`

### `storage/artifacts.py`

`DatasetArtifactStore` 管理一个 Dataset 的稳定文件布局：

```text
datasets/<dataset-name>/
├── input_manifest.json
├── work/
│   ├── text_chunks.jsonl
│   ├── plot_extractions.jsonl
│   ├── character_filter.json
│   ├── plots.json
│   ├── dialogue_extractions.jsonl
│   └── character_profiles.jsonl
├── dataset.json
├── characters/
├── exports/
└── report/
```

职责包括：

- Dataset 名称安全化和重名数字后缀。
- 所有 Artifact 路径必须位于 Project 内。
- UTF-8 JSON/JSONL 读写。
- 临时文件、Flush、fsync 和原子替换。
- 根据 Input Locator 和空结果判断待恢复 Unit。

### `storage/database.py`

`ProjectDatabase` 管理 `<project>/.rar/rar.sqlite3`。

SQLite 表：

- `chat_sessions`
- `messages`
- `tool_calls`
- `workflow_runs`
- `events`
- `artifact_refs`
- `model_usage`

当前已直接用于 Chat Session、消息、Tool Calls 和模型 Usage。Domain 内容仍以 Dataset 文件为准；SQLite 不参与 TextChunk 或 PlotChunk 完成判断。

## 12. 导出：`export/`

### `export/sharegpt.py`

`ShareGPTExporter` 从 DatasetBundle 和 Plots 生成训练数据：

- 每个 Conversation 为其中每个正式发言角色生成一个训练样本。
- 将目标角色消息变成 `assistant`，其余角色和 Environment 变成 `user`。
- 所有内容保持 `角色：内容`，Environment 保持 `Environment：*(...)*`。
- 连续同 Speaker 和连续同 Role 会确定性合并。
- Assistant 消息写入 `loss: true`。
- 删除末尾没有目标回复的 User 消息。
- 生成 `all.jsonl` 和每角色 JSONL。

`ShareGPTExportReport` 返回输出目录、总样本数和每角色样本数。

## 13. 扩展协议：`extensions.py`

当前预留的内部扩展接口：

- `ResourceAdapter`：漫画、扫描件、视频、音频等资源预处理。
- `ToolProvider`：内置工具、MCP 或其他外部工具注册。
- `Exporter`：ShareGPT 之外的训练格式。
- `ReportGenerator`：HTML 等人类可读报告。
- `WorkflowExtension`：未来稳定、长期运行的其他媒体 Workflow。

扩展协议仅规定边界，不要求文字核心理解具体媒体实现。

## 14. 模块依赖方向

推荐保持以下依赖方向：

```text
api / cli / main
      │
      ├── agent ───────┐
      │                ├── models
      └── workflow ────┘
             │
             ├── text
             ├── prompts
             ├── storage
             └── export
                    │
                  domain
```

关键约束：

- `domain` 不依赖 API、Agent 或 Workflow。
- `text` 负责确定性算法，不直接访问模型 Provider。
- `workflow` 组合模块，不把算法重复写在编排代码中。
- `agent` 不直接实现每种业务任务。
- `models` 不理解 Plot、Conversation 或 DatasetBundle。
- `storage/artifacts.py` 管理领域文件；`storage/database.py` 管理运行元数据。

## 15. 常见修改位置

| 需求 | 主要修改位置 |
|---|---|
| 修改 Dataset JSON 结构 | `domain/models.py`，随后更新 Workflow、导出和测试 |
| 调整 TextChunk 策略 | `text/chunking.py`、`text/tokenizer.py` |
| 调整剧情重建规则 | `text/plot_rebuilder.py` |
| 调整对白定位 | `text/dialogue.py` |
| 调整角色筛选确定性规则 | `text/characters.py` |
| 修改固定 Stage 顺序 | `workflow/dataset_build.py` |
| 新增或替换 Prompt | `prompts/defaults/` 或 `WorkflowConfig.prompt_overrides` |
| 新增模型 Provider | 实现 `ModelClient`，并更新 `settings.py` |
| 新增 Agent 工具 | 创建 `RegisteredTool` 或实现 `ToolProvider` |
| 新增训练格式 | 实现新的 `Exporter` |
| 新增 HTML 报告 | 实现 `ReportGenerator` |
| 新增漫画/视频流程 | 实现 `ResourceAdapter`，必要时实现 `WorkflowExtension` |
| 新增 API | `api/app.py`，业务逻辑应下沉到对应模块 |

## 16. 当前边界

当前 `src` 已完成文字核心和通用 Agent 基础运行时，但以下内容尚未实现为生产功能：

- 漫画、扫描件、视频和纯音频资源处理。
- 用户只提供作品名称时的网络搜索与下载。
- HTML 可视化报告。
- MCP 和 Computer Use Provider 的实际注册实现。
- 远程部署、账号、多人协作和密钥存储。

这些能力应通过现有扩展边界逐步加入，不应破坏 DatasetBundle、文件恢复和通用 AgentHarness 的职责划分。
