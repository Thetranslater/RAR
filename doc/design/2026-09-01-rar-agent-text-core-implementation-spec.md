# RAR Agent V1 核心架构与文字对话提取实现规格

> 文档状态：文字核心设计已冻结，可进入实现  
> 产品名称：RAR Agent（Read And Retrieve）  
> 产品定位：面向个人爱好者的非商用、本地优先角色对话数据生产工具  
> V1 核心：从文字资源生成可溯源的角色扮演对话 DatasetBundle，并导出 ShareGPT 训练数据  
> 扩展方向：漫画、扫描件、带音频视频和仅作品名称输入  
> 架构基线：固定提取 Workflow + 通用 AgentHarness + 可组合工具 + 文件型领域产物 + 确定性验证

## Problem Statement

人工从小说等长篇文字资源中抽取角色扮演对话，需要持续阅读原文、识别剧情边界、重建跨段剧情、判断说话人、保留环境旁白、整理角色档案、记录来源并转换训练格式。资源规模增加后，这项工作耗时、重复，并且难以保持一致性和可追溯性。

用户需要一个以 Chat 为主要入口的本地 Agent。用户可以提供文件、文件夹、媒体资源或作品名称，Agent 在指定工作目录内查找和读取资源，按照稳定流程提取数据，并把中间结果和最终结果保存为可检查、可恢复和可由 Git 管理的文件。

项目同时面对两类不同问题：

- 大规模对话提取具有稳定顺序、明确阶段、大量可并行单元和可程序化恢复条件，适合固定 Workflow。
- 修改、合并、修复、查询和重新导出无法提前穷举，适合由通用 AgentHarness 根据用户意图动态读取文件和调用工具。

如果所有任务都实现为专用 Workflow，项目会产生重复的状态类、恢复机制和任务处理器；如果所有任务都交给自由 Agent，则会失去批处理、确定性恢复和结构验证。因此，RAR 必须分离固定数据生产平面和开放式 Agent 交互平面。

系统必须满足以下约束：

- 面向个人爱好者和非商用用途，但功能完整度和可靠性向成熟生产工具对齐。
- 一个 Project 只代表一个工作目录及其安全边界，不绑定单个 Dataset 或 ExtractionRun。
- 一个 Project 可以包含多个资源、Dataset 和并发 ExtractionRun。
- Workspace 内读取默认允许；Workspace 之外的访问必须被 Sandbox 阻止或获得明确扩展授权。
- Workflow 自动运行模式只控制阶段是否暂停，不等价于权限模式。
- 模型输出必须通过结构和领域校验，不保存或依赖通用置信度分数。
- 中断、限流、断网和单个模型任务失败不能迫使整个长任务从头重跑。
- Git 由用户控制，RAR 不主动初始化、提交、合并或恢复仓库。
- 大型文本不得整体塞入模型上下文，必须通过 Chunk、批处理、有限上下文和文件引用控制规模。

## Solution

构建一个本地 Web Agent。浏览器中的 Chat UI 面向用户，本地 Python 服务承载 AgentHarness、DatasetBuildWorkflow、模型适配、工具调度、SQLite 运行元数据和文件型领域产物。前端使用 TypeScript；V1 只支持本地服务，不设计远程部署、账户或多人协作。

系统具有两个执行平面：

1. `DatasetBuildWorkflow` 是固定的数据生产流程。它负责输入解析、文字分块、剧情提取、候选角色聚合与档案生成、剧情重建、对话提取、DatasetBundle 组装和 ShareGPT 导出。
2. `AgentHarness` 是通用多轮工具调用循环。修改、合并、修复、检查和重新导出都是普通 Agent 交互，不建立专用 CorrectionRun、MergeRun 或任务处理器层级。

文字核心 Workflow 按以下顺序执行：

1. 解析用户输入，生成有序 InputManifest。
2. 从文字资源生成接近配置 Token 预算、优先保持完整句子的 TextChunk。
3. 并行执行每个 TextChunk 的 Plot 提取，得到 Plot 边界、连续状态和 CharacterCandidate。
4. 按剧情候选的首个名称确定性聚合同一角色的名称和 description；每个聚合角色调用一次角色档案模型，同时选择正式名称并生成档案。
5. 根据模型选择重排名称：正式名称位于首位，其余输入名称全部保留为别名。
6. 根据原文边界和连续状态确定性重建 Plot，并直接切成用于对话提取的大型 PlotChunk。
7. 并行从每个 PlotChunk 提取 Utterance；对话提取不接收前后 PlotChunk 上下文。
8. 按 PlotChunk index 确定性组装 Conversation，使一个 Plot 对应一个 Conversation。
9. 生成 DatasetBundle；中间产物不复制进 Bundle，只通过相对路径和 index 引用。
10. 使用确定性 Exporter 把 DatasetBundle 转换成按目标角色拆分的 ShareGPT 训练样本。

Prompt 是可替换组件而不是 Workflow 定义。项目提供可运行的默认占位 Prompt；用户可以替换 Prompt 文件，但替换内容必须遵守该阶段固定的输入输出 Schema。改变 Schema 属于 WorkflowExtension，不属于普通 Prompt 替换。

HTML 报告是次要派生能力。V1 只保留 ReportGenerator 接口和报告输出位置，不把 HTML 实现作为文字核心完成条件。用户通过 Chat 指示修改结果时，AgentHarness 读取相关文件、执行工具并运行 Validator，不进入提取 Workflow。

## User Stories

1. As a personal hobbyist, I want to submit a text resource and receive role-play dialogue training data, so that I do not need to extract dialogue manually.
2. As a personal hobbyist, I want the product to state that it is for non-commercial use, so that its intended scope remains clear.
3. As a user, I want to use RAR through a local Web UI, so that I do not need a heavy desktop application.
4. As a user, I want Chat to be the primary interface, so that extraction and later maintenance use one interaction model.
5. As a user, I want one working directory to define a Project, so that filesystem access has a clear boundary.
6. As a user, I want one Project to contain multiple resources, datasets and runs, so that Project is not confused with one extraction task.
7. As a user, I want workspace reads available by default, so that analysis is not interrupted by repetitive approval prompts.
8. As a user, I want Sandbox enforcement even in permissive operation mode, so that the Agent cannot escape the selected workspace.
9. As a user, I want external paths to require an explicit scope extension, so that the Agent cannot silently inspect unrelated files.
10. As a user, I want Workflow automation and side-effect permission configured separately, so that one-click execution does not redefine security boundaries.
11. As a user, I want staged confirmation at major Workflow stages, so that internal Chunk calls do not require individual confirmation.
12. As a user, I want an automatic mode, so that a long extraction can run without business-stage pauses.
13. As a user, I want to interrupt a run and later request continuation, so that long work remains controllable.
14. As a user, I want the application to inspect existing stage artifacts before resuming, so that completed units are not repeated.
15. As a user, I want one failed unit represented explicitly, so that result ordering remains stable and recovery is deterministic.
16. As a user, I want independent extraction units to continue after another unit fails, so that one error does not waste completed work.
17. As a user, I want the Workflow to stop before dependent finalization when a stage remains incomplete, so that partial results are not presented as complete.
18. As a user, I want to provide a file, folder, media resource or work title, so that RAR can start from the information I possess.
19. As a user, I want RAR to search the Project before using the network, so that local resources are preferred.
20. As a user, I want RAR to choose an obtainable resource when one exists, so that discovery does not stop unnecessarily.
21. As a user, I want the actual ordered input set recorded, so that an ExtractionRun can be understood and resumed.
22. As a user, I want long text split near a configurable Token budget, so that model inputs remain manageable.
23. As a user, I want Chunking to prefer complete sentences, so that arbitrary Token boundaries do not damage the text.
24. As a user, I want volume, chapter and other literary structure preserved in meta, so that different genres remain representable.
25. As a user, I want TextChunk to remain compact, so that it is easy to inspect and reuse as the first processing source.
26. As a user, I want Plot detected before dialogue extraction, so that one Conversation corresponds to a coherent narrative event.
27. As a user, I want Plot boundaries grounded in exact source sentences, so that reconstruction remains traceable.
28. As a user, I want Plots crossing TextChunks reconstructed deterministically, so that Chunk boundaries do not fragment conversations.
29. As a user, I want chapter boundaries respected, so that unrelated sections are not reconstructed into one Plot.
30. As a user, I want the final source Chunk forced to a finished Plot when no next context exists, so that a model truncation mistake does not leave an impossible open tail.
31. As a user, I want candidate character names and descriptions collected during Plot extraction, so that later character work reuses the same analysis.
32. As a user, I want vague names such as narrator, I or classmate filtered before profiles are generated, so that unrelated roles are not merged globally.
33. As a user, I want each deterministically aggregated character processed by one profile call, so that formal-name selection and profile generation happen together.
34. As a user, I want the filtering model to see names and limited descriptions rather than full Plot text, so that the call stays focused and bounded.
35. As a user, I want the selected formal name and aliases to come from extracted names, so that filtering does not invent identities.
36. As a user, I want deterministic alias-overlap merging after model filtering, so that obvious duplicate groups are resolved without another model call.
37. As a user, I want a reconstructed Plot split directly into dialogue-sized PlotChunks, so that the RLFF state-oriented small-Chunk layer is not copied.
38. As a user, I want dialogue extraction to consume only the current PlotChunk, so that extraction inputs remain independent and parallelizable.
39. As a user, I want known characters supplied to dialogue extraction, so that speaker assignment can use formal names, aliases and descriptions.
40. As a user, I want unknown and temporary speakers preserved, so that the model is not forced to fabricate a known identity.
41. As a user, I want narration and objective scene information retained as Environment, so that role-play context is not lost.
42. As a user, I want Environment content retain the `*(...)*` representation, so that narration remains distinguishable.
43. As a user, I want model Utterance order preserved, so that the output reflects the extraction result without programmatic reordering.
44. As a user, I want source location to use the extracted order as a forward search cursor, so that repeated dialogue resolves to later occurrences naturally.
45. As a user, I want a failed source match to preserve the Utterance, so that source alignment does not erase useful dialogue.
46. As a user, I want every Utterance point to its PlotChunk, so that even an unaligned dialogue remains reviewable.
47. As a user, I want exact or sufficiently strong fuzzy matches point to TextChunk spans, so that original dialogue can be inspected quickly.
48. As a user, I want no confidence field stored, so that uncalibrated scores do not masquerade as quality guarantees.
49. As a user, I want one Plot assembled into one Conversation without another model call, so that final grouping is deterministic.
50. As a user, I want formal speaker aliases normalized to the preferred name, so that final conversations use consistent identities.
51. As a user, I want every observed name retained after formal-name selection, so that alternate names remain available as aliases.
52. As a user, I want all descriptions for one formal character combined in Plot order, so that profiles reflect narrative development.
53. As a user, I want exact duplicate descriptions removed but contradictions preserved, so that evolving information is not silently discarded.
54. As a user, I want character descriptions bounded deterministically before profile generation, so that requests stay within the configured context budget.
55. As a user, I want every formal character to have a non-empty profile, so that DatasetBundle is usable for role training.
56. As a user, I want profiles exported as plain text, so that they can be reused outside the structured Bundle.
57. As a user, I want DatasetBundle contain only maintained final results and references, so that intermediate processing data remains separate.
58. As a user, I want domain relationships use relative paths and indexes rather than generated IDs, so that artifacts remain human-readable.
59. As a user, I want one DatasetBundle to generate replaceable training formats, so that training data is not duplicated in the maintained result.
60. As a user, I want ShareGPT supported first, so that V1 produces immediately usable fine-tuning data.
61. As a user, I want one training sample generated per Conversation and target formal speaker, so that each character receives role-specific supervision.
62. As a user, I want every exported Utterance formatted as speaker name plus content, so that user and assistant messages remain structurally consistent.
63. As a user, I want Environment exported with its explicit speaker name, so that training data does not mix narration with unnamed text.
64. As a user, I want assistant messages marked with `loss: true`, so that the training target is explicit.
65. As a user, I want system prompts rendered from character, profile and Plot variables, so that training context is configurable.
66. As a user, I want combined and per-character ShareGPT files, so that I can train globally or inspect one role.
67. As a user, I want Prompt files replaceable without changing Workflow code, so that extraction behavior can be tuned independently.
68. As a user, I want replaced Prompts continue to obey stable Schemas, so that recovery and validation remain reliable.
69. As a user, I want HTML reporting left as an extension point, so that the core implementation is not blocked by a secondary presentation feature.
70. As a user, I want to describe a correction through Chat, so that I do not need to edit JSON manually.
71. As a user, I want the Agent to determine every affected file, so that related outputs can be updated together.
72. As a user, I want modifications validated after tool execution, so that success is supported by deterministic evidence.
73. As a user, I want dataset merging handled as an ordinary Agent task, so that it does not require a dedicated task framework.
74. As a user, I want an erroneously separate Dataset remain mergeable later, so that automatic name matching need not be perfect.
75. As a user, I want V1 merge append Conversations without semantic deduplication, so that merge behavior remains predictable.
76. As a user, I want merge avoid moving or reorganizing original resources, so that the operation stays limited to maintained results.
77. As a user, I want Git controlled explicitly by me, so that the Agent does not alter version history unexpectedly.
78. As a user, I want the application warn when Git protection is absent, so that I understand recovery limits.
79. As a user, I want local Git without mandatory LFS or remote hosting, so that personal use remains simple.
80. As a user, I want different models selectable by stage, so that cost and capability can be balanced.
81. As a user, I want DeepSeek and Qwen supported first, so that the initial implementation can be validated with available providers.
82. As a developer, I want provider-specific behavior behind adapters, so that Workflow code does not depend on one model vendor.
83. As a developer, I want fake scripted model adapters, so that tests do not depend on live model behavior.
84. As a developer, I want one Tool Registry and Dispatcher, so that Shell, direct tools and MCP capabilities share execution semantics.
85. As a developer, I want only relevant tools sent to the model, so that Agent context remains focused.
86. As a developer, I want SQLite store lightweight operational metadata rather than domain artifacts, so that files remain the source of truth for extraction recovery.
87. As a maintainer, I want tests at DatasetBuildWorkflow and AgentHarness seams, so that internal implementations can evolve without parallel test frameworks.
88. As a maintainer, I want deterministic stage tests built from RLFF fixtures, so that reconstruction, location and export behavior can be verified without live models.
89. As a future extension author, I want media adapters preserve spatial or temporal native structures, so that non-text resources are not forced into TextChunk or a universal ContentSegment.

## Implementation Decisions

### 1. Product scope, deployment and technology

- The product is local-only, non-commercial and intended for individual hobbyists.
- The V1 user interface is a browser-based Chat UI; no Electron or native desktop shell is used.
- The backend is Python with FastAPI, Pydantic, SQLAlchemy, Alembic and one SQLite database per Project.
- The frontend is React, TypeScript and Vite.
- Python dependencies use uv; frontend dependencies use pnpm.
- One launcher starts the local Web service and Worker responsibilities.
- Local HTTP handles ordinary requests. Streaming Chat, tool progress and run status use one local streaming mechanism selected during implementation.
- Remote deployment, public hosting, accounts and multi-user collaboration are not planned.

### 2. Core terminology

- Project: one selected working directory and filesystem safety boundary.
- Dataset: one maintained logical result for a work or selected resource group.
- ExtractionRun: one execution of the fixed DatasetBuildWorkflow.
- AgentRun: one generic multi-round AgentHarness interaction.
- Stage: a user-visible major Workflow phase.
- Unit: one independently executable and recoverable item inside a Stage, usually a Chunk or character.
- InputManifest: the ordered resource set actually used by an ExtractionRun.
- TextChunk: the first processing source produced from text.
- PlotFragment: one model-extracted Plot boundary observation inside a TextChunk.
- CharacterCandidate: one raw name, alias and description observation from Plot extraction.
- ResolvedCharacter: one deterministically aggregated character whose formal name and profile were returned by the profile model.
- Plot: one reconstructed continuous narrative unit corresponding to one Conversation.
- PlotChunk: one dialogue-model-sized processing window inside a Plot.
- Utterance: one extracted role line or Environment narration item.
- Conversation: the ordered Utterances belonging to one Plot.
- CharacterProfile: the final formal character name, aliases, profile and Plot references.
- DatasetBundle: the maintained structured Dataset result used by Exporters.
- RunArtifact: an intermediate or diagnostic file outside DatasetBundle.

### 3. Two execution planes

- DatasetBuildWorkflow is the highest seam for fixed extraction.
- AgentHarness is the highest seam for open-ended modification, merge, repair, query and re-export.
- The Workflow owns stable dependencies and stage gates; the model does not dynamically invent the extraction stage graph.
- AgentHarness owns iterative reasoning and tool selection for ordinary Chat requests.
- Tools, validators, exporters and stateless services provide deterministic capabilities to both planes.
- V1 does not introduce task-specific CorrectionRun, MergeRun, ChangePlanner, ChangeSetExecutor or handler hierarchies.
- A specialized Workflow is added only when an operation has stable long-lived asynchronous dependencies comparable to extraction.

### 4. AgentHarness

- AgentHarness is an imperative multi-round model-and-tool loop, following the general architecture demonstrated by Cyrene-Agent.
- Each round assembles stable policy, current workspace and Dataset context, recent conversation, persisted tool observations and a filtered tool catalog.
- The model may return text, Tool Calls, or both. Tool observations feed the next round.
- A tool-free final response ends the turn.
- Data-changing work may report completion only after the applicable deterministic Validator succeeds.
- A user message such as “继续任务” starts a new Agent turn. The model reads conversation history, current files, stored tool observations and Git diff when available; the system does not automatically replay uncertain mutations.
- Todo may exist as an Agent notebook, but it is not business state or completion evidence.
- Modification and merge use ordinary file, Shell, Git-read and domain validation tools. They do not resume through extraction stage artifacts unless the user explicitly asks to resume an ExtractionRun.

### 5. Tool Registry, Dispatcher and Sandbox

- Tool Registry is the single registration interface for built-in tools and future MCP-backed or external tools.
- Tool definitions declare name, description, input Schema, effect kind, risk, context requirements, concurrency behavior and executor.
- The model receives only tools relevant to the current Stage or Agent intent.
- Tool Dispatcher is the universal seam for argument validation, workspace containment, permission handling, execution, output persistence and normalized observations.
- Shell, direct Tool Call and MCP implementations may coexist behind Tool Dispatcher.
- Full tool output is stored outside model context; the model receives a bounded preview and a retrievable reference.
- Workspace containment is enforced independently of permission mode.
- General Agent operation supports configurable approval behavior. V1 may default to allowing operations within the selected Workspace to reduce repeated approvals, while external scope expansion remains explicit.
- DatasetBuildWorkflow uses separate Run configuration for stage confirmation and side-effect approval behavior.
- A Stage write allowance covers only its predefined output targets. Writes to any other path are rejected unless separately authorized.
- Related network searches may share one Run-scoped grant.
- Overwrite and deletion remain distinguishable effects even in permissive modes.

### 6. Model provider layer

- Workflow and retry logic do not depend on one model provider.
- A common ModelClient interface covers generation, streaming, tool use and structured output.
- Provider adapters preserve provider-specific parameters and capabilities instead of forcing every provider into the least common denominator.
- DeepSeek and Qwen are the initial production adapters.
- Tests use a scripted fake adapter and never require live model calls.
- One global default model exists, with optional Stage overrides for Plot extraction, character profile generation, dialogue extraction, vision and derivative work.
- One global scheduler enforces provider and model concurrency limits across runs.
- Each extraction Unit has at most three model attempts by default.
- Deterministic parsing or structure repair is attempted before consuming another model call.
- Adapter or SDK hidden retries are disabled or included in the same attempt budget.
- Provider, model and actual prompt file are stored with model-produced RunArtifacts.
- `prompt_file` contains the actual configured Prompt filename. No ambiguous Prompt version label or Prompt hash is required in V1.

### 7. SQLite and operational persistence

- Each Project owns one SQLite database.
- SQLite stores Chat sessions, Agent turns, Tool Calls, overall Workflow status, lightweight events, artifact references and model usage.
- SQLite does not store TextChunk, Plot, Conversation, CharacterProfile or DatasetBundle content.
- SQLite does not store per-Chunk completion as the authoritative recovery source.
- Fixed Workflow recovery is derived from files at predefined Dataset paths.
- Generic AgentHarness recovery uses persisted transcript, tool observations, current files and optional Git diff.
- A lightweight log records errors and operational outcomes; no compliance audit subsystem is built.

### 8. Dataset artifact layout

Each independent extraction result uses one Dataset directory with this stable artifact contract:

```text
datasets/<dataset-name>/
├── input_manifest.json
├── work/
│   ├── text_chunks.jsonl
│   ├── plot_extractions.jsonl
│   ├── character_profiles.jsonl
│   ├── plots.json
│   └── dialogue_extractions.jsonl
├── dataset.json
├── characters/
├── exports/
└── report/
```

- No `runs/` hierarchy is introduced in V1.
- No independent `conversations.json` is required; Conversation assembly is deterministic from dialogue extraction results.
- A colliding Dataset directory name receives a numeric suffix rather than overwriting an existing Dataset.
- A later ordinary Agent interaction may merge separately produced Datasets.
- Domain artifacts use UTF-8 JSON, JSONL or text and remain inspectable by users and Git.
- Operational timestamps, model timings and run history do not belong in DatasetBundle, Plot or TextChunk.

### 9. InputManifest and resource discovery

- InputManifest freezes the actual ordered inputs selected after local search or approved network discovery.
- Input entries identify local workspace path, resource type, display name, narrative order and open metadata.
- Work title, aliases, series, volume, edition, language and media type may live in manifest or Dataset metadata; no separate WorkMeta file is required.
- V1 recovery does not compute or compare full content hashes and does not attempt to detect changed source content.
- File path and index are the V1 identity relationship. Strong stable identities and automatic rechunk invalidation are deferred.
- Resource discovery searches the Project first, then performs an approved network search or download when necessary.
- RAR chooses an obtainable source when one exists and asks for help only when no usable source can be found or downloaded.
- InputManifest is not a long-term Resource Catalog and does not monitor resources after extraction.

### 10. Workflow modes and Stage gates

- Staged-confirmation mode pauses only at major business Stages, not every model Unit.
- Automatic mode runs the fixed Stage graph without business-stage pauses.
- Permission behavior is configured independently from staged or automatic Workflow behavior.
- In staged mode the Agent presents the completed Stage result for confirmation before continuing.
- Stage output writes target only the fixed files declared for that Stage.
- Independent Units inside extraction, dialogue and profile Stages run concurrently within scheduler limits.
- A failed Unit does not cancel independent Units in the same Stage.
- When all Units finish, any `result: {}` placeholder marks the Stage incomplete.
- An incomplete Stage reports failures and stops before dependent Stages.
- On continuation, code scans the fixed artifact, reruns only missing or empty results and then resumes the Stage graph.
- Fixed Workflow recovery is entirely programmatic; an LLM does not decide which completed Chunk should be rerun.

### 11. TextChunk contract

TextChunk contains exactly the following domain fields:

| Field | Type | Meaning |
|---|---|---|
| `file` | string | Workspace-relative source file path |
| `index` | non-negative integer | Zero-based index within that file |
| `text` | string | Exact Chunk content |
| `token_count` | non-negative integer | Actual size under the configured tokenizer |
| `meta` | JSON object | Open literary structure metadata |

- `file + index` locates a TextChunk.
- JSONL order follows InputManifest resource order, then file-local index.
- Meta may contain nested JSON values such as volume, chapter, part, act, scene or episode.
- Meta does not contain characters, Plots, speakers, model analysis or confidence.
- Tokenizer identity and target size belong in effective Run configuration or InputManifest, not every TextChunk.
- Initial target size is approximately 5120 Tokens and remains configurable.
- Chunking prefers complete sentences and may slightly exceed the target rather than splitting a sentence only to satisfy the budget.

### 12. Plot extraction contract

- Plot extraction operates independently on each TextChunk.
- It may receive a configurable small number of sentences from adjacent TextChunks only when they share the same source section. Adjacent context is used for boundary judgment, not extracted as current content.
- The default adjacent Plot context may begin at three sentences and remain configurable.
- Each `plot_extractions.jsonl` line identifies its input by the TextChunk artifact path and index and stores `prompt_file`, provider, model and `result`.
- A valid result contains exactly `characters`, `plots` and `state`.
- `state` is `finished` or `truncated`.
- `plots` is an ordered array of source-grounded `[start, end]` sentence pairs. Either boundary may be null where continuity requires it.
- A CharacterCandidate contains `names`, non-empty `description` and nullable `plot_indexes`.
- Names is a non-empty array. The first name is the candidate's most complete extracted name.
- `plot_indexes` points to local entries in the result's `plots`. Multiple indexes are allowed.
- `plot_indexes: null` means the observation cannot be assigned more narrowly and applies to the current continuous Plot context.
- When `plots` is empty, all candidate `plot_indexes` values are null.
- Plot boundary strings must locate in the current TextChunk text.
- A final TextChunk with no `next` context cannot remain open. If the model returns `truncated` and a null final end, parsing forces `finished` and uses the current Chunk's last sentence as the end.
- Invalid structure is deterministically repaired when safe, otherwise retried within the Unit's three-attempt budget.
- Final failure preserves the line and input locator with `result: {}`.

### 13. Plot reconstruction rules

- `plots: []` means the model considers the Chunk continuous, equivalent to a null start and null end observation.
- When an open Plot exists, a continuous Chunk is appended in full.
- At the start of a source section, a null start resolves to the first sentence of the current TextChunk.
- At the end of a source section, any open Plot is forcibly closed at the final sentence.
- Reconstruction never joins across incompatible chapter or other section boundaries established by TextChunk meta.
- Plot boundaries are located against exact TextChunk text; reconstruction restores source text rather than a model summary.
- No separate normalization or repair artifact is written. Deterministic correction rules are part of reconstruction, and final corrected output appears only in `plots.json`.
- Reconstructed Plot has no full `text` field. Its content is represented by ordered PlotChunks.
- Each Plot contains `index`, open `meta`, `character_refs` and `chunks`.
- A character reference points to `character_profiles.jsonl` plus the resolved character index.
- Each PlotChunk contains `index`, `text`, `token_count` and source references.
- Reconstructed Plot text is split directly near the dialogue budget, initially about 5120 Tokens, using sentence-aware boundaries.
- The RLFF state-extraction split into roughly 512-Token Chunks and later merge step is not used.

### 14. Character aggregation, formal-name selection and profile generation

- This Stage occurs after Plot extraction and before Plot reconstruction.
- Raw candidates are traversed in deterministic source order and assigned transient array indexes. These indexes are not domain IDs.
- Candidates with the same trimmed first name are aggregated. Names retain first-seen order and exact duplicate descriptions are removed; no fuzzy or semantic identity merge occurs.
- The configured total description budget is divided across aggregated characters. Plot text is not sent.
- Each aggregated character is one independent parallel model item and uses `character_profile.txt` exactly once.
- Model input is `{"character":{"names":[...],"description":"..."}}`.
- Model output is `{"profile":{"name":"formal name","content":"profile text"}}`.
- The selected formal name must occur in input `character.names`; invented names fail validation and retry within the shared attempt budget.
- The selected name is moved to the first position. Every other supplied name is retained in original order and later becomes a DatasetBundle alias.
- `profile.content` is the final non-empty profile text; there is no later profile-summary model Stage.
- Each `character_profiles.jsonl` record identifies the source candidate indexes in `plot_extractions.jsonl`, plus prompt file, provider, model and result.
- Final failure writes `result: {}` and blocks Plot reconstruction.

### 15. Character-to-Plot mapping

- Plot reconstruction maps each CharacterCandidate's local `plot_indexes` to final reconstructed Plot indexes.
- A nullable or unassigned observation is associated with every reconstructed Plot touched by its source TextChunk.
- Final Plot stores only `character_refs`; it does not copy raw descriptions.
- Dialogue extraction resolves only generated character profiles referenced by the current Plot.
- For each resolved character it supplies the reordered names and aggregated description.
- CharacterProfile `plot_refs` are derived from final Plots influenced by the character's candidate descriptions.

### 16. Dialogue extraction contract

- Each PlotChunk is one independent dialogue extraction Unit.
- Dialogue extraction receives only current PlotChunk text and current Plot character information. It does not receive previous or next PlotChunk content.
- Model output contains one ordered `utterances` array. Each item has non-empty `speaker` and `content`.
- The model is instructed to emit source order, but RAR does not reject, reorder or retry output merely because alignment suggests a different order.
- Known formal names or aliases are mapped to the resolved character's formal name.
- Unknown and temporary names are preserved as strings and do not automatically become CharacterProfiles.
- Narration, objective actions and environment use the exact speaker `Environment`.
- Environment content must remain in the `*(...)*` form in model output, intermediate artifacts and DatasetBundle.
- A valid no-content result is `utterances: []`.
- A result containing only Environment is valid and non-empty.
- Model, parsing or validation failure is represented by `result: {}` and is distinguishable from a successful empty array.
- Each `dialogue_extractions.jsonl` line identifies `plots.json`, Plot index and PlotChunk index, and stores prompt file, provider, model and result.

### 17. Dialogue source alignment

- Every final Utterance has at least one PlotChunk source reference.
- Environment receives only the PlotChunk reference and is not aligned to TextChunk text.
- Non-Environment dialogue is aligned using a forward source cursor initialized to zero for each PlotChunk result.
- Exact matching searches only from the current cursor to the end of PlotChunk text.
- A successful match advances the cursor to the matched end.
- Environment does not search and does not advance the cursor.
- A failed dialogue match preserves the Utterance, leaves the cursor unchanged and omits only the TextChunk span reference.
- Fuzzy matching is attempted only after exact matching for sufficiently long dialogue.
- Initial policy treats fewer than six non-whitespace characters as exact-only and uses a configurable conservative fuzzy threshold initially set to 90.
- Fuzzy failure does not trigger a model retry and no similarity or confidence value is stored.
- Text spans use character offsets in `TextChunk.text`; start is inclusive and end is exclusive.
- A PlotChunk reference contains workspace-relative artifact path, Plot index and PlotChunk index.
- A TextChunk span reference contains workspace-relative artifact path, TextChunk index, start and end.
- Source reference variants are distinguished by their fields; no reference ID or type discriminator is required.

### 18. Conversation assembly

- Conversation assembly is deterministic and does not call a model.
- Dialogue batches are sorted by PlotChunk index.
- Utterances retain model order inside each batch.
- One Plot produces one Conversation.
- Final Utterance indexes are assigned sequentially within the assembled Conversation and are not stable IDs.
- Conversation contains only a Plot reference and its ordered Utterances.
- PlotChunk batch boundaries do not need another field because every Utterance already points to its source PlotChunk.
- If any dialogue Unit has `result: {}`, the dialogue Stage is incomplete and assembly does not produce a final DatasetBundle.

### 19. Final CharacterProfile assembly

- No model call occurs in this step. Formal name, aliases and profile text already come from Stage 14.
- After Plot reconstruction, character references are used to derive final `plot_refs` deterministically.
- CharacterProfile contains only `name`, `aliases`, `profile` and `plot_refs`.
- No character ID is persisted.
- Plain-text exports use the character's DatasetBundle index plus a filesystem-safe name to avoid collisions.

### 20. DatasetBundle contract

DatasetBundle has the following top-level fields:

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | positive integer | Concrete persisted Schema revision; initial value is 1 |
| `name` | non-empty string | User-visible Dataset or work name |
| `meta` | JSON object | Open Dataset-level metadata |
| `resources` | ordered array | Local resources used by the Dataset |
| `characters` | CharacterProfile array | Final formal characters |
| `conversations` | Conversation array | Final Plot-level dialogue results |

- A Resource contains workspace-relative `path`, `media_type` and open `meta`.
- Resource order follows narrative input order.
- DatasetBundle does not store Run IDs, Dataset IDs, timestamps, retries, model usage or training samples.
- DatasetBundle does not copy TextChunk, Plot or PlotChunk text.
- Relationships use workspace-relative paths and indexes.
- Top-level Artifact registry is omitted because embedded references already identify required files.
- Meta accepts nested JSON and is inherited or displayed without requiring RAR to understand unknown keys.
- Training formats and reports are derived artifacts and can be regenerated from DatasetBundle and referenced RunArtifacts.

### 21. Prompt components

- Default Prompt files exist for Plot extraction, dialogue extraction and combined formal-name/profile generation.
- Initial Prompt content may be generated as implementation placeholders and refined independently.
- Prompt replacement is configured by file.
- Prompt text is not a persisted business entity and does not alter Workflow recovery semantics.
- Every replacement must preserve the Stage's fixed input/output contract.
- Tests validate Schema behavior and observable results, not exact Prompt wording.

### 22. ShareGPT Exporter

- ShareGPT is the only V1 training exporter. Additional formats are future deterministic Exporters.
- Export reads DatasetBundle, resolves Plot text through Conversation `plot_ref`, and renders a configured system template.
- Supported system template variables are `character`, `profile` and `plot`.
- Template variables may be omitted. V1 does not require RLFF tasks, random-block or rearrange syntax.
- For each Conversation, one sample is generated for every formal character that speaks in that Conversation.
- A formal character that does not speak is not a target sample.
- Temporary speakers never become assistant targets.
- The target formal character's Utterances become assistant messages.
- Every other character and Environment become user content.
- Before role mapping, consecutive Utterances from the same speaker are merged.
- Every exported Utterance line uses `speaker：content` with a full-width colon.
- Environment is exported as `Environment：*(...)*`.
- Target assistant content also retains its formal name prefix.
- After role mapping, consecutive messages with the same role are joined with newline separators.
- Trailing user-only messages are removed so that a sample ends with assistant.
- Samples without an assistant message are not emitted.
- A sample may begin with assistant.
- Each assistant message contains `loss: true`. System and user messages omit `loss`.
- Each JSONL row has a top-level `messages` array; the RLFF `details/data` wrapper is not copied.
- Export writes one combined JSONL and per-character JSONL files using DatasetBundle index plus safe character name.
- DatasetBundle remains unchanged by export.

### 23. HTML and other derived outputs

- Character profile text files are required deterministic derived outputs.
- HTML is a secondary presentation feature that displays DatasetBundle as a readable report and reduces the need to inspect JSON manually.
- V1 core reserves a ReportGenerator interface and report output directory but does not require HTML implementation.
- Corrected text, translations, summaries and other by-products are optional extensions selected by configuration.
- Derived outputs do not become additional sources of truth.

### 24. Modification, correction and merge

- Users describe modifications in natural language through Chat.
- AgentHarness identifies affected DatasetBundle, RunArtifacts and derived files by reading current files and references.
- The Agent may perform all related edits in one interaction, subject to Tool Dispatcher and Sandbox rules.
- Applicable Validators run before the Agent reports successful completion.
- A post-extraction correction modifies existing results rather than rerunning the whole Workflow unless the user explicitly requests a Stage rerun.
- A correction made during extraction may rerun only the affected Unit or Stage subdivision.
- Merge is an ordinary AgentHarness interaction, not a dedicated Workflow or state model.
- Automatic same-work checking may begin with exact Dataset name matching; a mistaken separate Dataset remains valid and can be merged later.
- V1 merge appends Conversations and combines character/profile information according to user direction.
- V1 does not perform semantic Conversation deduplication.
- V1 merge does not reorganize, move, rename or delete original resources.
- Git records file history when the user chooses to use it; DatasetRevision is not introduced.

### 25. Git behavior

- Git is optional, local and user-controlled.
- RAR may read status and diff as ordinary facts.
- RAR does not initialize a repository, commit, merge branches, restore files or rewrite history without explicit user instruction.
- RAR may suggest a checkpoint after extraction or a substantial modification.
- Local-only Git does not require Git LFS, although tracking large media can increase repository size.
- Temporary frames, caches, partial downloads and reproducible scratch artifacts are excluded from version control by default.

### 26. Extension architecture

- Text is the required core implementation.
- Manga, scans, video with audio and title-only discovery are optional modality-specific V1 extensions.
- No universal ContentSegment is introduced.
- Manga may define page, panel, speech-bubble, OCR and speaker-assignment artifacts.
- Scans may define page and OCR-block artifacts and may enter TextChunk only after reliable linearization.
- Video may define time ranges, transcripts, selected frames, shots, scenes and speaker evidence.
- Modalities converge at Plot or Conversation only where their own Workflow can produce that concept reliably.
- ResourceAdapter, WorkflowExtension, ToolProvider, Exporter and ModelProviderAdapter are internal extension interfaces.
- V1 does not build a third-party plugin marketplace or installation system.
- Multiple language variants within one Dataset are deferred. Language remains available in metadata.

## Testing Decisions

- Tests assert externally observable behavior and persisted contracts, not private helper order, exact Prompt prose or provider implementation details.
- The highest fixed-extraction seam is DatasetBuildWorkflow.
- The highest open-ended interaction seam is AgentHarness.
- No separate test harness is created for correction, merge, repair or re-export.
- All model-dependent tests use scripted fake adapters with deterministic outputs, truncation, malformed JSON and transport errors.
- DatasetBuildWorkflow end-to-end tests begin with InputManifest and assert fixed files, Stage gates, resume behavior and final DatasetBundle.
- Recovery tests prepopulate JSONL with successful rows, `result: {}` rows and interrupted tails, then prove only incomplete Units rerun.
- Recovery tests confirm valid `utterances: []` is not mistaken for failure.
- TextChunk tests cover sentence-aware splitting, per-file indexes, resource order, Token counts, nested meta and multi-file input.
- Plot extraction tests cover multiple local Plots, null boundaries, empty `plots`, character `plot_indexes`, malformed anchors, no-next forced close and three-attempt failure.
- Plot reconstruction tests cover open Plot continuation, source-section boundaries, full continuous Chunks, exact text recovery and direct large PlotChunk splitting.
- Character filtering tests cover exact-name preaggregation, vague-name removal, one global call, context-budget failure, candidate index validation and deterministic alias-overlap merging.
- Dialogue tests cover known aliases, temporary speakers, Environment formatting, valid empty output, pure Environment output and malformed output.
- Dialogue alignment tests cover forward cursor matching, repeated lines, Environment cursor behavior, exact-only short text, fuzzy threshold, failed match preservation and TextChunk span offsets.
- Conversation assembly tests prove PlotChunk ordering, model-order preservation and one Plot to one Conversation without another model call.
- Profile tests cover the exact prompt payload, description ordering, duplicate removal, formal-name validation, alias retention, parallel completion and resume.
- DatasetBundle tests validate the no-ID contract, reference resolution, minimal top-level structure and exclusion of RunArtifacts and runtime metadata.
- ShareGPT tests cover one Conversation by target character, formal speaker prefixes, full-width colon, Environment prefix, assistant `loss: true`, role merging, trailing-user removal, missing-assistant omission and combined/per-character outputs.
- Prompt replacement tests prove that different Prompt files do not change persisted Schema or recovery behavior.
- Tool Dispatcher tests cover workspace containment, configured permission behavior, rejected undeclared Stage targets, network grants, normalized failures and full-output references.
- AgentHarness tests cover no-tool completion, multi-round Tool Calls, file modification, validator-gated success, cancellation, uncertain mutation interruption and continuation from current files.
- Merge tests operate through AgentHarness and prove that no dedicated MergeRun or semantic Conversation deduplicator is invoked.
- Git tests verify read-only inspection by default and mutation only after explicit user request.
- SQLite tests prove that operational records can be recreated without treating SQLite as the source of truth for domain completion.
- Prior art from RLFF supplies fixtures and behavioral references for sentence context, Plot boundaries, Plot reconstruction, forward dialogue alignment and per-character ShareGPT conversion.
- Prior art from Cyrene-Agent supplies architectural references for AgentHarness, Tool Registry, Tool Dispatcher, checkpoints, tool observations and continuation behavior.
- HTML rendering tests are deferred with the HTML implementation; only the ReportGenerator extension contract needs a placeholder test in the core.

## Out of Scope

- Commercial licensing enforcement, paid plans or commercial accounts.
- Remote deployment, public hosting, account systems, team collaboration and role-based access control.
- A native desktop or Electron application.
- A general-purpose assistant unrelated to Dataset construction and maintenance.
- A role-play playground for chatting with extracted characters.
- HTML report implementation as a V1 text-core completion requirement.
- A universal ContentSegment shared by text, spatial and temporal media.
- Generic calibrated confidence scores or a universal quality metric.
- Prompt-injection defense for untrusted source content in the current version.
- A compliance audit subsystem.
- Automatic Git initialization, commit, merge, restore, remote synchronization or mandatory Git LFS.
- DatasetRevision, CorrectionRun, MergeRun, ChangePlanner, ChangeSetExecutor and per-task Handler hierarchies.
- Semantic Conversation deduplication during V1 merge.
- Automatic reorganization or deletion of original resources during merge.
- Full input content hashing, automatic source-change detection and cross-rechunk stable identity in V1.
- Direct original-source evidence for every sentence in a generated CharacterProfile.
- Reusing RLFF state extraction, transition graphs, roughly 512-Token state Chunks or validation-dataset logic.
- Multiple language variants inside one Dataset, automatic cross-language alignment and automatic translation-derived variants.
- Third-party plugin installation, marketplace, dependency resolution and plugin distribution.
- Final manga panel ordering, speech-bubble assignment, scan OCR Workflow, video segmentation, ASR, diarization and multimodal fusion algorithms in this text-core specification.
- Pure audio, arbitrary Web resources, PDF, Word and cloud-drive ingestion in the text core.
- Additional training formats beyond ShareGPT in V1.

## Further Notes

### Confirmed architectural principle

RAR has one generic Agent runtime, not one Agent implementation per task. AgentHarness owns iterative reasoning and tool use; tools own deterministic effects; Validators prove structural success; DatasetBuildWorkflow owns only the stable extraction sequence.

### File-first recovery principle

The fixed text Workflow treats its files as authoritative checkpoints. JSONL line order and `result: {}` placeholders preserve Unit identity and failure location. SQLite provides operational visibility but does not decide which TextChunk, PlotChunk or character is complete.

### Reference principle

V1 intentionally omits domain IDs. Persisted relationships use workspace-relative artifact paths and indexes. This is sufficient while inputs are assumed unchanged; stronger identity and invalidation belong to a later version.

### Important distinction from RLFF

RAR reuses RLFF concepts and code where useful, including sentence-aware Chunking, Plot boundary extraction, deterministic Plot reconstruction, forward-cursor dialogue alignment and per-character SFT conversion. RAR does not copy RLFF's state-extraction layer, small state Chunks, state graphs, task extraction or detailed training wrapper.

### Important distinction from Cyrene-Agent

RAR uses Cyrene-Agent as prior art for the generic Harness loop, tool registration and dispatch, persistence and continuation semantics. RAR does not copy its Electron boundary, dual ChatLoop paths, long-term memory hierarchy, persona system, Plan Mode UI, MCP installation system or unrelated desktop-assistant features.

### Implementation order

1. Define Pydantic domain models and Validators for all fixed artifacts.
2. Implement file layout, JSONL persistence and programmatic recovery.
3. Port or adapt deterministic TextChunk, Plot reconstruction, alignment and ShareGPT behavior from RLFF.
4. Implement ModelClient, DeepSeek and Qwen adapters plus scripted fake adapter.
5. Implement DatasetBuildWorkflow and scheduler around the deterministic stages.
6. Add default replaceable Prompt files.
7. Implement DatasetBundle and ShareGPT Exporter.
8. Implement AgentHarness, Tool Registry and Tool Dispatcher using the Cyrene-Agent architectural seam.
9. Add the local Web Chat UI and run-status streaming.
10. Leave ReportGenerator and media extension interfaces as placeholders for subsequent work.

### Implementation readiness criterion

The text core is implementation-ready when a local Chat request can bind a Project, resolve an InputManifest, execute and resume the fixed text Workflow, produce every declared artifact, aggregate character candidates, select formal names while retaining aliases, generate non-empty CharacterProfiles, reconstruct Plots, extract and align Conversations, assemble DatasetBundle, export per-character ShareGPT data, and later accept a generic Agent request that reads or modifies those files and validates the result without invoking a task-specific modification Workflow.
