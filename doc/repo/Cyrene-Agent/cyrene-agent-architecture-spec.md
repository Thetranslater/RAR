# Cyrene-Agent 架构现状与演进规范

> 文档类型：架构分析与实现规范  
> 适用对象：Agent Runtime、工具执行、权限隔离、记忆和恢复能力  
> 分析基线：2026-09-01 代码状态

## Problem Statement

Cyrene-Agent 是一个 Electron 桌面 Agent，已经同时具备聊天、工作、代码、学习、工具调用、记忆、Sandbox 和任务恢复能力，但这些能力跨越 Renderer、Main Process、Agent Runtime、工具注册、权限系统和多个持久化存储。

当前最需要明确的问题不是单一功能缺失，而是系统边界容易被误解：

- ChatLoop 与 Harness 的执行语义不同。
- 权限策略、Sandbox 和工具自身的路径校验并非同一层。
- Session history、Run checkpoint 和长期记忆分别持久化，存在语义不同步的可能。
- 工具调用已经具备副作用分类、重试和 uncertain effect 处理，但并非所有工具都经过同一种安全闸门。
- Run 可以在重启后被识别为 interrupted，但恢复仍由用户触发，不是自动继续执行。
- 主循环是命令式 while loop，而不是通用工作流图；Plan Mode 只是其上的辅助状态机。

如果没有一份统一规范，后续开发容易出现重复的调度逻辑、错误的自动重试、错误的副作用推断、越过 workspace 边界的文件操作，以及把“已持久化”误认为“已成功执行”。

## Solution

以 Harness 执行边界和 Tool Dispatcher 作为最高层 seam，统一描述 Agent 的运行、工具调用、权限判断、输出持久化、checkpoint 和恢复语义；保留当前 Electron Composition Root、AG-UI IPC、ChatLoop、记忆分层和现有工具注册机制。

整体采用以下原则：

1. Renderer 只负责交互和事件呈现，Agent、工具、权限和持久化由 Main Process 负责。
2. ChatLoop 继续服务于无工具的普通聊天；需要工具的模式统一进入 Harness。
3. Harness 使用持续循环：构造上下文、请求模型、执行工具、回写 transcript、checkpoint，直到成功、取消、超时或错误。
4. 工具必须声明风险、模式、副作用、验证策略和并发安全性。
5. 非幂等副作用在结果不确定时必须停止自动重放，并保留可供恢复的证据。
6. 完整工具输出与模型可见 preview 分离保存。
7. 恢复必须由持久化事实驱动，不能从“进程曾经调用过函数”推断工具已经成功。
8. 权限策略决定是否允许调用，Sandbox 决定受限 Shell 的资源边界，工具自身负责额外的参数和目标校验；三者必须保持显式分工。

## User Stories

1. As a desktop user, I want to send a message through a single chat interface, so that I do not need to understand which runtime executes it.
2. As a desktop user, I want streamed text, reasoning, tool progress, Todo changes, and terminal status to arrive through one event channel, so that the UI can present one coherent run.
3. As a desktop user, I want Chat mode without tools to remain fast and lightweight, so that ordinary conversation does not incur Harness overhead.
4. As a desktop user, I want Work, Code, and Learn mode to use the same reliable execution semantics, so that tools, retries, checkpoints, and cancellation behave predictably.
5. As a desktop user, I want a Work or Code session to require an explicit workspace binding, so that file operations do not silently target an unintended directory.
6. As a desktop user, I want risky operations to be allowed, denied, or sent for approval according to the selected permission level, so that I retain control over machine changes.
7. As a desktop user, I want Shell commands to be rejected before spawning when the Sandbox is unavailable for a write or unknown-effect command, so that a failed safety boundary cannot become a fail-open execution.
8. As a desktop user, I want read-only commands to degrade gracefully when the Sandbox is explicitly disabled, so that development environments can remain usable.
9. As a desktop user, I want catastrophic commands to be rejected regardless of permission level, so that full access does not remove basic safety guards.
10. As a desktop user, I want long-running Shell commands to have idle and total time limits, so that a stuck process cannot keep an Agent run alive forever.
11. As a desktop user, I want transient tool failures to retry with bounded exponential backoff and jitter, so that temporary service failures recover without creating retry storms.
12. As a desktop user, I want non-idempotent operations to avoid automatic retries, so that sending, publishing, deleting, or pushing cannot happen twice merely because the result was unclear.
13. As a desktop user, I want an interrupted run to be visibly marked as interrupted, so that the UI never presents an incomplete task as successful.
14. As a desktop user, I want to resume an interrupted run explicitly, so that I can decide whether the task should continue after reviewing its state.
15. As a desktop user, I want an interrupted external side effect to be represented as unknown, so that the Agent checks or asks before attempting it again.
16. As a desktop user, I want large tool outputs to be summarized for the model while remaining fully retrievable, so that useful evidence is preserved without exhausting context.
17. As a desktop user, I want recent session messages to survive application restart, so that conversation history remains available independently of a running Agent.
18. As a desktop user, I want durable profile and recent-state memory to be injected automatically when appropriate, so that the Agent can maintain continuity.
19. As a desktop user, I want semantic long-term memories to be searchable without injecting every memory into every prompt, so that retrieval stays relevant and token-efficient.
20. As a desktop user, I want explicit memory writes to respect locked profile fields and evidence rules, so that the Agent cannot silently invent personal facts.
21. As a developer, I want all built-in and MCP tools to share one registry contract, so that tool discovery, mode filtering, risk handling, and execution are consistent.
22. As a developer, I want tool execution to receive a structured context containing run, conversation, mode, workspace, signal, and permission information, so that tools do not reconstruct runtime state ad hoc.
23. As a developer, I want read-only tools to be eligible for safe parallel execution only when they explicitly declare concurrency safety, so that performance improvements do not introduce races.
24. As a developer, I want tool results to be committed in model call order even when execution is parallel, so that transcript semantics remain deterministic.
25. As a developer, I want the Composition Root to own lifecycle ordering while feature modules own business behavior, so that startup and shutdown remain testable.
26. As a developer, I want background initialization failures to be represented as degraded capabilities, so that optional services do not block the primary chat experience.
27. As a maintainer, I want tests to verify externally observable terminal states, recovery facts, permission outcomes, and tool evidence, so that implementation refactors do not change safety semantics accidentally.

## Implementation Decisions

### 1. Runtime and process boundary

- Electron Main Process owns Application lifecycle, Agent Runtime, tools, permissions, Sandbox, RAG, memory, channels, schedulers, and persistent stores.
- Renderer owns React state, message presentation, tool cards, Todo presentation, approval UI, and user-triggered recovery actions.
- Preload exposes only typed APIs for window control, chat persistence, AG-UI run/cancel/event handling, approval resolution, and related UI operations.
- AG-UI IPC remains the external seam for starting a run and receiving run events.
- The application startup remains phase-oriented: pre-ready, shell, core, background, and on-demand capabilities.
- The Composition Root coordinates lifecycle and dependency construction; it must not absorb feature-specific business logic.

### 2. Execution modes and main loop

- Chat without enabled tools uses a single-request ChatLoop with streaming and narrowly scoped stream/image fallbacks.
- Chat with explicitly enabled tools, Work, Code, and Learn use Harness.
- Harness is an imperative while loop, not a general workflow graph.
- Plan Mode remains an auxiliary state machine that restricts available operations and changes prompt/tool policy; it does not replace Harness.
- Each Harness round must preserve the following order:
  1. Build prompt layers and context usage.
  2. Compact context when the configured threshold is reached.
  3. Request the model.
  4. Persist the Assistant message and tool calls in the in-memory transcript.
  5. Execute and observe tools.
  6. Append model-facing tool results.
  7. Checkpoint the run.
  8. Continue or emit a canonical terminal result.
- `ask_user` is an exclusive interaction. Other calls from the same model turn become `not_executed`.
- Cancellation, timeout, runtime error, and success must produce distinct terminal semantics; consumers must not infer terminal status merely from a boolean terminated flag.

### 3. Tool registry and dispatch seam

- Tool Registry is the single catalog for built-in tools and dynamically discovered MCP tools.
- Tool definitions must expose stable identity, input schema, enabled state, mode availability, risk, side-effect kind, verification policy, context requirements, concurrency safety, and execution function.
- Mode filtering occurs before a run starts. Chat tools require explicit opt-in; Code, Work, and Learn use their mode allowlists and settings overrides.
- Tool Dispatcher is the highest common execution seam:
  - parse and validate arguments;
  - block duplicate uncertain-effect fingerprints;
  - check permission;
  - execute through the ledger where applicable;
  - normalize success/failure/unknown/not-executed outcomes;
  - persist full output;
  - emit an observation and append the transcript result.
- Built-in Harness tools are handled through the same dispatch contract but do not require an external Tool Definition.
- MCP annotations are advisory metadata. Missing or contradictory effect metadata must resolve conservatively to unknown or external side effect.
- A future unified policy guard should be placed at the Tool Dispatcher seam. Existing standalone policy helpers must not be treated as effective enforcement until they have an actual call path.

### 4. Permissions, filesystem, network, and Sandbox

- Permission levels remain `project-read-only`, `read-only`, `scoped`, `per-action`, and `full`.
- Permission risk categories remain safe, filesystem read, filesystem write, shell, network, and input control.
- Permission decides allow, ask, or deny. It does not replace resource isolation.
- Shell commands use a pre-spawn execution plan. The plan can be sandboxed, direct, or rejected; rejected plans must never reach process spawn.
- The Shell classifier remains conservative: redirection, pipes, command chaining, unknown commands, and ambiguous developer tools may be classified as write or unknown.
- Sandbox readiness failure is fail-closed for write and unknown-effect Shell operations.
- Direct filesystem tools must explicitly validate absolute paths and, where the operation is workspace-scoped, enforce containment against the resolved workspace root.
- Document tools must keep generated artifacts inside the bound workspace or their documented fallback directory.
- Network tools use permission checks and their own request validation. They must not be described as protected by the Shell Sandbox unless they actually execute inside it.
- Tool-specific timeouts and process-tree termination are part of the execution contract, not optional UI behavior.

### 5. Retry and error semantics

- Retry decisions depend on both error category and side-effect kind.
- Retryable categories are transient, timeout, rate-limited, and narrowly defined partial failure.
- Fatal, runtime-safety, permission, invalid-argument, not-found, semantic failure, and non-idempotent unknown effects are not automatically retried.
- Backoff remains bounded and jittered; every wait must observe the run AbortSignal.
- ChatLoop may fall back from streaming to non-streaming, but this is transport fallback rather than general retry.
- Tool timeout and cancellation must preserve partial output and effect uncertainty when available.
- Tool adapters should return structured failure categories instead of relying on legacy string prefixes. Legacy result parsing remains compatibility behavior only.

### 6. Session, memory, and RAG

- Session history is short-term conversational state. It stores the complete user/assistant/tool conversation independently of whether a run is active.
- L0 stores durable core profile fields.
- L1 stores recent goals, preferences, current project, and maintenance counters.
- L2 stores semantic conversational memories with evidence, source references, lifecycle status, access statistics, conflict links, and RAG synchronization status.
- Always-on context injects the appropriate L0/L1 information and activated Worldbook content.
- Semantic retrieval uses RAG rather than injecting all L2 records into every prompt.
- `user_memory` is semantic search; `read_memory` is direct inspection; `write_memory` is an explicit mutation path through the memory manager.
- Automatic memory maintenance runs asynchronously after successful runs and must not block the primary response.
- Memory maintenance cadence remains six turns for judging, five for resolver work, twenty for reflection/compression, and fifty for decay.
- L0 writes require explicit user attribution and field allowlisting; locked profile state prevents overwrite.
- Local memory persistence remains authoritative when RAG embedding synchronization fails; synchronization status must remain visible to maintenance logic.
- Worldbook activation state and L2 memory state are related but distinct domains and must not be conflated.

### 7. Checkpoint, output persistence, and recovery

- A durable Harness snapshot contains transcript messages, Todo and uncertain effects, tool output references, round count, cache epoch, and request/environment fingerprints.
- Run state is stored separately from chat history so that an interrupted execution can be recovered without mutating ordinary conversation history prematurely.
- Session files are authoritative for run state; the index is an acceleration structure and may be rebuilt or corrected.
- Checkpoints use atomic replacement. Hot-path index updates may be debounced, but terminal status must be flushed immediately.
- Full tool output is stored separately from the model preview and referenced by a stable `tool-result` identifier.
- On application restart, persisted `running` runs become `interrupted`.
- Recovery is explicit and user initiated.
- Recovery repairs the transcript by representing unresolved non-idempotent calls as unknown and unresolved read calls as not executed.
- Recovery must validate workspace compatibility and report provider, model, or tool-catalog changes.
- No recovery path may automatically replay an unresolved non-idempotent side effect.
- Task subagents use the same Harness contract but maintain their own task session persistence and lifecycle status.

### 8. Observability and lifecycle

- AG-UI events are the Renderer-facing presentation stream; durable run state is the recovery-facing fact store.
- Canonical terminal events must be emitted at most once, even when upstream completion and error callbacks race.
- Startup readiness, Agent runtime state, and run terminal state are separate concepts and must not share one overloaded status field.
- Optional background services may degrade independently; the primary chat path must remain available where possible.
- Shutdown must flush safety-critical state and terminate active external processes within bounded time.

## Testing Decisions

- Tests should assert externally observable behavior and stable contracts, not private helper call order.
- The highest-value Harness tests cover:
  - no-tool success;
  - multi-round tool execution;
  - context compaction;
  - exclusive `ask_user` behavior;
  - parallel read tools with deterministic commit order;
  - cancellation during model request and tool execution;
  - timeout terminal behavior;
  - fatal and unknown-effect halts;
  - uncertain-effect fingerprint blocking.
- Tool Dispatcher tests cover permission denial, approval timeout, malformed arguments, missing tools, normalized failures, output truncation, full-output references, and ledger deduplication.
- Sandbox tests cover catastrophic command rejection, direct read fallback, fail-closed write behavior, unavailable Bash, process timeout, output truncation, and process-tree termination.
- Recovery tests cover restart-to-interrupted conversion, workspace mismatch, provider/model/tool changes, unresolved non-idempotent calls, unresolved read calls, and idempotent restoration.
- Session and memory tests cover atomic persistence, schema migration, locked L0 fields, evidence requirements, RAG synchronization failure, conflict resolution, and maintenance cadence.
- Renderer integration tests cover one terminal event, approval request/response, interrupted-run notices, resume/takeover behavior, and event routing by run ID.
- Existing tests in the repository already provide prior art for Harness cancellation, Run Store recovery, tool output storage, memory maintenance, permission policy, and mode-specific tool behavior; new tests should extend those seams rather than introduce parallel test harnesses.

## Out of Scope

- Replacing Electron, React, AG-UI, or the existing IPC contract.
- Replacing the imperative Harness loop with LangGraph, a workflow DAG, or a generic orchestration engine.
- Introducing a third-party dependency-injection container.
- Rewriting existing Agent persona, prompt, channel, music, Live2D, or TTS behavior.
- Automatically replaying uncertain external side effects after crash recovery.
- Treating all network requests as Sandbox-protected without an explicit implementation.
- Merging session history, run checkpoints, and long-term memory into one storage format.
- Making every tool parallel by default.
- Forcing Todo usage for every short task.
- Publishing this document to an external issue tracker.

## Further Notes

### Current architecture status

The repository already contains the main building blocks described above: phased application startup, AG-UI IPC, ChatLoop/Harness routing, Tool Registry, permission levels, SRT integration, retry policy, uncertain-effect handling, session persistence, memory layers, RAG, run checkpoints, recovery repair, and full-output storage.

### Main architectural gaps

1. Filesystem path containment is not uniform across direct filesystem tools, document tools, Git, and Shell.
2. Network tools use direct requests and are not automatically covered by Shell Sandbox boundaries.
3. ChatLoop has weaker checkpoint and recovery semantics than Harness.
4. A declarative execution-policy helper exists but is not currently the universal enforcement seam.
5. Session history, run state, and memory state can be individually consistent while still becoming semantically divergent.
6. Synchronous checkpoint writes improve durability but may block the Electron Main Process under heavy tool activity.
7. Unknown effect classification is intentionally conservative and may reduce automation until tools provide stronger metadata.

### Recommended evolution order

1. Make Tool Dispatcher the explicit universal policy seam.
2. Standardize workspace containment and network boundary declarations.
3. Separate read-only observation from side-effect deduplication in the execution ledger.
4. Add tool completion semantics and self-verifying artifact evidence where needed.
5. Measure and, if necessary, move checkpoint writes away from the hottest synchronous path without weakening crash guarantees.
6. Add end-to-end tests that follow a real user run from IPC entry through terminal event and durable state.

