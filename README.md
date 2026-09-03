# RAR Agent

RAR (Read And Retrieve) is a local-first Web agent that extracts traceable role-play
dialogue datasets from text resources. The V1 implementation targets personal,
non-commercial use and exports maintained DatasetBundle files to ShareGPT JSONL.

RAR combines two execution paths behind one chat interface:

- `DatasetBuildWorkflow` runs the fixed text pipeline: TextChunk → Plot extraction →
  candidate aggregation plus formal-name/profile generation → Plot reconstruction →
  dialogue extraction → DatasetBundle → ShareGPT.
- `AgentHarness` handles open-ended inspection, correction, merge, and re-export requests
  through workspace-contained tools. It does not create a dedicated workflow class for
  every task.

The authoritative implementation specification is in
`doc/design/2026-09-01-rar-agent-text-core-implementation-spec.md`.

## Current V1 capabilities

- UTF-8 text files and ordered multi-resource InputManifests, with volume/chapter-aware
  section splitting before sentence-complete token chunking.
- File-based checkpoints with `result: {}` placeholders and deterministic resume.
- Source references from every dialogue back to PlotChunk and, when alignment succeeds,
  TextChunk character spans.
- One profile call per deterministically aggregated character; the selected formal name is
  moved first and every other observed name is retained as an alias.
- DatasetBundle, plain-text character profiles, and per-character ShareGPT JSONL.
- DeepSeek and Qwen through provider-neutral OpenAI-compatible adapters.
- Project-local SQLite for chat, tool-call, usage, and operational records only.
- Project-scoped chat management with lazy creation, archive/restore, pagination, and
  background Agent runs that survive page switching.
- Temporary approval requests for risky tools, per-chat cancellation, complete-turn context
  trimming, and one global model-concurrency scheduler shared by Chat and Workflow.
- Automatic and stage-confirmed extraction modes through the local API and Web UI.
- Responsive local React chat interface; no remote deployment or account system.

Manga, scan, and video workflows are extension points and are not implemented by the text
core yet. HTML reports also remain an optional post-V1 renderer.

## Development setup

Python 3.12 and pnpm are required.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
pnpm --dir web install
pnpm --dir web run build
```

Set one provider key in the environment. Keys are read at runtime and are never written to
RAR configuration or SQLite.

```powershell
$env:DEEPSEEK_API_KEY = "..."
# or: $env:QWEN_API_KEY = "..."
```

Start the local service for the current Project directory:

```powershell
.venv\Scripts\rar-agent serve --project . --provider deepseek --model deepseek-chat
```

Then open `http://127.0.0.1:8765`. The interface can start without a key for browsing local
results, but model-backed chat and extraction return a clear configuration error.

## InputManifest example

```json
{
  "name": "Example Work",
  "meta": {"language": "zh"},
  "resources": [
    {
      "path": "resources/book.txt",
      "resource_type": "text",
      "display_name": "Volume 1, Chapter 1",
      "narrative_order": 0,
      "meta": {"volume": 1, "chapter": 1}
    }
  ]
}
```

The paths are relative to the Project directory. A completed Dataset is written under
`datasets/<dataset-name>/`; intermediate JSON/JSONL files remain inspectable and are the
authoritative recovery source.

Existing intermediate files are intentional checkpoints. After changing chunk rules or
prompt templates, start a new extraction (or explicitly remove the old Dataset work files)
to observe the new behavior; resuming an existing Dataset reuses its completed records.

For a low-cost end-to-end smoke test, enable `debug` in the Web extraction dialog or API
request, or pass `--debug` to `rar-agent extract`. Chunking still processes the complete
input, while plot extraction and dialogue extraction each process at most five units;
character filtering, profile generation, DatasetBundle assembly, and ShareGPT export run
normally on that limited result.

## Verification

```powershell
.venv\Scripts\python -m pytest
.venv\Scripts\python -m ruff check src tests
.venv\Scripts\python -m mypy src
pnpm --dir web run typecheck
pnpm --dir web run test
pnpm --dir web run build
```
