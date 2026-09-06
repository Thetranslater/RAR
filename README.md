# RAR Agent

RAR (Read And Retrieve) is a local-first Web agent that extracts traceable role-play
dialogue datasets from text resources. The V1 implementation targets personal,
non-commercial use and exports maintained DatasetBundle files to ShareGPT JSONL.

RAR combines two execution paths behind one chat interface:

- `DatasetBuildWorkflow` runs the fixed text pipeline.
- `AgentHarness` handles open-ended inspection, correction, merge, and re-export requests
  through workspace-contained tools.

## V1 capabilities
Manga, scan, and video workflows are extension points and are not implemented by the text
core yet. HTML reports also remain an optional post-V1 renderer.

## setup

Python 3.12 required.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
pnpm --dir web install
pnpm --dir web run build
```

Set one provider key in the environment. Keys are read at runtime and are never written to
RAR configuration or SQLite.

Start the local service for the current Project directory:

```powershell
.venv\Scripts\rar-agent serve --project . --provider deepseek --model deepseek-chat
```
