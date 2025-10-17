# Shots on Goal

Shots on Goal is an experimental orchestrator for automating Bazel migrations (and other repository goals) with large language models. It treats a migration as a loop of goal → plan → execute → validate, recording every attempt so you can inspect what happened or resume later.

## Key capabilities

- **Ping‑pong automation** – Two LLM “teammates” alternate between implementation and decomposition. When one model fails validation, control flips to the other to review the work, create sub‑goals, or repair the failure.
- **Structured history** – Every session stores goals, attempts, tool calls, and git metadata in SQLite so you can audit what the agent did or restart from a previous state.
- **Branch-safe editing** – Each goal and attempt operates on its own git branch/worktree; successful attempts are merged back into the goal branch automatically.
- **Configurable tools** – The agent interacts with the repo via a curated toolset (read/write files, `bazel` commands, `ripgrep`, etc.) that runs inside a container for reproducibility.

## Prerequisites

- **Python**: 3.9+ (project is developed against Python 3.13).
- **Container runtime**: Either [`container`](https://github.com/jmorganca/container-cli) or Docker must be available on the PATH. The runtime is auto-detected.
- **LLM access**: The project uses [simonw/llm](https://github.com/simonw/llm) with the OpenRouter plugin. Configure your API key via `llm keys set openrouter` before running.
- **Dependencies**: Managed via `uv` (preferred) or any other virtual environment tool.

## Project layout

```
shots-on-goal/
├── shots_on_goal.py    # Main orchestrator script
├── test_shots_on_goal.py
├── Dockerfile          # Container image with Bazelisk & supporting tools
├── shots-on-goal.bazelrc
└── ...
```

## Getting started

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-org/shots-on-goal.git
   cd shots-on-goal
   ```

2. **Create a virtual environment & install dependencies (via `uv`)**

   ```bash
   uv sync
   ```

   This installs the `llm` CLI, OpenRouter plugin, and other Python dependencies listed in `pyproject.toml`.

3. **Configure LLM access**

   Set your OpenRouter key (supports both variable names):

   ```bash
   export OPENROUTER_API_KEY=your-token
   export OPENROUTER_KEY=$OPENROUTER_API_KEY  # optional alias
   ```

4. **Build (or pull) the runtime container**

   ```bash
   docker build -t shots-on-goal .
   ```

   You can also use the provided `build_image.sh` helper if you prefer the `container` CLI.

## Running the orchestrator

Invoke the orchestrator with a human-readable goal and a path to a local git repository:

```bash
uv run python shots_on_goal.py \
  --model-a openrouter/anthropic/claude-sonnet-4.5 \
  --model-b openrouter/openai/gpt-5-codex \
  "Migrate example-python-project to Bazel using bzlmod" \
  ../example-python-project
```

What happens:

1. A session directory is created under `sessions/` with a fresh SQLite database (`goals.db`).
2. The root goal is recorded and a dedicated git branch/worktree is created.
3. Model A attempts the goal; if validation fails, the orchestrator flips to Model B to review, decompose, and attack the resulting sub-goals.
4. All attempts, tool calls, diffs, validation logs, and branch names are persisted for later inspection.

You can resume or inspect sessions later with:

```bash
uv run python shots_on_goal.py --list
```

## Running tests

The project ships with a Python test suite that exercises the database helpers, git manager, tool execution, and orchestration scaffolding.

```bash
# Inside your virtual environment
uv run python test_shots_on_goal.py
```

> Note: Some tests interact with git and the container runtime. Ensure Docker (or the `container` CLI) is running before executing the suite.

## Troubleshooting tips

- **LLM failures**: Double-check your OpenRouter key and quota; re-run with `--verbose` for detailed logging.
- **Container errors**: Confirm the container runtime is available (`container --version` or `docker --version`) and that the `shots-on-goal` image exists.
- **Session restarts**: Use `sessions/session-*/goals.db` to inspect previous attempts or resume a session manually.

For deeper tracing, run with `--verbose` to elevate logging to DEBUG level.

---

Happy migrating! Contributions, issue reports, and ideas are welcome—this orchestrator is deliberately experimental and evolving quickly.
