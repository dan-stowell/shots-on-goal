# Shots on Goal

Shots on Goal is an experimental orchestrator for automating Bazel migrations (and other repository goals) with large language models. It treats a migration as a loop of goal → plan → execute → validate, recording every attempt so you can inspect what happened or resume later.

## Key capabilities

- **Automated implementation** – An LLM agent works on your goal with access to code editing tools, validation commands, and a generous tool budget.
- **Structured history** – Every session stores goals, attempts, tool calls, and git metadata in SQLite so you can audit what the agent did or restart from a previous state.
- **Branch-safe editing** – Each goal and attempt operates on its own git branch/worktree; successful attempts are merged back into the goal branch automatically.
- **Efficient tooling** – The agent interacts with the repo via a curated toolset (read/write files, batch file operations, `bazel` commands, `ripgrep`, etc.) that runs inside a container for reproducibility.

## Prerequisites

- **Python**: 3.9+ (project is developed against Python 3.13).
- **Container runtime**: Either Docker (`docker`) or Podman (`container`) must be available on the PATH. The runtime is auto-detected.
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

The CLI now always uses the V2 schema and containerised tool execution. Launch it with a goal and repo path:

```bash
uv run python shots_on_goal.py \
  --implementer-model openrouter/anthropic/claude-sonnet-4.5 \
  --max-tools 20 \
  --validation "bazel build //..." \
  --validation "bazel test //..." \
  "Add a docstring to the fibonacci function in mylib.py" \
  ../example-python-project
```

What happens behind the scenes:

1. A V2 SQLite database (`shots-on-goal-v2-<timestamp>.db`) is created to store sessions, goals, attempts, tool calls, etc.
2. The repo is checked out onto a dedicated branch/worktree (per session/goal/attempt).
3. The implementer model attempts to achieve the goal using available tools within the tool budget.
4. Each attempt runs tools inside the specified container (`--image`, default `shots-on-goal:latest`).
5. Validation commands are executed inside the same container; once all pass, the run stops.
6. All artefacts (attempts, results, validation runs) remain in the V2 database for inspection.

Commonly used flags:

| Flag | Description |
| ---- | ----------- |
| `--implementer-model` | LLM model identifier for implementation (default: `claude-sonnet-4.5`) |
| `--image` | Container image used for tool execution (default: `shots-on-goal:latest`) |
| `--max-tools` | Max tool calls before aborting an attempt (default: 100) |
| `--validation` | Shell command(s) to verify success (can be repeated) |
| `--verbose`, `-v` | Enable verbose (DEBUG) logging |

> **Tip:** The orchestrator respects `OPENROUTER_API_KEY` / `OPENROUTER_KEY`. Export those before running.

## Running tests

Two suites cover different layers:

```bash
# Schema + helper coverage
uv run python -m unittest test_schema_v2

# (Optional) Legacy/unit tests if you still have the older suite around
uv run python -m unittest test_shots_on_goal
```

The schema suite spins up temporary git repositories and in-memory SQLite databases, so make sure a git executable and your container runtime are available.

## Troubleshooting tips

- **LLM failures**: Double-check your OpenRouter key and quota; re-run with `--verbose` for detailed logging.
- **Container errors**: Confirm the container runtime is available (`container --version` or `docker --version`) and that the `shots-on-goal` image exists.
- **Inspecting sessions**: Database files are created as `shots-on-goal-v2-<timestamp>.db` in the current directory. Use any SQLite browser to inspect attempts, tool calls, and validation results.

For deeper tracing, run with `--verbose` to elevate logging to DEBUG level.

---

Happy migrating! Contributions, issue reports, and ideas are welcome—this orchestrator is deliberately experimental and evolving quickly.
