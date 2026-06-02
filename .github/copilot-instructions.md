# Futuresim — Copilot Instructions

## What This Repo Is

Multi-agent LLM forecasting simulator. Agents are given forecast questions (from HuggingFace or custom datasets), make probability predictions over time, and are scored with log/Brier/peer scoring. The environment steps through dates, exposes questions, and resolves them; agents decide how to search, remember, and forecast.

## Dev Environment

Requires Python 3.12 (exactly — `>=3.12,<3.13`). Managed with `uv`.

```bash
uv sync
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
cp .env.example .env        # then fill in OPENROUTER_API_KEY
```

## Key Commands

```bash
# Run a simulation
python scripts/run_forecast_sim.py --config configs/shared/default_sim.yaml

# Run without search (no LanceDB needed)
python scripts/run_forecast_sim.py --config configs/shared/default_nosearch_sim.yaml

# Resume from last checkpoint
python scripts/run_forecast_sim.py --resume /path/to/output_dir

# Restart from a specific day (preserves predictions before that day)
python scripts/run_forecast_sim.py --restart_from /path/to/run --restart_from_day 2025-04-05

# Validate a module compiles (quick sanity check, no test suite)
python -m py_compile agents/minimalHarnessAgent/*.py
python -m py_compile agents/basicAgent/*.py
```

There is no test suite. Syntax checks via `python -m py_compile` are the standard validation step mentioned in the docs.

## Architecture

### Simulation Loop

`SimulationEnvironment` (in `environment/env.py`) drives the simulation: it loads questions, steps dates, calls each agent's `act()` in parallel (via `ThreadPoolExecutor`), collects `PredictionSubmission` objects, scores them, and writes output.

Agents receive the current date and two interfaces:
- **`ForecastInterface`**: list questions, submit predictions.
- **`DocInterface`**: read context documents.

Predictions are `{outcome_str: probability}` dicts. Agents can submit multiple outcomes per question, capped at `max_outcomes_per_question` (default 5).

### Agent Scaffolds

All agents extend `BaseAgent` (`agents/base.py`) and implement `act()`. Scaffold is selected per config under `defaults.scaffold`.

| Scaffold | Class | Description |
|----------|-------|-------------|
| `basic` / `allQ` / `allqd` | `BasicAgent` | Chat-tools loop; tools are `query`, `search`, `submit`, `next` |
| `qwenbasic` / `qwenallq` | Thin wrappers | Qwen compat — does **not** replay hidden thinking in history |
| `minimalHarness` | `MinimalHarnessAgent` | Runs external CLI (Codex, Claude Code, OpenCode) via MCP |

`BasicAgent` uses multiple inheritance: `BasicChatProtocol`, `BasicPromptBuilder`, `BasicMemoryPromptBuilder`, `BasicActionHandlers`, `BaseAgent`. Each mixin owns one concern — don't mix responsibilities when extending.

### MinimalHarnessAgent Module Layout

Each file owns a specific concern; add behavior to the smallest fitting module:

- `agent.py` — shared lifecycle, workspace setup, signal waiting
- `state.py` — `state.json`, market CSV, article staging
- `mcp_server.py` — FastMCP tools exposed to the CLI
- `mcp_helpers.py` — MCP command lines, relay, tool filtering
- `sandbox.py` — `bwrap` filesystem + network isolation
- `egress_proxy.py` — allowlist proxy for `network_isolation=True`
- `config.py` — `MinimalHarnessConfig`
- `prompts/` — prompt builders per `prompt_mode`
- Backend files (`claude_code_agent.py`, `codex_agent.py`, `opencode_agent.py`) implement hook surface only; shared simulation logic stays in `agent.py`

Backend hooks: `_prepare_harness_launch`, `_start_harness`, `_resume_harness`, `_respawns_each_day`, `_next_day_returns_immediately`, `_after_next_day_signal`, `_add_sandbox_harness_install_binds`, `_sandbox_harness_home_subdirs`.

### Inference Providers

Located in `inference/`. Providers expose `chat()` and `chat_json()`. `OpenRouter` uses a singleton `GlobalRateLimiter` (token-bucket). The `deepseek` provider is separate from OpenRouter.

### Search Tools

`BaseSearchTool` (`agents/search_tools/base.py`) defines the contract: `search(query, max_results, max_date, search_type, min_date) -> List[SearchResult]` and `get_article(id) -> Optional[Article]`. Implementations: `LanceDB` (local corpus), `GoogleSearchTool` (Serper.dev). Add a new backend by subclassing `BaseSearchTool` and wiring it into the runner.

### Environment / Scoring

Scoring is in `environment/scoring/`. Log score, Brier score, and peer scoring are separate modules. `ansmatching.py` handles LLM-based answer equivalence matching, with optional shared cache (`matcher_cache.py`).

## Configuration

YAML configs live in `configs/shared/`. Key top-level fields:

```yaml
sim_name:          # Output directory name
start_date / end_date:
resolution_start / resolution_end:
split:             # HF dataset split, e.g. 'aljazeera2026Q1'
dataset:           # 'openforesight' or 'custom'
search_tool_type:  # 'lancedb' or 'google'
matching:          # 'deepseek' or 'openrouter'
defaults:
  scaffold:        # 'basic', 'allQ', 'minimalHarness', etc.
  provider:
  max_actions:
  temperature:
  max_tokens:
agents:            # list of agent overrides
```

`${FSIM_*}` placeholders in configs are expanded by `pathing.py`, which reads `.env` at startup. Shell exports override `.env` values.

## Key Conventions

- **`pathing.py` is the only place** that loads `.env` and expands config placeholders. Don't call `dotenv` directly elsewhere.
- **Scaffold names don't auto-select from model names** — always set `defaults.scaffold` explicitly in config.
- **Qwen scaffolds** intentionally omit historical hidden thinking from history replay.
- **Token budgets** (`max_total_tokens`, `force_submit_threshold_tokens`, `submit_reserve_tokens`) track context headroom, not cumulative spend. Enforce `force_submit_threshold_tokens >= submit_reserve_tokens`.
- **Matcher cache**: `split: test` runs auto-use shared cache if `FSIM_SIM_MATCHER_CACHE_DIR` is set; non-test runs need explicit `matcher_cache: {enabled: true}` in YAML.
- **Custom question sets** need only `qid`, `title`, `resolution_date`, `ground_truth_answer` columns (plus accepted aliases — see README).
- **Custom search corpora** need a LanceDB table named `articles` with `chunk_id`, `content`, `date`, `vector`, and a few metadata fields.
- Output per run: `config.json`, `actions.jsonl`, `daily_metrics.csv`, `test_daily_metrics.csv`, `agents/<agent_id>/model_raw_*.jsonl`.
