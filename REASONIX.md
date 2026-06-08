# REASONIX.md — Futuresim working knowledge

## Stack
- Language: Python 3.12 (strict — `>=3.12,<3.13`)
- Package manager: `uv`
- LLM backends: OpenRouter (openai-compat), DeepSeek (native), Anthropic
- Vector search: LanceDB 0.29.1
- Config: YAML via PyYAML + Pydantic models
- Data: HuggingFace `datasets`, pandas, pyarrow

## Layout
- `agents/` — Agent scaffolds (BasicAgent, AllQAgent, MinimalHarnessAgent), search tools, utils
- `environment/` — Simulation loop, scoring, answer matching, data loading, datasets
- `inference/` — LLM provider backends (OpenRouter, DeepSeek)
- `configs/shared/` — YAML simulation configs
- `scripts/` — CLI entry points (run_forecast_sim, build_dashboard, scan_polymarket, etc.)
- `pathing.py` — Single `.env` loader + `${FSIM_*}` config placeholder expander
- `logs/` — Run output (per-sim timestamped dirs). Not checked in.

## Commands
```bash
# Setup
uv sync
source .venv/bin/activate

# Run simulation (primary entry point)
python scripts/run_forecast_sim.py --config configs/shared/default_sim.yaml

# No-search variant (no LanceDB needed)
python scripts/run_forecast_sim.py --config configs/shared/default_nosearch_sim.yaml

# Resume / restart
python scripts/run_forecast_sim.py --resume /path/to/output_dir
python scripts/run_forecast_sim.py --restart_from /path/to/run --restart_from_day 2025-04-05

# Syntax validation (no test suite exists)
python -m py_compile agents/basicAgent/*.py
```

## Conventions
- `from __future__ import annotations` — used in ~20 files, standard throughout
- `pathing.py` is the sole `.env` loader; never call `dotenv` directly elsewhere
- Config placeholders use `${FSIM_VAR}` syntax, expanded by `pathing.py`
- Scaffold names are explicit in config (`defaults.scaffold`); model name does not auto-select scaffold
- Agent mixins (BasicAgent): protocol, prompts, memory, actions each in own file via multiple inheritance
- Search tools implement `BaseSearchTool` contract from `agents/search_tools/base.py`
- No test suite — validate with `python -m py_compile`
- Run output structure: `{output_base}/{sim_name}/{timestamp}/` with `config.json`, `actions.jsonl`, `daily_metrics.csv`, `agents/` subdir

## Watch out for
- Python exactly 3.12 required; 3.13+ will not work
- `logs/current_sim/` contains real run data — review before deleting
- `artifacts/` is for downloaded corpora (LanceDB index, news JSONL); not source code
- Config `${FSIM_*}` placeholders fail hard at startup if unresolved — check `.env` or export vars
- Qwen scaffolds intentionally omit hidden thinking from history; only final assistant content + tool calls are fed back
