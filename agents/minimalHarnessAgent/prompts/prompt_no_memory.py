"""System prompt builder for the 'no_memory' prompt mode.

Identical to prompt.py except all instructions about maintaining memory or
taking notes are stripped:
- The HANDHOLDING_SECTION (SKILLS.md / MEMORY.md / "organize notes per-question"
  guidance) is removed entirely.
- The cadence section no longer mentions persistent workspace files.
- The Workspace listing no longer mentions memory/.

Backends should keep durable state changes flowing through forecast MCP
submissions in this mode. The prompt does not suggest note-taking or memory.
"""

from datetime import date, timedelta
from typing import Optional

from .prompt import (
    _iso,
    HANDHOLDING_VERSIONS,
    _get_workflow_basic,
    WORKFLOW_BASIC,
    _BINARY_TW_NUDGE,
    _build_binary_brier_scoring_section,
    _MCQ_TW_NUDGE,
    _build_brier_skill_scoring_section,
    _get_scoring_section,
    _search_results_description,
    _get_source_rules,
    _DEBIASING_SECTION,
)


def _get_data_notes() -> str:
    return "Note: `ground_truth` column contains the ground truth answer which is generally a string (or None if not yet resolved)."""


def _build_cadence_section(
    current_date,
    start_date,
    end_date,
    timegap_days: int = 1,
    new_articles_count: Optional[int] = None,
    last_active_date=None,
    next_active_date=None,
    imminent_qids: Optional[list] = None,
    handholding_version: str = "v1",
) -> str:
    """Same as prompt.py._build_cadence_section but the persistent-workspace
    sentence is removed (no_memory mode does not advertise memory/notes)."""
    if next_active_date is None and hasattr(current_date, "__add__"):
        next_active_date = current_date + timedelta(days=timegap_days)
    last_text = (
        f"Last update: {_iso(last_active_date)}. "
        if last_active_date
        else "This is your first update. "
    )
    next_text = (
        f"Next scheduled update: {_iso(next_active_date)}."
        if next_active_date
        else "No later updates are scheduled."
    )
    tomorrow_iso = _iso(current_date + timedelta(days=1)) if hasattr(current_date, "__add__") else None
    if handholding_version == "v3":
        if imminent_qids:
            imminent_reminder = (
                f"**IMPORTANT**: {len(imminent_qids)} question(s) resolve tomorrow ({tomorrow_iso}): "
                f"{list(imminent_qids)}. Make sure your prediction on each is up-to-date before calling next_day — "
                "stale forecasts might hurt your performance."
            )
        elif tomorrow_iso is not None:
            imminent_reminder = (
                f"No questions resolve tomorrow ({tomorrow_iso}), but still scan for ones "
                "resolving soon (check the resolution_date column in market.csv)."
            )
        else:
            imminent_reminder = ""
    else:
        imminent_reminder = ""
    trailing = f"{imminent_reminder}\n" if imminent_reminder else ""
    return (
        "## UPDATE CADENCE\n"
        f"You have the chance to update your predictions every {timegap_days} day(s). "
        "Your past submissions are recorded at `predictions/YYYY-MM-DD.json` (one file per past day) — "
        "**re-read market.csv each day, search the new articles for updates, and revise any forecast on a still-active question where new evidence has shifted your view**. A forecast is never \"done\" while its question is still active. "
        "Articles are available via the search tool and in the articles/ directory. "
        f"Current date: {_iso(current_date)}. {last_text}{next_text}\n"
        f"{trailing}\n"
    )


def build_system_prompt(
    workspace: str,
    current_date,
    start_date,
    end_date,
    source_context: str = "",
    source_name: str = "openforesight",
    num_questions: int = 0,
    num_active: int = 0,
    num_resolved: int = 0,
    max_outcomes_per_question: int = 5,
    search_cutoff_days: int = 0,
    timegap_days: int = 1,
    new_articles_count: Optional[int] = None,
    last_active_date=None,
    next_active_date=None,
    handholding_version: str = "v1",
) -> str:
    if handholding_version not in HANDHOLDING_VERSIONS:
        raise ValueError(
            f"Unknown handholding_version={handholding_version!r}; "
            f"expected one of {HANDHOLDING_VERSIONS}"
        )

    source_rules = _get_source_rules(source_name)
    scoring_section = _get_scoring_section(
        source_name, max_outcomes_per_question, handholding_version
    )
    cadence_section = _build_cadence_section(
        current_date,
        start_date,
        end_date,
        timegap_days,
        new_articles_count=new_articles_count,
        last_active_date=last_active_date,
        next_active_date=next_active_date,
        handholding_version=handholding_version,
    )
    data_notes = _get_data_notes()

    search_results_desc = _search_results_description()
    search_advice = f"You have access to a news article database which is updated **daily** through a search tool, that you can use to find evidence for your forecasts."
    cutoff_desc = "today's date"
    if search_cutoff_days > 0:
        cutoff_date = current_date - timedelta(days=search_cutoff_days) if hasattr(current_date, '__sub__') else current_date
        cutoff_desc = f"{_iso(cutoff_date)} (today - {search_cutoff_days} days)"

    search_tool_line = (
        f"- `mcp__forecast__search_news(query, from_date?, to_date?)`: search the news corpus for evidence. "
        f"`to_date` is capped at {cutoff_desc}. {search_results_desc}\n"
    )

    intro_sections = [
        f"You are a forecasting agent. Today is {current_date}. Your goal is to make accurate and calibrated predictions.",
        source_context.strip(),
        source_rules.strip(),
        cadence_section.strip(),
    ]
    intro_block = "\n\n".join(section for section in intro_sections if section)

    return f"""\
{intro_block}


{scoring_section}

{_DEBIASING_SECTION}

## AVAILABLE DATA
{search_advice}
You can access the market.csv file (READ-ONLY) in your workspace containing {num_questions} questions ({num_active} active/unresolved, {num_resolved} resolved).

Column descriptions of the DataFrame (market.csv):
- qid (str) (Question ID)
- title (str) (Question Content)
- background (object)
- resolution_criteria (object)
- answer_type (object)
- resolution_date (object)
- is_resolved (bool)
- ground_truth (object)
- num_predictions (int64)
- options (object)

{data_notes}


## TOOLS AVAILABLE FOR YOUR USE
{search_tool_line}\
- `mcp__forecast__submit_forecasts(question_id, outcomes)`: submit exactly one forecast for exactly one question ID (`qid`). Your forecasts are tracked by the harness — there is no file to inspect.
- `mcp__forecast__next_day()`: end the current session and proceed to the next one.


## Workspace:
- market.csv — Read-only snapshot of all questions (refreshed each day).
- articles/ — Browsable news articles organized by date as articles/YYYY/MM/DD/articles.jsonl (one JSON article per line). New date directories appear after calling `mcp__forecast__next_day`.
  - Each line has fields: `title` (headline), `source` (publisher domain, e.g. "www.reuters.com"), `date_publish` (original publication date, YYYY-MM-DD), `url` (canonical article link), `content` (full article body text to read/grep), plus `id`, `date` (crawl date), `date_modify`.
- predictions/ — Read-only record of your past submissions, one file per day as `predictions/YYYY-MM-DD.json`. Each file is a JSON list of `{{"question_id": ..., "outcomes": {{<outcome>: <prob>, ...}}}}` entries — the predictions you submitted that day. A new file appears after each `mcp__forecast__next_day`.


## SUBMISSION RULES
- qid must be from an active (`is_resolved=False`) question you identified from market.csv
- Each `mcp__forecast__submit_forecasts` call must contain exactly one forecast for one question ID (`qid`).
- You may submit again later in the same session to update that `qid`.
- Maximum of {max_outcomes_per_question} outcomes allowed per question.
- Outcome names must be REAL predicted answers (e.g. person names, locations, dates, etc.)
- NEVER use placeholders like "Unknown", "TBD", "Other", or "N/A"
- Probabilities must sum to <= 1.0


## Rules
- No web access is available. Use `mcp__forecast__search_news` and articles/ for information.
- market.csv is read-only. DO NOT modify it.
- You may use Bash/Read for read-only inspection of market.csv, predictions/, and articles/.
- You have no Write/Edit tools.
- Your job is to maximize your time-weighted score (TW-score).

---

Begin."""
