#!/usr/bin/env python
"""
Polymarket Opportunity Scanner
================================
Batch-scans Polymarket markets, runs lightweight futuresim on each, and
compares AI agent probabilities against current market odds to identify
potentially mispriced markets (edges).

Usage:
    python scripts/scan_polymarket.py --tag crypto --min-volume 500 --limit 10 --parallel
    python scripts/scan_polymarket.py --tag politics --limit 5 --dashboard-top 3
    python scripts/scan_polymarket.py --tag ai --min-volume 1000 --provider deepseek --deepseek-model deepseek-v4-pro
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    sys.exit("requests module required. Install: pip install requests")

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathing import load_repo_env, REPO_ROOT as _REPO_ROOT

load_repo_env(_REPO_ROOT)

# ===========================================================================
# Constants
# ===========================================================================

GAMMA_API = "https://gamma-api.polymarket.com"
RUN_FORECAST_SCRIPT = _REPO_ROOT / "scripts" / "run_forecast_sim.py"
BUILD_DASHBOARD_SCRIPT = _REPO_ROOT / "scripts" / "build_dashboard.py"
OUTPUT_BASE = _REPO_ROOT / "logs" / "current_sim"
SCAN_LOG_DIR = _REPO_ROOT / "logs" / "scans"
SCANNED_CACHE = SCAN_LOG_DIR / "scanned_cache.json"
SCANNED_CACHE_TTL_HOURS = 24

# Polymarket Gamma API uses tag slugs (not numeric IDs) for filtering.
# Slugs that don't match any Polymarket tag trigger a keyword search fallback.
KNOWN_TAG_SLUGS = {
    "politics", "crypto", "sports", "science", "technology",
    "entertainment", "economics", "world", "governance",
    "culture", "space", "health", "gaming", "defi",
}

# How many days /before/ the market resolution date the simulation starts
SIM_WINDOW_DAYS = 3

# Per-market simulation timeout (seconds)
SIM_TIMEOUT = 2000

# Default concurrency
DEFAULT_MAX_WORKERS = 5

# Scanned-market dedup cache
SCANNED_CACHE = SCAN_LOG_DIR / "scanned_cache.json"
SCANNED_CACHE_TTL_HOURS = 24


# ===========================================================================
# Scanned-market cache helpers
# ===========================================================================

def _load_scanned_cache() -> Dict[str, str]:
    """Return {slug: iso_timestamp} for previously scanned markets."""
    if not SCANNED_CACHE.exists():
        return {}
    try:
        with open(SCANNED_CACHE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_scanned_cache(cache: Dict[str, str]) -> None:
    SCAN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(SCANNED_CACHE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass


def _prune_stale_entries(cache: Dict[str, str], ttl_hours: int = SCANNED_CACHE_TTL_HOURS) -> Dict[str, str]:
    """Remove entries older than *ttl_hours*."""
    cutoff = datetime.now() - timedelta(hours=ttl_hours)
    return {
        slug: ts for slug, ts in cache.items()
        if datetime.fromisoformat(ts) > cutoff
    }


def _filter_new_markets(markets: List[dict], cache: Dict[str, str]) -> List[dict]:
    """Return only markets whose slug is not in the cache."""
    new = [m for m in markets if m.get("slug") not in cache]
    skipped = len(markets) - len(new)
    if skipped:
        print(f"[Scanner] Skipping {skipped} already-scanned market(s).", flush=True)
    return new


# ===========================================================================
# Polymarket API helpers
# ===========================================================================

def fetch_markets(
    tag: Optional[str] = None,
    min_volume: float = 500,
    max_volume: Optional[float] = None,
    resolves_after: Optional[date] = None,
    resolves_before: Optional[date] = None,
    limit: int = 20,
    sort: str = "volume",
) -> List[dict]:
    """Fetch open markets from the Polymarket Gamma API with optional filtering.

    Returns a list of market dicts, sorted by the given field descending.
    Valid sort values: volume (trending), liquidity, created_at (newest).
    """
    sort_map = {"volume": "volume", "liquidity": "liquidity", "newest": "createdAt"}
    api_sort = sort_map.get(sort, "volume")

    params: dict = {
        "closed": "false",
        "limit": min(limit * 4, 200),  # overfetch so client-side filters have enough
        "order": api_sort,
        "ascending": "false",
    }

    # Use tag slug for known Polymarket categories; unknown tags filter client-side
    search_keyword: Optional[str] = None
    if tag is not None:
        tag_lower = tag.lower().strip()
        if tag_lower in KNOWN_TAG_SLUGS:
            params["tag"] = tag_lower
        else:
            # Not a Polymarket tag — fetch broader set and filter client-side
            search_keyword = tag_lower
            print(f"[Scanner] '{tag}' is not a Polymarket tag; filtering client-side by keyword.", flush=True)

    url = f"{GAMMA_API}/markets"
    page_size = 100  # Gamma API page size
    pages_to_fetch = 2
    print(f"[Scanner] Fetching markets from Polymarket...", flush=True)
    if "tag" in params:
        print(f"[Scanner]   tag={params['tag']}  volume>={min_volume}  limit={limit}", flush=True)
    elif search_keyword:
        print(f"[Scanner]   keyword='{search_keyword}'  scanning up to {page_size * pages_to_fetch} markets...", flush=True)

    # Paginate through results
    raw: list = []
    for page in range(pages_to_fetch):
        params["limit"] = page_size
        params["offset"] = page * page_size
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"[Scanner] Error fetching page {page + 1}: {e}", flush=True)
            break
        except Exception as e:
            print(f"[Scanner] Unexpected error on page {page + 1}: {e}", flush=True)
            break
        if not isinstance(data, list) or not data:
            break
        raw.extend(data)

    if not raw:
        print(f"[Scanner] No markets returned from API.", flush=True)
        return []

    print(f"[Scanner] Fetched {len(raw)} markets across {page + 1} page(s).", flush=True)

    # Filter client-side
    markets: List[dict] = []
    for m in raw:
        if not isinstance(m, dict):
            continue

        # Keyword filter (client-side, for non-Polymarket-tag searches)
        if search_keyword:
            title = (m.get("question") or m.get("title") or "").lower()
            desc = (m.get("description") or "").lower()
            if search_keyword not in title and search_keyword not in desc:
                continue

        # Volume filter
        vol = float(m.get("volume", 0) or 0)
        if vol < min_volume:
            continue
        if max_volume is not None and vol > max_volume:
            continue

        # Date filter
        end_raw = m.get("endDateIso") or m.get("endDate") or ""
        if end_raw:
            try:
                end_date = date.fromisoformat(end_raw[:10])
            except (ValueError, TypeError):
                end_date = None
        else:
            end_date = None

        # Skip markets without a resolution date
        if end_date is None:
            continue

        # Skip if too early or too late
        if resolves_after and end_date < resolves_after:
            continue
        if resolves_before and end_date > resolves_before:
            continue

        # Attach parsed date for convenience
        m["_end_date"] = end_date
        m["_volume"] = vol

        markets.append(m)

        if len(markets) >= limit:
            break

    print(f"[Scanner] Found {len(markets)} markets matching filters.", flush=True)
    return markets


def parse_market_outcomes(market: dict) -> Tuple[List[str], List[float]]:
    """Parse outcomes and prices from a Polymarket market dict."""
    outcomes_raw = market.get("outcomes") or "[]"
    prices_raw = market.get("outcomePrices") or "[]"
    if isinstance(outcomes_raw, str):
        try:
            outcomes_raw = json.loads(outcomes_raw)
        except json.JSONDecodeError:
            outcomes_raw = []
    if isinstance(prices_raw, str):
        try:
            prices_raw = json.loads(prices_raw)
        except json.JSONDecodeError:
            prices_raw = []
    outcomes = list(outcomes_raw) if isinstance(outcomes_raw, list) else []
    prices = [float(p) for p in prices_raw] if isinstance(prices_raw, list) else []
    return outcomes, prices


# ===========================================================================
# Question JSONL builder (matches build_dashboard.py pattern)
# ===========================================================================

def build_custom_question_jsonl(market: dict, output_path: Path) -> dict:
    """Build a single-question JSONL file from Polymarket market data.

    Returns the question dict for reference.
    """
    import hashlib

    title = market.get("question", market.get("title", "Untitled"))
    slug = market.get("slug", "")
    end_date = market.get("_end_date")

    outcomes, prices = parse_market_outcomes(market)

    # Determine answer type
    if len(outcomes) == 2 and outcomes[0].lower() in ("yes", "no"):
        answer_type, options = "binary", ["Yes", "No"]
    elif len(outcomes) > 2:
        answer_type, options = "multichoice", list(outcomes)
    else:
        answer_type, options = "binary", ["Yes", "No"]

    desc = (market.get("description") or "")[:2000]

    # Unique qid from the slug (matches build_dashboard.py convention)
    qid = "PM" + hashlib.sha1(slug.encode()).hexdigest()[:6].upper()

    question = {
        "qid": qid,
        "title": title,
        "resolution_date": end_date.isoformat() if end_date else (date.today() + timedelta(days=30)).isoformat(),
        "ground_truth_answer": outcomes[0] if outcomes else "",
        "background": desc,
        "answer_type": answer_type,
        "options": json.dumps(options) if options else None,
        "resolution_criteria": market.get("resolutionSource", "") or "",
        "source": "polymarket",
        "source_split": "scan",
        "prompt": (
            f"Polymarket question. Current probability: {prices[0] if prices else 'N/A'}. "
            f"Volume: ${float(market.get('volume', 0)):,.0f}. "
            f"Research and submit your forecast."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(question, f, ensure_ascii=False)
        f.write("\n")

    return question


# ===========================================================================
# Futuresim runner (patterns from build_dashboard.py)
# ===========================================================================

def run_futuresim(
    dataset_path: Path,
    start_date: str,
    end_date: str,
    *,
    provider: str = "deepseek",
    model: str = "deepseek-v4-pro",
    sim_name: str = "scan",
    max_actions: int = 2,
    temperature: float = 0.0,
    timeout: int = SIM_TIMEOUT,
):
    """Run a lightweight futuresim on a single-question dataset.

    Returns the output directory path on success, or None on failure.
    """
    cmd = [
        sys.executable, str(RUN_FORECAST_SCRIPT),
        "--provider", provider, "--deepseek_model", model,
        "--matching", "exact",
        "--dataset", "custom", "--dataset_path", str(dataset_path),
        "--start_date", start_date, "--end_date", end_date,
        "--sim_name", sim_name,
        "--max_actions", str(max_actions),
        "--temperature", str(temperature),
        "--daily_submit",
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"    [Runner] Timed out after {timeout}s.", flush=True)
        return None

    if result.returncode != 0:
        stderr_tail = result.stderr.strip().splitlines()[-3:] if result.stderr else []
        tail_text = " | ".join(line.rstrip() for line in stderr_tail) if stderr_tail else "unknown error"
        print(f"    [Runner] Failed (exit {result.returncode}): {tail_text[-200:]}", flush=True)
        return None

    # Find the most recent output directory for this sim_name
    log_base = OUTPUT_BASE / sim_name
    if log_base.exists():
        dirs = sorted(
            log_base.iterdir(),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for d in dirs:
            if d.is_dir() and (d / "config.json").exists():
                return str(d)

    print(f"    [Runner] Could not find output directory under {log_base}.", flush=True)
    return None


# ===========================================================================
# Prediction extractor
# ===========================================================================

def extract_final_prediction(output_dir: str) -> Optional[Dict[str, float]]:
    """Extract the agent's final prediction from the simulation output.

    Reads actions.jsonl, collects all predictions, and returns the final
    outcome→probability mapping (the one from the latest sim_date).
    """
    actions_path = os.path.join(output_dir, "actions.jsonl")
    if not os.path.exists(actions_path):
        return None

    predictions_by_date: Dict[str, Dict[str, float]] = {}

    with open(actions_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            if entry.get("type") != "prediction":
                continue
            sim_date = entry.get("sim_date", "")
            outcomes = entry.get("outcomes", {})
            if outcomes and sim_date:
                predictions_by_date[sim_date] = {
                    str(k): float(v) for k, v in outcomes.items()
                }

    if not predictions_by_date:
        return None

    # Return the last prediction by sim_date
    last_date = max(predictions_by_date)
    return predictions_by_date[last_date]


def extract_agent_prob_yes(outcomes: Optional[Dict[str, float]]) -> Optional[float]:
    """Extract the probability for 'Yes' (binary) or highest-prob outcome."""
    if not outcomes:
        return None
    yes_prob = outcomes.get("Yes")
    if yes_prob is not None:
        return yes_prob
    yes_prob = outcomes.get("yes")
    if yes_prob is not None:
        return yes_prob
    # Multi-outcome: return max probability
    if outcomes:
        return max(outcomes.values())
    return None


def extract_evidence_from_sim(output_dir: str) -> dict:
    """Extract the agent's reasoning, counterfactuals, sentiment, and search
    evidence from the simulation's raw daily log.

    Returns a dict with keys: reasoning_bullets, counterfactual, sentiment_score,
    searches, evidence_diversity.
    """
    result = {
        "reasoning_bullets": [],
        "counterfactual": None,
        "sentiment_score": None,
        "searches": [],
        "evidence_diversity": None,
    }

    # Find the agent subdirectory
    agents_dir = os.path.join(output_dir, "agents")
    agent_dir = None
    if os.path.isdir(agents_dir):
        for d in os.listdir(agents_dir):
            p = os.path.join(agents_dir, d)
            if os.path.isdir(p) and d.startswith("basic_"):
                agent_dir = p
                break
    if not agent_dir:
        return result

    raw_log = os.path.join(agent_dir, "model_raw_daily.jsonl")
    if not os.path.exists(raw_log):
        return result

    day_text: Dict[str, List[str]] = {}
    day_searches: Dict[str, List[str]] = {}

    with open(raw_log, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            sim_date = entry.get("sim_date", "")
            meta = entry.get("metadata", {}) if isinstance(entry.get("metadata"), dict) else {}
            phase = meta.get("phase", "")
            resp = entry.get("response", "")

            # Extract reasoning text (text before tool calls or standalone)
            if resp and isinstance(resp, str) and len(resp) > 20:
                if not resp.startswith("TOOL_CALLS"):
                    day_text.setdefault(sim_date, []).append(resp.strip())

            # Extract tool calls with arguments
            if "TOOL_CALLS" in resp and "\n" in resp:
                try:
                    tc_data = json.loads(resp.split("\n", 1)[1]) if "\n" in resp else []
                    for tc in (tc_data if isinstance(tc_data, list) else []):
                        args_raw = tc.get("arguments_raw", "{}")
                        args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw

                        # Search queries
                        if tc.get("name") == "search_news":
                            q = args.get("query", "")
                            if q:
                                day_searches.setdefault(sim_date, []).append(q)

                        # Submission reasoning
                        if tc.get("name") == "submit_forecasts":
                            reasoning = args.get("reasoning", "")
                            if reasoning:
                                result["reasoning_bullets"].append(reasoning[:800])
                            cfact = args.get("counterfactual", "")
                            if cfact and not result["counterfactual"]:
                                result["counterfactual"] = cfact[:500]
                            div = args.get("evidence_diversity")
                            if div is not None and result["evidence_diversity"] is None:
                                try:
                                    result["evidence_diversity"] = int(div)
                                except (ValueError, TypeError):
                                    pass

                        # Market sentiment from submit
                        if tc.get("name") == "submit_forecasts":
                            sentiment = args.get("market_sentiment_score")
                            if sentiment is not None and result["sentiment_score"] is None:
                                try:
                                    result["sentiment_score"] = float(sentiment)
                                except (ValueError, TypeError):
                                    pass
                except Exception:
                    pass

    # Merge searches
    all_searches = []
    for d in sorted(day_searches):
        for s in day_searches[d]:
            if s not in all_searches:
                all_searches.append(s)
    result["searches"] = all_searches[:10]  # cap at 10 unique queries

    return result


# ===========================================================================
# Gap analysis (LLM-powered)
# ===========================================================================

def analyze_gap_with_llm(
    result: dict,
    evidence: dict,
    *,
    provider: str = "deepseek",
    model: str = "deepseek-v4-pro",
) -> Optional[str]:
    """Use the LLM to explain why the agent's forecast differs from (or aligns
    with) the Polymarket price.

    Constructs a focused prompt from the evidence extracted during the sim run
    and returns a concise natural-language analysis (3-5 sentences).
    """
    market_prob = result.get("market_prob")
    agent_prob = result.get("agent_prob")
    gap = result.get("gap")

    if market_prob is None or agent_prob is None:
        return None

    # Build the evidence summary
    reasoning_text = ""
    if evidence.get("reasoning_bullets"):
        reasoning_text = evidence["reasoning_bullets"][0][:600]

    counterfactual = evidence.get("counterfactual") or ""
    sentiment = evidence.get("sentiment_score")
    searches = evidence.get("searches", [])
    diversity = evidence.get("evidence_diversity")

    sentiment_label = "unknown"
    if sentiment is not None:
        if sentiment > 0.3:
            sentiment_label = "market appears OVERLY OPTIMISTIC"
        elif sentiment < -0.3:
            sentiment_label = "market appears OVERLY PESSIMISTIC"
        else:
            sentiment_label = "market appears BALANCED"

    gap_direction = "above" if agent_prob > market_prob else "below"
    gap_size = "large" if (gap or 0) > 0.15 else ("moderate" if (gap or 0) > 0.05 else "small")

    prompt = f"""Analyze the gap between an AI forecasting agent and Polymarket odds for a prediction market.

MARKET: {result['title'][:120]}
Polymarket probability: {market_prob:.1%}
AI agent probability: {agent_prob:.1%}
Gap: {gap:.1%} (agent is {gap_direction} market by {gap:.1%})
Sentiment assessment: {sentiment_label}
Evidence diversity: {diversity or 'unknown'} distinct sources
Search queries used: {', '.join(searches[:5]) if searches else 'none recorded'}

Agent's reasoning:
{reasoning_text[:600] if reasoning_text else '(no reasoning extracted)'}

Agent's counterfactual (scenario for opposite outcome):
{counterfactual[:300] if counterfactual else '(not provided)'}

In 2-4 sentences, explain WHY this gap exists (or doesn't). Consider:
- Does the agent have evidence the market is ignoring (or vice versa)?
- Is there emotional sentiment skewing the market price?
- Is this a genuine contrarian opportunity, or is the agent likely overconfident?
- If the gap is small (<5pp), is there true consensus or just lack of information?

Be specific and reference the actual evidence. Start directly with your analysis — no preamble."""

    try:
        from inference.deepseek import DeepSeekInference
        llm = DeepSeekInference(model)
        response = llm.complete(prompt, temperature=0.3, max_tokens=300)
        return response.strip()
    except Exception as e:
        # Fallback: build a rule-based summary from available evidence
        return _fallback_gap_summary(result, evidence, gap_size, gap_direction, sentiment_label)


def _fallback_gap_summary(
    result: dict,
    evidence: dict,
    gap_size: str,
    gap_direction: str,
    sentiment_label: str,
) -> str:
    """Rule-based gap summary when the LLM is unavailable."""
    gap = result.get("gap", 0) or 0
    parts = []

    if gap > 0.15:
        parts.append(
            f"Large {gap:.0%} gap: agent is {gap_direction} market. "
            f"This may indicate a contrarian opportunity if the agent's evidence is sound."
        )
    elif gap > 0.05:
        parts.append(
            f"Moderate {gap:.0%} gap: agent leans {gap_direction} consensus. "
            f"The agent's independent research may be picking up signals the crowd is discounting."
        )
    else:
        parts.append(
            f"Tight {gap:.0%} gap: agent broadly agrees with the market. "
            f"Either both see the same evidence or there is insufficient new information to deviate."
        )

    if evidence.get("counterfactual"):
        parts.append(f"Counterfactual considered: {evidence['counterfactual'][:150]}.")

    if sentiment_label:
        parts.append(f"Sentiment: {sentiment_label}.")

    div = evidence.get("evidence_diversity")
    if div is not None:
        if div < 2:
            parts.append("Evidence diversity was low; the agent may be under-informed.")
        else:
            parts.append(f"Evidence drawn from {div} distinct sources.")

    return " ".join(parts)


# ===========================================================================
# Core scan logic
# ===========================================================================

def scan_single_market(
    market: dict,
    *,
    provider: str = "deepseek",
    model: str = "deepseek-v4-pro",
    max_actions: int = 2,
    temperature: float = 0.0,
    timeout: int = SIM_TIMEOUT,
) -> dict:
    """Build question JSONL, run futuresim, extract prediction, and
    compare against Polymarket odds for a single market.

    Returns a result dict with fields:
        title, slug, market_prob, agent_prob, gap, volume, end_date,
        status, output_dir, error
    """
    title = market.get("question", market.get("title", "Unknown"))
    slug = market.get("slug", "unknown")
    market_id = market.get("id", "")
    end_date = market.get("_end_date")
    volume = market.get("_volume", float(market.get("volume", 0) or 0))

    # Parse market probabilities
    _outcomes, prices = parse_market_outcomes(market)
    market_prob = prices[0] if prices else None

    result = {
        "title": title,
        "slug": slug,
        "market_id": market_id,
        "market_prob": market_prob,
        "agent_prob": None,
        "gap": None,
        "volume": volume,
        "end_date": end_date.isoformat() if end_date else "",
        "status": "unknown",
        "output_dir": None,
        "error": None,
        "evidence": {},
        "gap_analysis": None,
    }

    # Validate market has needed data
    if not end_date:
        result["status"] = "skipped"
        result["error"] = "no end date"
        return result

    if market_prob is None:
        result["status"] = "skipped"
        result["error"] = "no market probability"
        return result

    # Build simulation window: SIM_WINDOW_DAYS before resolution
    sim_end = end_date
    sim_start = end_date - timedelta(days=SIM_WINDOW_DAYS)

    # Build question JSONL in a temp location
    question_dir = OUTPUT_BASE / "scan_inputs" / slug
    question_dir.mkdir(parents=True, exist_ok=True)
    question_path = question_dir / "question.jsonl"

    try:
        build_custom_question_jsonl(market, question_path)
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"failed to build question JSONL: {e}"
        return result

    # Run futuresim
    sim_name = f"scan_{slug[:40]}"
    print(f"  [{slug[:40]}] Running futuresim ({sim_start} → {sim_end})...", flush=True)

    output_dir = run_futuresim(
        dataset_path=question_path,
        start_date=sim_start.isoformat(),
        end_date=sim_end.isoformat(),
        sim_name=sim_name,
        provider=provider,
        model=model,
        max_actions=max_actions,
        temperature=temperature,
        timeout=timeout,
    )

    if not output_dir:
        result["status"] = "error"
        result["error"] = "simulation failed or timed out"
        return result

    result["output_dir"] = output_dir

    # Extract agent prediction
    outcomes = extract_final_prediction(output_dir)
    agent_prob = extract_agent_prob_yes(outcomes)

    if agent_prob is None:
        result["status"] = "no_prediction"
        result["error"] = "agent did not submit a forecast"
        return result

    result["agent_prob"] = agent_prob
    result["gap"] = abs(agent_prob - market_prob)
    result["status"] = "success"

    # Extract evidence from the simulation log
    result["evidence"] = extract_evidence_from_sim(output_dir)

    # Run gap analysis for non-trivial gaps (LLM or fallback)
    if result["gap"] > 0.03:
        result["gap_analysis"] = analyze_gap_with_llm(
            result, result["evidence"],
            provider=provider, model=model,
        )
    else:
        result["gap_analysis"] = _fallback_gap_summary(
            result, result["evidence"],
            gap_size="small",
            gap_direction="above" if agent_prob > market_prob else "below",
            sentiment_label="balanced",
        )

    print(f"    [{slug[:40]}] market={market_prob:.1%} agent={agent_prob:.1%} gap={result['gap']:.1%}",
          flush=True)

    return result


# ===========================================================================
# HTML report generator
# ===========================================================================

def build_html_report(results: List[dict], output_path: Path, metadata: dict = None) -> Path:
    """Generate a standalone HTML report from scan results.

    The report includes:
    - Summary stats cards (total, successful, edges, avg gap)
    - Sortable, searchable results table with color-coded gaps
    - Horizontal bar chart of top gaps (Chart.js from CDN)
    - Polymarket links for each market
    - Dark mode support

    Returns the path to the generated HTML file.
    """
    metadata = metadata or {}
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    # ── Compute summary stats ──────────────────────────────────────────
    total = len(results)
    success = [r for r in results if r["status"] == "success"]
    success_count = len(success)
    edge_count = sum(1 for r in success if (r["gap"] or 0) > 0.15)
    notable_count = sum(1 for r in success if 0.10 < (r["gap"] or 0) <= 0.15)
    error_count = sum(1 for r in results if r["status"] not in ("success", "no_prediction"))
    gaps = [r["gap"] for r in success if r["gap"] is not None]
    avg_gap = sum(gaps) / len(gaps) if gaps else 0.0
    max_gap = max(gaps) if gaps else 0.0

    tag_label = metadata.get("tag", "all")
    model_label = metadata.get("model", "unknown")

    # ── Build table rows ───────────────────────────────────────────────
    table_rows: List[str] = []
    detail_rows: List[str] = []
    for i, r in enumerate(results, 1):
        title = r["title"] or "Unknown"
        slug = r["slug"] or ""
        mkt_p = r["market_prob"]
        agt_p = r["agent_prob"]
        gap = r["gap"]
        vol = r.get("volume") or 0
        end_d = r.get("end_date", "")[:10]
        status = r["status"]
        analysis = r.get("gap_analysis") or ""
        evidence = r.get("evidence") or {}

        # Format values
        mkt_str = f"{mkt_p:.1%}" if mkt_p is not None else "—"
        agt_str = f"{agt_p:.1%}" if agt_p is not None else "—"
        gap_str = f"{gap:.1%}" if gap is not None else "—"
        vol_str = f"${vol:,.0f}" if vol else "—"

        # Row class and badge
        if status != "success":
            row_class = "row-error"
            badge = '<span class="badge badge-error">ERROR</span>'
        elif gap is None:
            row_class = "row-skipped"
            badge = '<span class="badge badge-skip">NO PRED</span>'
        elif gap > 0.15:
            row_class = "row-edge"
            badge = '<span class="badge badge-edge">EDGE</span>'
        elif gap > 0.10:
            row_class = "row-notable"
            badge = '<span class="badge badge-notable">NOTABLE</span>'
        else:
            row_class = "row-ok"
            badge = '<span class="badge badge-ok">inline</span>'

        # Gap bar visualization (inline mini-bar)
        gap_pct = (gap or 0) * 100
        bar_width = min(gap_pct * 3, 100)  # scale: 33% gap = full bar
        gap_bar = f'<div class="gap-bar"><div class="gap-fill" style="width:{bar_width}%"></div></div>' if gap is not None else ""
        has_analysis = bool(analysis and status == "success")

        # Use numeric id when available (redirects reliably); fall back to slug
        mid = r.get("market_id", "")
        polymarket_url = f"https://polymarket.com/market/{mid}" if mid else (f"https://polymarket.com/event/{slug}" if slug else "#")

        # Evidence summary line
        sentiment_val = evidence.get("sentiment_score")
        sentiment_str = f"{sentiment_val:+.2f}" if sentiment_val is not None else "—"
        searches = evidence.get("searches", [])
        cfact = (evidence.get("counterfactual") or "")[:200]
        diversity = evidence.get("evidence_diversity")

        table_rows.append(f"""\
            <tr class="{row_class}">
                <td class="toggle">{'<button class=\"expand-btn\" onclick=\"toggleDetail({i})\" title=\"Show analysis\">&#9654;</button>' if has_analysis else ''}</td>
                <td class="rank">{i}</td>
                <td class="title"><a href="{polymarket_url}" target="_blank" rel="noopener">{_esc(title[:80])}</a></td>
                <td class="num">{mkt_str}</td>
                <td class="num">{agt_str}</td>
                <td class="num gap-cell">{gap_str}{gap_bar}</td>
                <td class="num">{vol_str}</td>
                <td class="date">{end_d}</td>
                <td class="flag">{badge}</td>
            </tr>""")

        if has_analysis:
            detail_rows.append(f"""\
            <tr class="detail-row" id="detail-{i}" style="display:none">
                <td></td>
                <td colspan="8" class="detail-cell">
                    <div class="analysis-card">
                        <div class="analysis-header">
                            <strong>Gap Analysis</strong>
                            <span class="analysis-meta">Sentiment: {sentiment_str} &middot; Sources: {diversity or '?'} &middot; Searches: {len(searches)}</span>
                        </div>
                        <div class="analysis-body">{_esc(analysis)}</div>
                        {f'<div class="analysis-cfact"><strong>Counterfactual:</strong> {_esc(cfact)}</div>' if cfact else ''}
                        {f'<div class="analysis-searches"><strong>Searches:</strong> {_esc(", ".join(searches[:5]))}</div>' if searches else ''}
                    </div>
                </td>
            </tr>""")

    # ── Build chart data (top 20) ──────────────────────────────────────
    chart_markets = success[:20]
    chart_labels = json.dumps([(r["title"] or "")[:50] for r in chart_markets])
    chart_mkt = json.dumps([round(r["market_prob"] * 100, 1) if r["market_prob"] is not None else 0 for r in chart_markets])
    chart_agt = json.dumps([round(r["agent_prob"] * 100, 1) if r["agent_prob"] is not None else 0 for r in chart_markets])
    chart_gaps = json.dumps([round((r["gap"] or 0) * 100, 1) for r in chart_markets])

    # ── Assemble HTML ──────────────────────────────────────────────────
    html = f"""\
<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Polymarket Scan — {tag_label} — {timestamp}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root {{
    --bg: #0d1117;
    --bg-card: #161b22;
    --bg-hover: #1c2333;
    --border: #30363d;
    --text: #e6edf3;
    --text-dim: #8b949e;
    --green: #3fb950;
    --red: #f85149;
    --amber: #d2991d;
    --blue: #58a6ff;
    --edge-bg: rgba(248,81,73,0.12);
    --notable-bg: rgba(210,153,29,0.10);
    --ok-bg: rgba(63,185,80,0.06);
    --error-bg: rgba(139,148,158,0.08);
    --font: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:var(--font); background:var(--bg); color:var(--text); min-height:100vh; }}
.container {{ max-width:1400px; margin:0 auto; padding:24px; }}

/* Header */
.header {{ display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:16px; margin-bottom:24px; }}
.header h1 {{ font-size:1.6rem; font-weight:700; letter-spacing:-0.01em; }}
.header h1 span {{ color:var(--blue); }}
.header .meta {{ color:var(--text-dim); font-size:0.85rem; margin-top:4px; }}

/* Stats cards */
.stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin-bottom:24px; }}
.stat-card {{ background:var(--bg-card); border:1px solid var(--border); border-radius:8px; padding:16px; }}
.stat-card .label {{ font-size:0.75rem; text-transform:uppercase; letter-spacing:0.05em; color:var(--text-dim); margin-bottom:4px; }}
.stat-card .value {{ font-size:1.5rem; font-weight:700; }}
.stat-card .value.green {{ color:var(--green); }}
.stat-card .value.red {{ color:var(--red); }}
.stat-card .value.amber {{ color:var(--amber); }}

/* Chart */
.chart-container {{ background:var(--bg-card); border:1px solid var(--border); border-radius:8px; padding:20px; margin-bottom:24px; }}
.chart-container h2 {{ font-size:1rem; margin-bottom:16px; color:var(--text-dim); }}
.chart-wrap {{ position:relative; height:400px; max-height:50vh; }}

/* Search & filters */
.toolbar {{ display:flex; gap:12px; margin-bottom:16px; flex-wrap:wrap; align-items:center; }}
.toolbar input {{ flex:1; min-width:240px; padding:8px 12px; background:var(--bg-card); border:1px solid var(--border); border-radius:6px; color:var(--text); font-size:0.9rem; outline:none; }}
.toolbar input:focus {{ border-color:var(--blue); }}
.toolbar select {{ padding:8px 12px; background:var(--bg-card); border:1px solid var(--border); border-radius:6px; color:var(--text); font-size:0.85rem; cursor:pointer; }}
.toolbar .count {{ color:var(--text-dim); font-size:0.85rem; white-space:nowrap; }}

/* Table */
.table-wrap {{ overflow-x:auto; border:1px solid var(--border); border-radius:8px; max-height:70vh; overflow-y:auto; }}
.table-wrap thead {{ position:sticky; top:0; z-index:2; }}
.table-wrap thead th {{ position:sticky; top:0; background:var(--bg-card); }}
table {{ width:100%; border-collapse:collapse; font-size:0.88rem; }}
thead {{ position:sticky; top:0; z-index:1; }}
th {{ background:var(--bg-card); padding:10px 12px; text-align:left; font-weight:600; color:var(--text-dim); font-size:0.78rem; text-transform:uppercase; letter-spacing:0.04em; border-bottom:2px solid var(--border); cursor:pointer; user-select:none; white-space:nowrap; }}
th:hover {{ color:var(--text); }}
th.sorted {{ color:var(--blue); }}
th .arrow {{ margin-left:4px; font-size:0.7rem; }}
td {{ padding:10px 12px; border-bottom:1px solid var(--border); }}
tr:hover td {{ background:var(--bg-hover); }}
td.rank {{ color:var(--text-dim); font-weight:600; width:40px; }}
td.title a {{ color:var(--text); text-decoration:none; }}
td.title a:hover {{ color:var(--blue); text-decoration:underline; }}
td.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
td.date {{ color:var(--text-dim); white-space:nowrap; font-size:0.82rem; }}
td.gap-cell {{ min-width:120px; }}

/* Row styles */
.row-edge td {{ background:var(--edge-bg); }}
.row-notable td {{ background:var(--notable-bg); }}
.row-ok td {{ background:var(--ok-bg); }}

/* Badges */
.badge {{ display:inline-block; padding:2px 8px; border-radius:10px; font-size:0.72rem; font-weight:600; letter-spacing:0.03em; text-transform:uppercase; }}
.badge-edge {{ background:rgba(248,81,73,0.2); color:var(--red); }}
.badge-notable {{ background:rgba(210,153,29,0.2); color:var(--amber); }}
.badge-ok {{ background:rgba(63,185,80,0.15); color:var(--green); }}
.badge-error {{ background:rgba(139,148,158,0.15); color:var(--text-dim); }}
.badge-skip {{ background:rgba(139,148,158,0.12); color:var(--text-dim); }}

/* Gap mini-bar */
.gap-bar {{ display:inline-block; width:60px; height:5px; background:rgba(255,255,255,0.08); border-radius:3px; margin-left:6px; vertical-align:middle; }}
.gap-fill {{ height:100%; background:var(--red); border-radius:3px; min-width:2px; }}

/* Footer */
.footer {{ text-align:center; color:var(--text-dim); font-size:0.78rem; margin-top:24px; padding:16px; }}

/* Expand button */
.toggle {{ width:32px; text-align:center; }}
.expand-btn {{ background:none; border:none; color:var(--text-dim); cursor:pointer; font-size:0.7rem; padding:2px 6px; border-radius:4px; transition:all 0.15s; }}
.expand-btn:hover {{ color:var(--blue); background:rgba(88,166,255,0.12); }}
.expand-btn.open {{ transform:rotate(90deg); }}

/* Detail row */
.detail-row td {{ padding:0; border-bottom:2px solid var(--border); }}
.detail-cell {{ padding:12px 16px !important; background:var(--bg) !important; }}

/* Analysis card */
.analysis-card {{ display:grid; gap:8px; font-size:0.84rem; line-height:1.5; }}
.analysis-header {{ display:flex; justify-content:space-between; align-items:baseline; flex-wrap:wrap; gap:8px; }}
.analysis-header strong {{ color:var(--blue); }}
.analysis-meta {{ color:var(--text-dim); font-size:0.78rem; }}
.analysis-body {{ color:var(--text); padding:8px 0; border-top:1px solid var(--border); border-bottom:1px solid var(--border); }}
.analysis-cfact, .analysis-searches {{ color:var(--text-dim); font-size:0.8rem; }}

/* Responsive */
@media (max-width:768px) {{
    .container {{ padding:12px; }}
    .header h1 {{ font-size:1.2rem; }}
    .stats {{ grid-template-columns:repeat(2,1fr); }}
    th, td {{ padding:6px 8px; font-size:0.8rem; }}
    .gap-bar {{ display:none; }}
}}
</style>
</head>
<body>
<div class="container">

<!-- Header -->
<div class="header">
    <div>
        <h1>Polymarket <span>Opportunity Scanner</span></h1>
        <div class="meta">Tag: {tag_label} &middot; Model: {model_label} &middot; {timestamp}</div>
    </div>
</div>

<!-- Stats -->
<div class="stats">
    <div class="stat-card">
        <div class="label">Markets Scanned</div>
        <div class="value">{total}</div>
    </div>
    <div class="stat-card">
        <div class="label">Successful</div>
        <div class="value green">{success_count}</div>
    </div>
    <div class="stat-card">
        <div class="label">Potential Edges (>15pp)</div>
        <div class="value red">{edge_count}</div>
    </div>
    <div class="stat-card">
        <div class="label">Notable Gaps (10-15pp)</div>
        <div class="value amber">{notable_count}</div>
    </div>
    <div class="stat-card">
        <div class="label">Avg Gap</div>
        <div class="value">{avg_gap:.1%}</div>
    </div>
    <div class="stat-card">
        <div class="label">Max Gap</div>
        <div class="value red">{max_gap:.1%}</div>
    </div>
</div>

<!-- Chart -->
<div class="chart-container">
    <h2>Market vs Agent Probability (top {len(chart_markets)} by gap)</h2>
    <div class="chart-wrap"><canvas id="gapChart"></canvas></div>
</div>

<!-- Toolbar -->
<div class="toolbar">
    <input type="text" id="search" placeholder="Search markets..." oninput="filterTable()">
    <select id="statusFilter" onchange="filterTable()">
        <option value="all">All statuses</option>
        <option value="edge">Edges (>15pp)</option>
        <option value="notable">Notable (10-15pp)</option>
        <option value="ok">In line</option>
        <option value="error">Error / No pred</option>
    </select>
    <span class="count" id="rowCount">{total} rows</span>
</div>

<!-- Table -->
<div class="table-wrap">
<table>
<thead>
<tr>
    <th onclick="sortTable(0)" class="sorted"># <span class="arrow">▼</span></th>
    <th onclick="sortTable(1)">Market</th>
    <th onclick="sortTable(2)" class="num">Mkt %</th>
    <th onclick="sortTable(3)" class="num">Agent %</th>
    <th onclick="sortTable(4)" class="num sorted">Gap <span class="arrow">▼</span></th>
    <th onclick="sortTable(5)" class="num">Volume</th>
    <th onclick="sortTable(6)">End Date</th>
    <th>Flag</th>
</tr>
</thead>
<tbody id="tableBody">
{''.join(table_rows)}
{''.join(detail_rows)}
</tbody>
</table>
</div>

<div class="footer">
    Generated by futuresim scan_polymarket.py &middot; <a href="https://polymarket.com" style="color:var(--blue)">Polymarket</a>
</div>

</div>

<script>
// ── Chart ──────────────────────────────────────────────────────────────
const ctx = document.getElementById('gapChart').getContext('2d');
new Chart(ctx, {{
    type: 'bar',
    data: {{
        labels: {chart_labels},
        datasets: [
            {{
                label: 'Market %',
                data: {chart_mkt},
                backgroundColor: 'rgba(88,166,255,0.5)',
                borderColor: 'rgba(88,166,255,0.9)',
                borderWidth: 1,
            }},
            {{
                label: 'Agent %',
                data: {chart_agt},
                backgroundColor: 'rgba(248,81,73,0.5)',
                borderColor: 'rgba(248,81,73,0.9)',
                borderWidth: 1,
            }},
        ]
    }},
    options: {{
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
            legend: {{ labels: {{ color: '#8b949e', font: {{ size: 12 }} }} }},
            tooltip: {{ callbacks: {{ label: ctx => ctx.dataset.label + ': ' + ctx.raw + '%' }} }}
        }},
        scales: {{
            x: {{ max:100, ticks:{{ color:'#8b949e', callback:v=>v+'%' }}, grid:{{ color:'rgba(48,54,61,0.5)' }} }},
            y: {{ ticks:{{ color:'#8b949e', font:{{ size:11 }}, maxTicksLimit:20 }}, grid:{{ display:false }} }}
        }}
    }}
}});

// ── Expand / collapse detail rows ──────────────────────────────────────
function toggleDetail(idx) {{
    const detail = document.getElementById('detail-' + idx);
    const btn = document.querySelector(`tr:nth-of-type(${{idx}}) .expand-btn`);
    // nth-of-type is approximate; iterate to find the right button
    const allBtns = document.querySelectorAll('.expand-btn');
    let targetBtn = null;
    allBtns.forEach(b => {{
        const tr = b.closest('tr');
        const rankTd = tr?.querySelector('.rank');
        if (rankTd && rankTd.textContent.trim() === String(idx)) targetBtn = b;
    }});
    if (detail) {{
        const isOpen = detail.style.display !== 'none';
        detail.style.display = isOpen ? 'none' : 'table-row';
        if (targetBtn) {{
            targetBtn.classList.toggle('open', !isOpen);
            targetBtn.innerHTML = isOpen ? '&#9654;' : '&#9660;';
        }}
    }}
}}

// ── Search & Filter ────────────────────────────────────────────────────
function filterTable() {{
    const query = document.getElementById('search').value.toLowerCase();
    const status = document.getElementById('statusFilter').value;
    const rows = document.querySelectorAll('#tableBody tr');
    let visible = 0;
    rows.forEach(row => {{
        // Skip detail rows — they'll be hidden/shown with their parent
        if (row.classList.contains('detail-row')) return;

        const title = row.querySelector('.title')?.textContent.toLowerCase() || '';
        const rankEl = row.querySelector('.rank');
        const idx = rankEl ? rankEl.textContent.trim() : '';
        const isEdge = row.classList.contains('row-edge');
        const isNotable = row.classList.contains('row-notable');
        const isOk = row.classList.contains('row-ok');
        const isErr = row.classList.contains('row-error') || row.classList.contains('row-skipped');

        let statusMatch = status === 'all'
            || (status === 'edge' && isEdge)
            || (status === 'notable' && isNotable)
            || (status === 'ok' && isOk)
            || (status === 'error' && isErr);

        const textMatch = !query || title.includes(query);
        const show = statusMatch && textMatch;
        row.style.display = show ? '' : 'none';

        // Hide associated detail row
        const detail = document.getElementById('detail-' + idx);
        if (detail) detail.style.display = 'none';

        if (show) visible++;
    }});
    document.getElementById('rowCount').textContent = visible + ' / {total} rows';
}}

// ── Sort ───────────────────────────────────────────────────────────────
let sortCol = 4, sortAsc = false;

function sortTable(col) {{
    const tbody = document.getElementById('tableBody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    if (sortCol === col) sortAsc = !sortAsc; else {{ sortCol = col; sortAsc = col === 0; }}

    // Separate main rows and detail rows
    const mainRows = rows.filter(r => !r.classList.contains('detail-row'));
    const detailRows = rows.filter(r => r.classList.contains('detail-row'));

    mainRows.sort((a,b) => {{
        let va = a.children[col]?.textContent.trim().replace(/[$%,]/g,'') || '';
        let vb = b.children[col]?.textContent.trim().replace(/[$%,]/g,'') || '';
        let na = parseFloat(va), nb = parseFloat(vb);
        if (!isNaN(na) && !isNaN(nb)) return sortAsc ? na - nb : nb - na;
        return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
    }});

    // Update header arrows
    document.querySelectorAll('th').forEach((th,i) => {{
        th.classList.toggle('sorted', i === col);
        const arrow = th.querySelector('.arrow');
        if (arrow) arrow.textContent = i === col ? (sortAsc ? '▲' : '▼') : '';
    }});

    // Re-append main rows in sorted order
    mainRows.forEach(r => tbody.appendChild(r));

    // Re-append detail rows after their parent
    mainRows.forEach(mr => {{
        const rankEl = mr.querySelector('.rank');
        const idx = rankEl ? rankEl.textContent.trim() : '';
        const detail = document.getElementById('detail-' + idx);
        if (detail) tbody.appendChild(detail);
    }});
}}
</script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _esc(text: str) -> str:
    """HTML-escape a string."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


# ===========================================================================
# Output formatting
# ===========================================================================

def print_results_table(results: List[dict]):
    """Print a ranked table of scan results, sorted by gap descending."""
    print()
    print("=" * 110)
    print("POLYMARKET SCAN RESULTS")
    print("=" * 110)

    # Header
    header = f"{'#':>3}  {'Market':<42} {'Mkt%':>6} {'Agent%':>7} {'Gap':>7} {'Volume':>10} {'End':>12}  {'Flag'}"
    print(header)
    print("-" * 110)

    for i, r in enumerate(results, 1):
        title = r["title"][:40].ljust(42)
        mkt = f"{r['market_prob']:.1%}" if r["market_prob"] is not None else "N/A"
        agt = f"{r['agent_prob']:.1%}" if r["agent_prob"] is not None else "N/A"
        gap = f"{r['gap']:.1%}" if r["gap"] is not None else "N/A"
        vol = f"${r['volume']:,.0f}" if r.get("volume") else "-"
        end = r.get("end_date", "")[:10]

        gap_val = r["gap"] or 0
        if gap_val > 0.15:
            flag = "** POTENTIAL EDGE **"
        elif gap_val > 0.10:
            flag = "  notable gap     "
        elif r["status"] == "success":
            flag = "  in line         "
        elif r["status"] == "no_prediction":
            flag = "  NO PREDICTION   "
        elif r["status"] == "error":
            flag = "  ERROR           "
        else:
            flag = "  skipped         "

        print(f"{i:>3}  {title} {mkt:>6} {agt:>7} {gap:>7} {vol:>10} {end:>12}  {flag}")

    print("-" * 110)
    success_count = sum(1 for r in results if r["status"] == "success")
    edge_count = sum(1 for r in results if (r["gap"] or 0) > 0.15)
    print(f"  {success_count}/{len(results)} successful | {edge_count} potential edges (gap > 15pp)")
    print()


def save_results_csv(results: List[dict], output_path: Path):
    """Save full scan results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank", "title", "slug", "market_id", "market_prob", "agent_prob", "gap",
        "volume", "end_date", "status", "error", "output_dir",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, r in enumerate(results, 1):
            row = {k: r.get(k, "") for k in fieldnames}
            row["rank"] = i
            writer.writerow(row)
    print(f"Results saved to {output_path}", flush=True)


# ===========================================================================
# Dashboard trigger
# ===========================================================================

def run_dashboards(results: List[dict], top_n: int):
    """Launch build_dashboard.py for the top N markets by gap."""
    # Filter to successful markets with a gap
    candidates = [r for r in results if r["status"] == "success" and r["gap"] is not None]
    candidates.sort(key=lambda r: r["gap"] or 0, reverse=True)
    top = candidates[:top_n]

    if not top:
        print("[Dashboard] No successful markets to dashboard.", flush=True)
        return

    print(f"\n[Dashboard] Building dashboards for top {len(top)} markets...", flush=True)

    for i, r in enumerate(top, 1):
        slug = r["slug"]
        print(f"  [{i}/{len(top)}] {r['title'][:60]} (gap={r['gap']:.1%})", flush=True)
        cmd = [
            sys.executable, str(BUILD_DASHBOARD_SCRIPT),
            "--slug", slug,
        ]
        try:
            subprocess.run(
                cmd,
                cwd=str(_REPO_ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=600,  # 10 min per dashboard
            )
            print(f"    Dashboard complete.", flush=True)
        except subprocess.TimeoutExpired:
            print(f"    Dashboard timed out.", flush=True)
        except Exception as e:
            print(f"    Dashboard error: {e}", flush=True)

    print(f"[Dashboard] Done.", flush=True)


# ===========================================================================
# Main entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Polymarket Opportunity Scanner — find mispriced markets via AI forecasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python scripts/scan_polymarket.py --tag crypto --min-volume 500 --limit 10 --parallel
              python scripts/scan_polymarket.py --tag ai --limit 5 --dashboard-top 3
              python scripts/scan_polymarket.py --tag politics --min-volume 2000 --limit 20 --parallel
        """),
    )

    # Market discovery
    parser.add_argument("--tag", default=None,
                        help="Tag filter (e.g. politics, crypto, ai, sports, science)")
    parser.add_argument("--min-volume", type=float, default=10_000,
                        help="Minimum trading volume (USD). Default: 10000")
    parser.add_argument("--max-volume", type=float, default=None,
                        help="Maximum trading volume (USD). Default: none")
    parser.add_argument("--limit", type=int, default=20,
                        help="Max markets to scan. Default: 20")
    parser.add_argument("--sort", default="volume", choices=["volume", "liquidity", "newest"],
                        help="Sort markets by: volume (trending), liquidity, newest. Default: volume")
    parser.add_argument("--resolves-after", default=None,
                        help="Earliest resolution date (YYYY-MM-DD). Default: today")
    parser.add_argument("--resolves-before", default=None,
                        help="Latest resolution date (YYYY-MM-DD). Default: 3 months from today")

    # Simulation settings
    parser.add_argument("--provider", default="deepseek",
                        choices=["deepseek", "openrouter"],
                        help="Inference provider. Default: deepseek")
    parser.add_argument("--deepseek-model", default="deepseek-v4-pro",
                        help="DeepSeek model name. Default: deepseek-v4-pro")
    parser.add_argument("--openrouter-model", default=None,
                        help="OpenRouter model name (e.g. deepseek/deepseek-v3.2)")
    parser.add_argument("--max-actions", type=int, default=5,
                        help="Max actions per simulation day. Default: 5")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Model temperature. Default: 0.0")
    parser.add_argument("--timeout", type=int, default=SIM_TIMEOUT,
                        help=f"Per-market timeout in seconds. Default: {SIM_TIMEOUT}")

    # Concurrency
    parser.add_argument("--parallel", action="store_true", default=False,
                        help=f"Run up to {DEFAULT_MAX_WORKERS} markets concurrently")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
                        help=f"Max parallel workers. Default: {DEFAULT_MAX_WORKERS}")

    # Output
    parser.add_argument("--no-cache", action="store_true", default=False,
                        help="Ignore scanned-market cache and re-scan everything")
    parser.add_argument("--dashboard-top", type=int, default=None,
                        help="After scanning, run build_dashboard.py on top N markets by gap")
    parser.add_argument("--html-from-csv", default=None,
                        help="Regenerate HTML report from an existing CSV file (skips scanning)")

    args = parser.parse_args()

    # ── Regenerate HTML from CSV mode ──────────────────────────────────
    if args.html_from_csv:
        csv_path = Path(args.html_from_csv)
        if not csv_path.exists():
            print(f"Error: CSV not found: {csv_path}")
            sys.exit(1)
        results = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append({
                    "title": row.get("title", ""),
                    "slug": row.get("slug", ""),
                    "market_prob": float(row["market_prob"]) if row.get("market_prob") and row["market_prob"] != "" else None,
                    "agent_prob": float(row["agent_prob"]) if row.get("agent_prob") and row["agent_prob"] != "" else None,
                    "gap": float(row["gap"]) if row.get("gap") and row["gap"] != "" else None,
                    "volume": float(row["volume"]) if row.get("volume") and row["volume"] != "" else 0,
                    "end_date": row.get("end_date", ""),
                    "status": row.get("status", "unknown"),
                    "output_dir": row.get("output_dir", ""),
                    "error": row.get("error", ""),
                })
        results.sort(key=lambda r: (r["gap"] is not None and r["status"] == "success", r["gap"] or 0), reverse=True)
        html_path = csv_path.with_suffix(".html")
        build_html_report(results, html_path, metadata={"tag": "from-csv", "model": "N/A"})
        print(f"HTML report regenerated: {html_path}")
        return

    # Resolve date defaults
    today = date.today()
    resolves_after = date.fromisoformat(args.resolves_after) if args.resolves_after else today
    resolves_before = (
        date.fromisoformat(args.resolves_before)
        if args.resolves_before
        else today + timedelta(days=90)
    )

    print("=" * 70)
    print("POLYMARKET OPPORTUNITY SCANNER")
    print("=" * 70)
    print(f"  Tag:           {args.tag or 'any'}")
    print(f"  Min volume:    ${args.min_volume:,.0f}")
    print(f"  Max volume:    {'${:,.0f}'.format(args.max_volume) if args.max_volume else 'none'}")
    print(f"  Date range:    {resolves_after} → {resolves_before}")
    print(f"  Limit:         {args.limit} markets")
    print(f"  Provider:      {args.provider}")
    print(f"  Model:         {args.deepseek_model or args.openrouter_model}")
    print(f"  Max actions:   {args.max_actions}")
    print(f"  Temperature:   {args.temperature}")
    print(f"  Sort:          {args.sort}")
    print(f"  Parallel:      {args.parallel} (workers={args.max_workers})")
    print(f"  Dashboard top: {args.dashboard_top or 'off'}")
    print()

    # ── Step 1: Fetch markets ──────────────────────────────────────────
    markets = fetch_markets(
        tag=args.tag,
        min_volume=args.min_volume,
        max_volume=args.max_volume,
        resolves_after=resolves_after,
        resolves_before=resolves_before,
        limit=args.limit,
        sort=args.sort,
    )

    if not markets:
        print("[Scanner] No markets found matching filters. Exiting.", flush=True)
        return

    # ── Dedup: filter out already-scanned markets, re-fetch if needed ──
    if not args.no_cache:
        cache = _prune_stale_entries(_load_scanned_cache())
        markets = _filter_new_markets(markets, cache)
        if not markets:
            # All top-N cached — bump the limit to skip ahead
            bumped_limit = args.limit + len(cache)
            print(f"[Scanner] Top {args.limit} all cached, fetching up to {bumped_limit}...", flush=True)
            markets = fetch_markets(
                tag=args.tag,
                min_volume=args.min_volume,
                max_volume=args.max_volume,
                resolves_after=resolves_after,
                resolves_before=resolves_before,
                limit=bumped_limit,
                sort=args.sort,
            )
            markets = [m for m in markets if m.get("slug") not in cache][:args.limit]
            if not markets:
                print("[Scanner] All markets already scanned (use --no-cache to force). Exiting.", flush=True)
                return
            print(f"[Scanner] Found {len(markets)} new market(s) beyond cache.", flush=True)
    else:
        cache = {}
        print("[Scanner] Cache disabled (--no-cache).", flush=True)

    # ── Step 2: Run scans ──────────────────────────────────────────────
    model_name = args.openrouter_model if args.provider == "openrouter" else args.deepseek_model

    scan_specs = [
        {
            "market": m,
            "provider": args.provider,
            "model": model_name,
            "max_actions": args.max_actions,
            "temperature": args.temperature,
            "timeout": args.timeout,
        }
        for m in markets
    ]

    results: List[dict] = []

    if args.parallel and len(scan_specs) > 1:
        print(f"[Scanner] Running {len(scan_specs)} markets in parallel "
              f"(max {args.max_workers} at a time)...", flush=True)
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_map = {
                executor.submit(scan_single_market, **spec): spec
                for spec in scan_specs
            }
            for future in as_completed(future_map):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    spec = future_map[future]
                    slug = spec["market"].get("slug", "?")
                    results.append({
                        "title": spec["market"].get("question", "?"),
                        "slug": slug,
                        "market_id": spec["market"].get("id", ""),
                        "market_prob": None,
                        "agent_prob": None,
                        "gap": None,
                        "volume": float(spec["market"].get("volume", 0) or 0),
                        "end_date": "",
                        "status": "error",
                        "output_dir": None,
                        "error": str(e),
                        "evidence": {},
                        "gap_analysis": None,
                    })
    else:
        print(f"[Scanner] Running {len(scan_specs)} markets sequentially...", flush=True)
        for spec in scan_specs:
            try:
                result = scan_single_market(**spec)
                results.append(result)
            except Exception as e:
                slug = spec["market"].get("slug", "?")
                results.append({
                    "title": spec["market"].get("question", "?"),
                    "slug": slug,
                    "market_id": spec["market"].get("id", ""),
                    "market_prob": None,
                    "agent_prob": None,
                    "gap": None,
                    "volume": float(spec["market"].get("volume", 0) or 0),
                    "end_date": "",
                    "status": "error",
                    "output_dir": None,
                    "error": str(e),
                    "evidence": {},
                    "gap_analysis": None,
                })

    # Sort by gap descending (None/error at bottom)
    results.sort(key=lambda r: (r["gap"] is not None and r["status"] == "success", r["gap"] or 0), reverse=True)

    # ── Save scanned slugs to cache ────────────────────────────────────
    if not args.no_cache:
        now = datetime.now().isoformat()
        for r in results:
            slug = r.get("slug")
            if slug:
                cache[slug] = now
        _save_scanned_cache(cache)

    # ── Step 3: Output results ─────────────────────────────────────────
    print_results_table(results)

    timestamp = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    csv_path = SCAN_LOG_DIR / f"scan_results_{timestamp}.csv"
    save_results_csv(results, csv_path)

    # Generate HTML report
    html_path = SCAN_LOG_DIR / f"scan_results_{timestamp}.html"
    build_html_report(results, html_path, metadata={
        "tag": args.tag or "all",
        "model": model_name,
    })
    print(f"HTML report saved to {html_path}", flush=True)
    print(f"  Open with: start {html_path}", flush=True)

    # ── Step 4: Optional dashboards ────────────────────────────────────
    if args.dashboard_top:
        run_dashboards(results, args.dashboard_top)


if __name__ == "__main__":
    main()
