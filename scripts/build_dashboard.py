#!/usr/bin/env python
"""
Polymarket + Futuresim Dashboard Builder
=========================================
Runs futuresim on a Polymarket question, then generates a standalone HTML
dashboard with Chart.js comparing AI agent predictions to Polymarket odds.

Usage:
    python scripts/build_dashboard.py --slug new-rhianna-album-before-gta-vi-926
    python scripts/build_dashboard.py --search "Fed rate hike"
    python scripts/build_dashboard.py --slug some-market --record-snapshot
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import textwrap
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    sys.exit("requests module required. Install: pip install requests")

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathing import load_repo_env, REPO_ROOT as _REPO_ROOT

load_repo_env(_REPO_ROOT)

from inference.deepseek import DeepSeekInference


# ===================================================================
# Polymarket price history via Gamma API trend data
# ===================================================================
# NOTE: CLOB /prices-history always returns empty for all markets.
# Gamma API provides oneDayPriceChange, oneWeekPriceChange, oneMonthPriceChange
# which we use to back-project approximate daily prices. CSV snapshots
# (recorded each run) provide actual historical anchor points.

def fetch_pm_price_history(market: dict, sim_start: date, sim_end: date) -> Dict[str, float]:
    """Build approximate price history from Gamma API trend data.
    Uses lastTradePrice + price change deltas to back-project daily values.
    Falls back to CSV snapshots for real historical data."""
    result: Dict[str, float] = {}

    current = float(market.get("lastTradePrice", 0))
    if not current:
        return result

    today = date.today()

    # Gamma price change deltas (scalar, not percentage)
    delta_1d = float(market.get("oneDayPriceChange", 0) or 0)
    delta_1w = float(market.get("oneWeekPriceChange", 0) or 0)
    delta_1mo = float(market.get("oneMonthPriceChange", 0) or 0)

    # Walk backward from today, blending delta signals
    for d in sorted_dates_between(sim_start, min(sim_end, today)):
        days_ago = (today - d).days
        if days_ago <= 0:
            result[d.isoformat()] = current
        elif days_ago == 1:
            result[d.isoformat()] = max(0.0, min(1.0, current - delta_1d))
        elif days_ago <= 7:
            # Linear interpolate between 1d and 7d delta
            frac = (days_ago - 1) / 6.0
            delta = delta_1d + frac * (delta_1w - delta_1d)
            result[d.isoformat()] = max(0.0, min(1.0, current - delta))
        elif days_ago <= 30:
            frac = (days_ago - 7) / 23.0
            delta = delta_1w + frac * (delta_1mo - delta_1w)
            result[d.isoformat()] = max(0.0, min(1.0, current - delta))
        # Beyond 30 days: no signal, don't include

    return result


def sorted_dates_between(start: date, end: date):
    """Yield all dates from start to end inclusive, in order."""
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
GAMMA_API = "https://gamma-api.polymarket.com"
RUN_FORECAST_SCRIPT = REPO_ROOT / "scripts" / "run_forecast_sim.py"


# ===================================================================
# Polymarket API client
# ===================================================================

class PolymarketClient:
    """Thin wrapper around the Polymarket Gamma Markets API (no key required)."""

    @staticmethod
    def _parse_outcomes(market: dict) -> Tuple[List[str], List[float]]:
        outcomes_raw = market.get("outcomes") or "[]"
        prices_raw = market.get("outcomePrices") or "[]"
        if isinstance(outcomes_raw, str):
            outcomes_raw = json.loads(outcomes_raw)
        if isinstance(prices_raw, str):
            prices_raw = json.loads(prices_raw)
        return list(outcomes_raw), [float(p) for p in prices_raw]

    @staticmethod
    def lookup_by_slug(slug: str) -> Optional[dict]:
        url = f"{GAMMA_API}/markets"
        try:
            resp = requests.get(url, params={"limit": 1, "slug": slug}, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            return data[0] if data else None
        except requests.RequestException as e:
            print(f"[Polymarket] Lookup error: {e}")
            return None

    @staticmethod
    def search(keyword: str, limit: int = 10) -> List[dict]:
        """Search events by market_title via /events/similar, then fetch full
        details to extract the nested markets. Returns a flat list of market dicts."""
        # Step 1 — find matching events
        try:
            resp = requests.get(f"{GAMMA_API}/events/similar",
                                params={"market_title": keyword, "limit": min(limit, 20),
                                        "closed": "false"}, timeout=15)
            resp.raise_for_status()
            events = resp.json()
        except requests.RequestException as e:
            print(f"[Polymarket] Search error: {e}")
            return []

        if not events:
            return []

        # Step 2 — fetch each event's detail to get the markets
        markets: List[dict] = []
        for ev in events[:limit]:
            eid = ev.get("id")
            if not eid:
                continue
            try:
                er = requests.get(f"{GAMMA_API}/events/{eid}", timeout=15)
                if er.status_code != 200:
                    continue
                detail = er.json()
                for m in detail.get("markets") or []:
                    # Inherit event-level endDate if market doesn't have one
                    if not m.get("endDate"):
                        m["endDate"] = detail.get("endDate") or ev.get("endDate", "")
                    markets.append(m)
                    if len(markets) >= limit:
                        break
            except requests.RequestException:
                continue
            if len(markets) >= limit:
                break
        return markets

    @staticmethod
    def fetch_detail(market_id: str) -> Optional[dict]:
        url = f"{GAMMA_API}/markets/{market_id}"
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            print(f"[Polymarket] Detail error: {e}")
            return None

    @staticmethod
    def get_current_probability(market: dict) -> Optional[float]:
        outcomes, prices = PolymarketClient._parse_outcomes(market)
        if not prices:
            return None
        if outcomes and outcomes[0].lower() == "yes":
            return prices[0]
        return max(prices)

    @staticmethod
    def resolve_market(slug: Optional[str] = None,
                       keyword: Optional[str] = None) -> Optional[dict]:
        if slug:
            summary = PolymarketClient.lookup_by_slug(slug)
            if not summary:
                # Slug not found — auto-fallback to keyword search
                print(f"[Polymarket] Slug not found: {slug}. Trying keyword search...")
                keyword = slug.replace("-", " ")
                slug = None  # fall through to search below
            else:
                market_id = summary.get("id", "")
                detail = PolymarketClient.fetch_detail(market_id) if market_id else None
                market = detail or summary
                print(f"[Polymarket] Loaded: {market.get('question', slug)}")
                return market
        if keyword:
            results = PolymarketClient.search(keyword, limit=10)
            if not results:
                print(f"[Polymarket] No results for '{keyword}'.")
                return None
            summary = results[0]
            market_id = summary.get("id", "")
            detail = PolymarketClient.fetch_detail(market_id) if market_id else None
            market = detail or summary
            print(f"[Polymarket] Matched: {market.get('question', keyword)}")
            return market
        return None


# ===================================================================
# Custom question builder
# ===================================================================

def build_custom_question_jsonl(market: dict, output_path: Path,
                                 resolution_date_override: Optional[str] = None) -> dict:
    title = market.get("question", market.get("title", "Untitled"))
    end_date_raw = market.get("endDate", market.get("endDateIso", ""))
    if resolution_date_override:
        res_date = resolution_date_override
    elif end_date_raw:
        res_date = end_date_raw[:10]
    else:
        res_date = (date.today() + timedelta(days=30)).isoformat()

    outcomes, prices = PolymarketClient._parse_outcomes(market)
    if len(outcomes) == 2 and outcomes[0].lower() == "yes":
        answer_type, options = "binary", ["Yes", "No"]
    elif len(outcomes) > 2:
        answer_type, options = "multichoice", list(outcomes)
    else:
        answer_type, options = "binary", ["Yes", "No"]

    desc = (market.get("description") or "")[:2000]
    import hashlib
    qid = "PM" + hashlib.sha1(market.get("slug", "pm").encode()).hexdigest()[:6].upper()

    question = {
        "qid": qid,
        "title": title,
        "resolution_date": res_date,
        "ground_truth_answer": outcomes[0] if outcomes else "",
        "background": desc,
        "answer_type": answer_type,
        "options": json.dumps(options) if options else None,
        "resolution_criteria": market.get("resolutionSource", "") or "",
        "source": "polymarket",
        "source_split": "dashboard",
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
    print(f"[Builder] Question written: {output_path}")
    return question


# ===================================================================
# Futuresim runner
# ===================================================================

def run_futuresim(dataset_path: Path, start_date: str, end_date: str, *,
                  provider: str = "deepseek", model: str = "deepseek-v4-flash",
                  matching: str = "exact", sim_name: str = "dashboard",
                  max_actions: int = 3, temperature: float = 0.7,
                  force_submit: bool = False, timeout: int = 1800,
                  resolution_end: Optional[str] = None) -> Optional[str]:
    cmd = [
        sys.executable, str(RUN_FORECAST_SCRIPT),
        "--provider", provider, "--deepseek_model", model,
        "--matching", matching,
        "--dataset", "custom", "--dataset_path", str(dataset_path),
        "--start_date", start_date, "--end_date", end_date,
        "--sim_name", sim_name,
        "--max_actions", str(max_actions),
        "--temperature", str(temperature),
    ]
    if force_submit:
        cmd.extend(["--daily_submit"])
    if resolution_end:
        cmd.extend(["--resolution_end", resolution_end])
    print(f"\n[Runner] Launching...")
    print("-" * 60)
    try:
        result = subprocess.run(cmd, cwd=str(REPO_ROOT),
                                text=True, encoding="utf-8", timeout=timeout)
    except subprocess.TimeoutExpired:
        print("[Runner] Timed out.")
        return None
    print("-" * 60)
    if result.returncode != 0:
        print(f"[Runner] Failed (exit {result.returncode}).")
        return None
    # Parse output dir from captured stdout (we no longer capture, so search logs)
    log_base = REPO_ROOT / "logs" / "current_sim" / sim_name
    if log_base.exists():
        dirs = sorted(log_base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for d in dirs:
            if d.is_dir() and (d / "config.json").exists():
                print(f"[Runner] Output: {d}")
                return str(d)
    print("[Runner] Could not find output directory.")
    return None


# ===================================================================
# Prediction + reasoning extractor
# ===================================================================

def extract_daily_data(output_dir: str) -> List[dict]:
    """
    Extract per-day prediction + reasoning from the simulation output.
    Returns a list of {date, prob_yes, reasoning} dicts.
    """
    actions_path = os.path.join(output_dir, "actions.jsonl")
    raw_log_path = os.path.join(output_dir, "agents")
    # Find the agent subdirectory
    agent_dir = None
    if os.path.isdir(raw_log_path):
        for d in os.listdir(raw_log_path):
            p = os.path.join(raw_log_path, d)
            if os.path.isdir(p) and d.startswith("basic_"):
                agent_dir = p
                break

    # Collect predictions from actions.jsonl
    predictions: Dict[str, dict] = {}  # date -> {prob, ...}
    if os.path.exists(actions_path):
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
                yes_p = outcomes.get("Yes", outcomes.get("yes"))
                if yes_p is None and outcomes:
                    yes_p = max(outcomes.values())
                if sim_date and yes_p is not None:
                    predictions[sim_date] = {"prob_yes": float(yes_p),
                                              "outcomes": outcomes}

    # Extract reasoning and sentiment from raw daily log
    if agent_dir:
        raw_log = os.path.join(agent_dir, "model_raw_daily.jsonl")
        if os.path.exists(raw_log):
            _extract_reasoning(raw_log, predictions)
            _extract_sentiment_scores(raw_log, predictions)

    # Build daily list
    result = []
    for d in sorted(predictions):
        entry = predictions[d]
        result.append({
            "date": d,
            "prob_yes": entry.get("prob_yes", 0),
            "reasoning": entry.get("reasoning", ""),
            "searches": entry.get("searches", []),
            "sentiment_score": entry.get("sentiment_score"),
        })
    return result


def _extract_reasoning(raw_log_path: str, predictions: Dict[str, dict]):
    """Enrich predictions dict with reasoning from the model_raw_daily log."""
    # Track per-day context: all text analysis + searches + queries
    day_text: Dict[str, List[str]] = {}
    day_searches: Dict[str, List[str]] = {}

    with open(raw_log_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            sim_date = d.get("sim_date", "")
            meta = d.get("metadata", {})
            phase = meta.get("phase", "")
            resp = d.get("response", "")

            # Extract reasoning text: lines before TOOL_CALLS or plain text responses
            if resp and not resp.startswith("TOOL_CALLS"):
                text = resp.strip()
                # Filter out empty/whitespace-only
                if text and len(text) > 10:
                    day_text.setdefault(sim_date, []).append(text)

            # Also grab text from TOOL_CALLS blocks that contain analysis
            if resp.startswith("TOOL_CALLS"):
                # Check for text before the TOOL_CALLS marker
                parts = resp.split("TOOL_CALLS:", 1)
                if len(parts) > 1 and parts[0].strip():
                    text = parts[0].strip()
                    if len(text) > 10:
                        day_text.setdefault(sim_date, []).append(text)

            # Collect search queries as evidence
            if phase == "search" and "TOOL_CALLS" in resp:
                try:
                    tc_data = json.loads(resp.split("\n", 1)[1]) if "\n" in resp else []
                    for tc in (tc_data if isinstance(tc_data, list) else []):
                        args_raw = tc.get("arguments_raw", "{}")
                        args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                        q = args.get("query", "")
                        if q:
                            day_searches.setdefault(sim_date, []).append(q)
                except Exception:
                    pass

            # Capture query_df analysis
            if phase == "query":
                try:
                    tc_data = json.loads(resp.split("\n", 1)[1]) if "\n" in resp else []
                    for tc in (tc_data if isinstance(tc_data, list) else []):
                        args_raw = tc.get("arguments_raw", "{}")
                        args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                        code = args.get("code", "")
                        if "df[" in code:
                            day_text.setdefault(sim_date, []).append(
                                "[Queried dataset to analyze active questions]"
                            )
                except Exception:
                    pass

    # Merge into predictions
    for sim_date in predictions:
        searches = day_searches.get(sim_date, [])
        texts = day_text.get(sim_date, [])
        # Build reasoning: text analysis first, then searches as evidence
        parts = []
        for t in texts[:3]:  # up to 3 analysis snippets
            clean = t.replace("\n", " ").strip()[:250]
            if clean and clean not in parts:
                parts.append(clean)
        if searches:
            parts.append(f"Searched: {', '.join(searches[:3])}")
        predictions[sim_date]["reasoning"] = " | ".join(parts)[:600]
        predictions[sim_date]["searches"] = searches


def _extract_sentiment_scores(raw_log_path: str, predictions: Dict[str, dict]):
    """Extract market sentiment scores from evidence log entries (type=sentiment)."""
    try:
        with open(raw_log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                sim_date = entry.get("sim_date", "")
                # Evidence items are logged with the evidence dict as prompt, and
                # the response field contains the JSON string of the evidence item.
                # Check both the response field (for evidence_* phase entries) and
                # direct metadata.
                meta = entry.get("metadata", {})
                phase = meta.get("phase", "")
                if phase == "evidence_sentiment":
                    try:
                        resp = entry.get("response", "")
                        if isinstance(resp, str):
                            evidence = json.loads(resp)
                        else:
                            evidence = resp
                        if isinstance(evidence, dict) and evidence.get("type") == "sentiment":
                            score = evidence.get("market_sentiment_score")
                            if score is not None:
                                predictions.setdefault(sim_date, {})["sentiment_score"] = float(score)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue
                # Also try extracting from the prompt field directly (some paths)
                prompt = entry.get("prompt", "")
                if isinstance(prompt, dict) and prompt.get("type") == "sentiment":
                    score = prompt.get("market_sentiment_score")
                    if score is not None and sim_date:
                        predictions.setdefault(sim_date, {})["sentiment_score"] = float(score)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"  [Sentiment] Warning: error reading sentiment scores: {e}")


# ===================================================================
# Polymarket history
# ===================================================================

def load_pm_history(slug: str) -> Dict[str, float]:
    csv_path = REPO_ROOT / f"polymarket_history_{slug}.csv"
    if not csv_path.exists():
        return {}
    history = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d_str = row.get("date", "").strip()
            p_str = row.get("probability", "").strip()
            if d_str and p_str:
                try:
                    history[d_str] = float(p_str)
                except ValueError:
                    continue
    return history


def build_pm_datasets(csv_history: Dict[str, float], gamma_history: Dict[str, float],
                       dates: List[str]) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    """Build two Polymarket chart series:
    - snapshots: dot markers from CSV snapshots (nulls elsewhere)
    - trend_line: Gamma-estimated daily values as a dashed line (nulls where no estimate)
    """
    sorted_dates = sorted(set(dates))
    snapshots: List[Optional[float]] = []
    trend_line: List[Optional[float]] = []
    for d in sorted_dates:
        if d in csv_history:
            snapshots.append(csv_history[d] * 100)
        else:
            snapshots.append(None)
        if d in gamma_history:
            trend_line.append(gamma_history[d] * 100)
        else:
            trend_line.append(None)
    return snapshots, trend_line


def record_pm_snapshot(slug: str) -> Optional[float]:
    market = PolymarketClient.lookup_by_slug(slug)
    if not market:
        return None
    prob = PolymarketClient.get_current_probability(market)
    if prob is None:
        return None
    csv_path = REPO_ROOT / f"polymarket_history_{slug}.csv"
    today_str = date.today().isoformat()
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["date", "slug", "probability"])
        w.writerow([today_str, slug, prob])
    print(f"[Snapshot] {slug} = {prob:.4f} @ {today_str}")
    return prob


# ===================================================================
# Sentiment / Emotional Analysis (rule-based)
# ===================================================================

def generate_sentiment_analysis(daily_data: List[dict],
                                csv_history: Dict[str, float]) -> str:
    """Produce a rule-based emotional analysis of the market over time.

    Uses daily sentiment scores, price movement, and agent-market gaps
    to generate a narrative similar to:
      "Polymarket shows 'overly pessimistic' signals..."
    """
    if not daily_data:
        return ""

    # Collect data across the date range
    scores = []
    pm_prices = []
    agent_prices = []
    dates = []
    for d in daily_data:
        sent = d.get("sentiment_score")
        agent_p = d.get("prob_yes")
        pm_p = csv_history.get(d["date"])
        dates.append(d["date"])
        scores.append(sent)
        pm_prices.append(pm_p)
        agent_prices.append(agent_p)

    has_scores = any(s is not None for s in scores)
    has_pm = any(p is not None for p in pm_prices)

    if not has_scores and not has_pm:
        return ""

    paragraphs = []

    # ── 1. Overall sentiment verdict ──────────────────────────────
    if has_scores:
        valid_scores = [s for s in scores if s is not None]
        if valid_scores:
            avg_score = sum(valid_scores) / len(valid_scores)
            last_score = valid_scores[-1]
            if avg_score <= -0.5:
                verdict = "overly pessimistic"
            elif avg_score >= 0.5:
                verdict = "overly optimistic"
            elif avg_score <= -0.2:
                verdict = "slightly pessimistic"
            elif avg_score >= 0.2:
                verdict = "slightly optimistic"
            else:
                verdict = "balanced"

            recent_scores = valid_scores[-3:] if len(valid_scores) >= 3 else valid_scores
            if all(s <= -0.7 for s in recent_scores):
                intensity = "strong and persistent"
            elif all(s <= -0.5 for s in recent_scores):
                intensity = "consistent"
            elif len(recent_scores) >= 2 and recent_scores[-1] <= -0.6:
                intensity = "intensifying"
            else:
                intensity = "moderate"

            paragraphs.append(
                f"Sentiment Analysis Result: Polymarket shows \"{verdict}\" tendencies "
                f"(average sentiment score: {avg_score:+.2f}, last score: {last_score:+.2f}). "
                f"The signal is {intensity}."
            )

            # Consistent extreme periods
            extreme_streaks = _find_extreme_streaks(zip(dates, valid_scores))
            if extreme_streaks:
                for start_d, end_d, label in extreme_streaks:
                    paragraphs.append(
                        f"From {start_d} to {end_d}, sentiment scores were consistently "
                        f"{label}, indicating a sustained emotional bias in the market "
                        f"during this period."
                    )

    # ── 2. Extreme price level analysis ────────────────────────────
    if has_pm:
        valid_pm = [(d, p) for d, p in zip(dates, pm_prices) if p is not None]
        if valid_pm:
            first_pm = valid_pm[0][1]
            last_pm = valid_pm[-1][1]
            max_pm = max(p for _, p in valid_pm)
            min_pm = min(p for _, p in valid_pm)

            if last_pm < 0.01:
                paragraphs.append(
                    f"Extreme price ({last_pm*100:.1f}%) is itself a strong emotional indicator. "
                    f"When a market prices below 1%, it typically represents a \"near-impossible\" "
                    f"consensus. Such extreme values often accompany systematic pessimism, but "
                    f"they are also fertile ground for irrational sentiment — even mildly "
                    f"positive news can cause prices to multiply."
                )
            elif last_pm > 0.99:
                paragraphs.append(
                    f"Extreme price ({last_pm*100:.1f}%) signals near-certainty among "
                    f"market participants. Prices above 99% often reflect complacency "
                    f"or euphoria — the market believes the outcome is guaranteed."
                )

            # Price range
            pm_range = max_pm - min_pm
            if pm_range > 0.15:
                paragraphs.append(
                    f"The price has swung {pm_range*100:.0f} points over the observation "
                    f"window (from {min_pm*100:.1f}% to {max_pm*100:.1f}%)."
                )

    # ── 3. Price crash / trend detection ───────────────────────────
    if has_pm:
        valid_pm_dates = [(d, p) for d, p in zip(dates, pm_prices) if p is not None]
        if len(valid_pm_dates) >= 3:
            first_price = valid_pm_dates[0][1]
            last_price = valid_pm_dates[-1][1]
            price_drop = first_price - last_price
            if price_drop > 0.10:
                paragraphs.append(
                    f"The price collapse process reveals panic-style selling: "
                    f"from {first_price*100:.1f}% at the start to {last_price*100:.1f}% "
                    f"at the end — a drop of {price_drop*100:.1f} percentage points. "
                    f"Without accompanying catastrophic news in the agent's search results, "
                    f"this likely reflects emotional selling (e.g., original bullish funds "
                    f"exiting, or a \"Google can't win\" narrative taking hold)."
                )
            elif price_drop < -0.10:
                paragraphs.append(
                    f"The price surged from {first_price*100:.1f}% to {last_price*100:.1f}% "
                    f"(+{abs(price_drop)*100:.1f} points). Without proportional news, "
                    f"this may indicate FOMO-driven buying or herd behavior."
                )

    # ── 4. Evidence the market may be ignoring ─────────────────────
    agent_gaps = []
    for d in daily_data:
        agent_p = d.get("prob_yes")
        pm_p = csv_history.get(d["date"])
        if agent_p is not None and pm_p is not None:
            gap = agent_p - pm_p
            if abs(gap) > 0.15:
                agent_gaps.append((d["date"], gap, d.get("reasoning", "")))

    if agent_gaps:
        # Find unique reasoning themes in gap days
        gap_themes = []
        for _, gap, reasoning in agent_gaps[-3:]:
            # Extract key phrases from reasoning (first 200 chars)
            snippet = reasoning[:200] if reasoning else ""
            if snippet and len(snippet) > 20:
                gap_themes.append(snippet)

        if gap_themes:
            paragraphs.append(
                f"The agent found evidence that the market appears to be ignoring. "
                f"On days with large agent-market gaps (>15 points), the agent's "
                f"reasoning included themes such as: "
                + "; ".join(f"\"{t[:100]}...\"" for t in gap_themes[:2])
                + ". If the market is not incorporating this information, it may "
                + "represent a genuine information edge rather than pure sentiment."
            )

    if not paragraphs:
        return ""

    # Format as the example shows
    lines = ["Emotion Analysis Result: Polymarket shows the following signals:\n"]
    for i, p in enumerate(paragraphs):
        lines.append(p)
        lines.append("")
    return "\n".join(lines).strip()


def _find_extreme_streaks(date_score_pairs) -> List[Tuple[str, str, str]]:
    """Find consecutive periods where sentiment scores were extreme."""
    streaks = []
    current_start = None
    current_end = None
    current_label = None
    count = 0

    for d, s in date_score_pairs:
        if s is None:
            continue
        if s <= -0.7:
            label = "overly pessimistic (<= -0.7)"
        elif s >= 0.7:
            label = "overly optimistic (>= +0.7)"
        elif s <= -0.5:
            label = "moderately pessimistic"
        elif s >= 0.5:
            label = "moderately optimistic"
        else:
            label = None

        if label is None or label != current_label:
            if current_start and count >= 2:
                streaks.append((current_start, current_end, current_label))
            current_start = d if label else None
            current_end = d if label else None
            current_label = label
            count = 1 if label else 0
        else:
            current_end = d
            count += 1

    if current_start and count >= 2:
        streaks.append((current_start, current_end, current_label))

    return streaks


# ===================================================================
# AI Narrative Summary
# ===================================================================

def generate_ai_summary(market: dict, daily_data: List[dict],
                        csv_history: Dict[str, float],
                        model: str = "deepseek-v4-flash") -> str:
    """Use an LLM to produce a plain-English narrative explaining:
    - What evidence the AI agent relied on
    - Why the agent's view differs from (or matches) Polymarket
    - Key turning points in the agent's predictions over time
    """
    if not daily_data:
        return ""

    title = market.get("question", market.get("title", "Unknown"))
    current_pm = PolymarketClient.get_current_probability(market) or 0

    # Build a compact day-by-day table for the prompt
    day_lines = []
    for d in daily_data:
        agent_p = d["prob_yes"]
        pm_raw = csv_history.get(d["date"])
        pm_p = pm_raw if pm_raw is not None else None
        gap = (agent_p - pm_p) if pm_p is not None else None
        reasoning = d.get("reasoning", "")[:300]
        searches = d.get("searches", [])[:3]
        sentiment = d.get("sentiment_score")
        gap_str = f"{gap:+.1%}" if gap is not None else "no PM data"
        pm_str = f"{pm_p:.1%}" if pm_p is not None else "—"
        sent_str = f" | sentiment={sentiment:+.2f}" if sentiment is not None else ""
        day_lines.append(
            f"  {d['date']}: agent={agent_p:.1%} | polymarket={pm_str} | "
            f"gap={gap_str}{sent_str}\n"
            f"    evidence: {reasoning or '(none)'}\n"
            f"    searches: {', '.join(searches) if searches else '(none)'}"
        )

    days_text = "\n".join(day_lines)

    # Gather sentiment context if available
    sentiment_context = ""
    has_sentiment = any(d.get("sentiment_score") is not None for d in daily_data)
    if has_sentiment:
        scores = [d.get("sentiment_score") for d in daily_data if d.get("sentiment_score") is not None]
        if scores:
            sentiment_context = (
                f"\nMarket sentiment scores range from {min(scores):+.2f} to {max(scores):+.2f} "
                f"(where -1.0 = overly pessimistic, +1.0 = overly optimistic, 0.0 = balanced)."
            )

    prompt = f"""You are a forecasting analyst explaining AI agent predictions to a general audience.

MARKET QUESTION: {title}
Current Polymarket probability: {current_pm:.1%}{sentiment_context}

Below is a day-by-day log of an AI forecasting agent's predictions compared to Polymarket odds:

{days_text}

Write a 2-3 paragraph plain-English narrative (like a news analyst would) that explains:

1. EVIDENCE: What information and evidence did the AI agent rely on to form its view? Summarize the key facts, searches, or reasoning the agent used.

2. THE GAP: Why does the AI agent's probability differ from Polymarket's (or agree with it)? What specific evidence or perspective explains the difference? Point to concrete examples from the log above.

3. TREND: Did the agent's opinion shift over time? If so, what new information caused the shift?

4. SENTIMENT: Based on the sentiment scores, was the market emotionally skewed (overly pessimistic or optimistic)? Did the agent detect and exploit this?

Keep it concise and readable. Avoid jargon. Use specific numbers from the log to support your points."""

    try:
        inference = DeepSeekInference(model, max_retries=1, base_delay=2, max_delay=10)
        text, _ = inference.chat(
            messages=[{"role": "user", "content": prompt}],
            sampling_params={"temperature": 0.3, "max_tokens": 800},
        )
        return text.strip()
    except Exception as e:
        print(f"  [Summary] LLM call failed: {e}")
        return ""


# ===================================================================
# HTML Dashboard generator
# ===================================================================

def _summary_card(summary_text: str) -> str:
    """Build the AI narrative summary card HTML."""
    if not summary_text:
        return ""
    escaped = summary_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Convert markdown-style **bold** to <strong> tags, and newline-separated
    # paragraphs to <p> tags for readability
    import re
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    paragraphs = escaped.split("\n\n")
    paras_html = "\n".join(f"<p>{p.replace(chr(10), '<br>')}</p>" for p in paragraphs if p.strip())
    return f"""
<div class="card">
    <h2>AI Analyst Summary</h2>
    <div class="summary-text">
        {paras_html}
    </div>
</div>"""


def _sentiment_card(sentiment_text: str) -> str:
    """Build the sentiment / emotional analysis card HTML."""
    if not sentiment_text:
        return ""
    escaped = sentiment_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    import re
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    paragraphs = escaped.split("\n\n")
    paras_html = "\n".join(f"<p>{p.replace(chr(10), '<br>')}</p>" for p in paragraphs if p.strip())
    return f"""
<div class="card">
    <h2>Market Sentiment & Emotional Analysis</h2>
    <div class="summary-text">
        {paras_html}
    </div>
</div>"""


def generate_dashboard(market: dict, daily_data: List[dict],
                       csv_history: Dict[str, float], gamma_history: Dict[str, float],
                       output_path: Path, current_pm_prob: float,
                       ai_summary: str = "", sentiment_analysis: str = ""):
    """Build a standalone HTML dashboard file."""
    title = market.get("question", market.get("title", "Polymarket Market"))
    slug = market.get("slug", "unknown")

    # Prepare chart data
    labels = [d["date"] for d in daily_data]
    agent_data = [d["prob_yes"] * 100 for d in daily_data]
    pm_snapshots, pm_trend_line = build_pm_datasets(csv_history, gamma_history, labels)

    # Collect sentiment scores for display
    has_sentiment = any(d.get("sentiment_score") is not None for d in daily_data)

    # Build a lookup for the detail table (prefer CSV snapshots, fall back to Gamma)
    pm_lookup = {}
    for d in labels:
        if d in csv_history:
            pm_lookup[d] = csv_history[d]
        elif d in gamma_history:
            pm_lookup[d] = gamma_history[d]
        else:
            pm_lookup[d] = None  # No Polymarket data for this date

    # Build reasoning rows with full text in data attribute for tooltip
    reasoning_rows = ""
    for d in daily_data:
        agent_p = d["prob_yes"] * 100
        pm_raw = pm_lookup.get(d["date"])
        pm_p = pm_raw * 100 if pm_raw is not None else None
        if pm_p is not None:
            diff = agent_p - pm_p
        else:
            diff = None
        searches = d.get("searches", [])
        search_str = ", ".join(searches[:3]) if searches else ""
        reasoning = d.get("reasoning", "")
        if not reasoning and searches:
            reasoning = f"Searched: {search_str}"
        elif not reasoning:
            reasoning = "No reasoning recorded."
        pm_is_snapshot = d["date"] in csv_history
        if pm_p is not None:
            pm_label = f"{pm_p:.1f}%"
            if pm_is_snapshot:
                pm_label += " *"
        else:
            pm_label = "—"
        if diff is not None:
            diff_class = "positive" if diff > 0 else "negative" if diff < 0 else "neutral"
            diff_str = f"{diff:+.1f}%"
        else:
            diff_class = "neutral"
            diff_str = "—"
        # Sentiment score cell
        sent_score = d.get("sentiment_score")
        if sent_score is not None:
            if sent_score <= -0.5:
                sent_class = "negative"
                sent_emoji = "&#128553;"  # weary face
            elif sent_score <= -0.2:
                sent_class = "sentiment-mild-bear"
                sent_emoji = "&#128542;"  # disappointed
            elif sent_score >= 0.5:
                sent_class = "positive"
                sent_emoji = "&#128513;"  # grinning
            elif sent_score >= 0.2:
                sent_class = "sentiment-mild-bull"
                sent_emoji = "&#128522;"  # smiling
            else:
                sent_class = "neutral"
                sent_emoji = "&#128528;"  # neutral face
            sent_str = f"{sent_score:+.2f}"
        else:
            sent_class = "neutral"
            sent_str = "—"
            sent_emoji = ""

        # Escape reasoning for HTML attribute
        reasoning_escaped = reasoning.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#39;")
        sent_cell = f'<td class="{sent_class}">{sent_emoji} {sent_str}</td>' if has_sentiment else ""
        reasoning_rows += f"""
        <tr>
            <td>{d['date']}</td>
            <td>{pm_label}</td>
            <td>{agent_p:.1f}%</td>
            <td class="{diff_class}">{diff_str}</td>
            {sent_cell}
            <td class="reasoning" data-full="{reasoning_escaped}">{reasoning[:200]}{'...' if len(reasoning) > 200 else ''}</td>
        </tr>"""

    has_trend = any(v is not None for v in pm_trend_line)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Futuresim Dashboard — {title[:60]}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #0f172a; color: #e2e8f0; padding: 24px 40px; }}
h1 {{ font-size: 1.5rem; font-weight: 600; margin-bottom: 4px; }}
.subtitle {{ color: #94a3b8; font-size: 0.9rem; margin-bottom: 24px; }}
.card {{ background: #1e293b; border-radius: 12px; padding: 24px; margin-bottom: 24px;
         border: 1px solid #334155; }}
.card h2 {{ font-size: 1.1rem; margin-bottom: 16px; color: #cbd5e1; }}
.chart-container {{ min-height: 420px; height: 55vh; position: relative; }}
.card.scrollable {{ max-height: 60vh; overflow-y: auto; }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
th {{ text-align: left; padding: 10px 12px; border-bottom: 2px solid #475569;
      color: #94a3b8; font-weight: 500; position: sticky; top: 0; background: #1e293b; z-index: 1; }}
td {{ padding: 10px 12px; border-bottom: 1px solid #334155; }}
.reasoning {{ max-width: 500px; font-size: 0.8rem; color: #94a3b8; cursor: pointer;
    word-break: break-word; }}
.reasoning:hover {{ color: #e2e8f0; }}
.positive {{ color: #4ade80; font-weight: 600; }}
.negative {{ color: #f87171; font-weight: 600; }}
.neutral {{ color: #94a3b8; font-weight: 600; }}
.tooltip-box {{ position: fixed; background: #0f172a; color: #e2e8f0; border: 1px solid #475569;
    border-radius: 8px; padding: 16px 20px; max-width: 540px; font-size: 0.85rem;
    line-height: 1.5; box-shadow: 0 8px 24px rgba(0,0,0,0.6); z-index: 1000;
    pointer-events: none; opacity: 0; transition: opacity 0.15s; }}
.tooltip-box.visible {{ opacity: 1; }}
.legend {{ display: flex; gap: 20px; font-size: 0.8rem; color: #94a3b8; margin-top: 8px; flex-wrap: wrap; }}
.legend span {{ display: flex; align-items: center; gap: 6px; }}
.legend .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
.legend .dash {{ width: 16px; height: 0; border-top: 2px dashed #38bdf8; display: inline-block; }}
.stats {{ display: flex; gap: 24px; flex-wrap: wrap; }}
.stat {{ background: #0f172a; border-radius: 8px; padding: 16px 20px; min-width: 140px; }}
.stat .value {{ font-size: 1.6rem; font-weight: 700; margin-top: 4px; }}
.stat .label {{ font-size: 0.8rem; color: #94a3b8; }}
.summary-text {{ font-size: 0.9rem; line-height: 1.7; color: #cbd5e1; }}
.summary-text p {{ margin-bottom: 12px; }}
.summary-text strong {{ color: #f59e0b; }}
.sentiment-mild-bear {{ color: #fb923c; font-weight: 500; }}
.sentiment-mild-bull {{ color: #34d399; font-weight: 500; }}
</style>
</head>
<body>

<h1>Futuresim vs Polymarket</h1>
<div class="subtitle">{title} &nbsp;|&nbsp; Slug: <code>{slug}</code></div>

<div class="stats">
    <div class="stat">
        <div class="label">Current Polymarket</div>
        <div class="value">{current_pm_prob*100:.1f}% Yes</div>
    </div>
    <div class="stat">
        <div class="label">AI Agent Opinion</div>
        <div class="value">{agent_data[-1] if agent_data else 'N/A'}% Yes</div>
    </div>
    <div class="stat">
        <div class="label">Resolution Date</div>
        <div class="value" style="font-size:1.2rem">{market.get('endDate', market.get('endDateIso', '?'))[:10]}</div>
    </div>
</div>

<div class="card">
    <h2>Probability Over Time</h2>
    <div class="legend">
        <span><span class="dot" style="background:#38bdf8"></span> PM snapshots (recorded)</span>{'<span><span class="dash"></span> PM trend (Gamma estimate)</span>' if has_trend else ''}
        <span><span style="display:inline-block;width:14px;height:0;border-top:2.5px solid #f59e0b;vertical-align:middle"></span> AI Agent</span>
    </div>
    <div style="color:#64748b;font-size:0.75rem;margin-top:4px">
        Polymarket does not expose public historical price data (CLOB /prices-history always empty).
        {'The dashed line is estimated from Gamma API priceChange deltas. ' if has_trend else ''}Blue dots are real CSV snapshots recorded over time. The agent line shows what the AI predicted each day.
    </div>
    <div class="chart-container">
        <canvas id="probabilityChart"></canvas>
    </div>
</div>

{_summary_card(ai_summary)}

{_sentiment_card(sentiment_analysis)}

<div class="card scrollable">
    <h2>Daily Details &amp; AI Reasoning</h2>
    <table>
        <thead>
            <tr><th>Date</th><th>Polymarket</th><th>AI Agent</th><th>Diff</th>{
            '<th>Sentiment</th>' if has_sentiment else ''
            }<th>AI Reasoning / Evidence</th></tr>
        </thead>
        <tbody>{reasoning_rows}</tbody>
    </table>
</div>

<script>
const ctx = document.getElementById('probabilityChart').getContext('2d');
new Chart(ctx, {{
    type: 'line',
    data: {{
        labels: {json.dumps(labels)},
        datasets: [
            {{
                label: 'Polymarket (snapshot)',
                data: {json.dumps(pm_snapshots)},
                borderColor: '#38bdf8',
                backgroundColor: '#38bdf8',
                tension: 0,
                pointRadius: 6,
                pointStyle: 'circle',
                showLine: false,
                spanGaps: false,
                fill: false,
                order: 1,
            }},
            {{
                label: 'Polymarket (trend)',
                data: {json.dumps(pm_trend_line if has_trend else [])},
                borderColor: '#38bdf8',
                backgroundColor: 'transparent',
                borderDash: [6, 4],
                borderWidth: 1,
                tension: 0.2,
                pointRadius: 0,
                fill: false,
                order: 2,
            }},
            {{
                label: 'AI Agent',
                data: {json.dumps(agent_data)},
                borderColor: '#f59e0b',
                backgroundColor: 'rgba(245,158,11,0.08)',
                tension: 0.3,
                pointRadius: 4,
                borderWidth: 2.5,
                fill: true,
                order: 0,
            }}
        ]
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: false,
        interaction: {{
            mode: 'index',
            intersect: false,
        }},
        scales: {{
            y: {{ min: 0, max: 100, ticks: {{ color: '#94a3b8', callback: v => v + '%', stepSize: 10 }},
                  grid: {{ color: '#334155' }} }},
            x: {{ ticks: {{ color: '#94a3b8', maxTicksLimit: 20, maxRotation: 45 }},
                  grid: {{ color: '#334155' }} }}
        }},
        plugins: {{
            legend: {{ labels: {{ color: '#e2e8f0' }} }},
            tooltip: {{
                callbacks: {{
                    afterBody: function(items) {{
                        if (items.length && items[0].dataset.label === 'AI Agent') {{
                            var idx = items[0].dataIndex;
                            var reasons = {json.dumps([d.get('reasoning','') for d in daily_data])};
                            if (idx < reasons.length && reasons[idx]) {{
                                var text = reasons[idx];
                                // Word-wrap at ~80 chars
                                return '\\n' + text.replace(/(.{{70,90}}) /g, '$1\\n');
                            }}
                        }}
                        return '';
                    }}
                }}
            }}
        }}
    }}
}});
</script>
<div class="tooltip-box" id="tooltip"></div>
<script>
// Hover tooltip on reasoning cells — show FULL reasoning text
const tooltip = document.getElementById('tooltip');
document.querySelectorAll('.reasoning').forEach(el => {{
    el.addEventListener('mouseenter', e => {{
        var full = el.getAttribute('data-full') || el.textContent.trim();
        if (full) {{
            tooltip.textContent = full;
            tooltip.classList.add('visible');
            positionTooltip(e);
        }}
    }});
    el.addEventListener('mousemove', e => positionTooltip(e));
    el.addEventListener('mouseleave', () => tooltip.classList.remove('visible'));
}});
function positionTooltip(e) {{
    var left = Math.min(e.clientX + 16, window.innerWidth - 560);
    var top = Math.min(e.clientY + 12, window.innerHeight - tooltip.offsetHeight - 12);
    tooltip.style.left = left + 'px';
    tooltip.style.top = top + 'px';
}}
</script>
</body></html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[Dashboard] Generated: {output_path}")
    print(f"  Open in browser: file:///{output_path.resolve()}")


# ===================================================================
# CLI
# ===================================================================

def _find_existing_output(slug: str) -> Optional[str]:
    """Find the most recent simulation output directory for this slug."""
    log_base = REPO_ROOT / "logs" / "current_sim" / "dashboard"
    if not log_base.exists():
        return None
    dirs = sorted(log_base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for d in dirs:
        if d.is_dir() and (d / "config.json").exists():
            return str(d)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Build a Polymarket + Futuresim comparison dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python scripts/build_dashboard.py --slug new-rhianna-album-before-gta-vi-926
              python scripts/build_dashboard.py --search "Fed rate hike"
              python scripts/build_dashboard.py --slug some-market --record-snapshot
        """),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--slug", help="Polymarket market slug")
    group.add_argument("--search", help="Keyword search for a Polymarket market")

    parser.add_argument("--start_date", default=None,
                        help="Sim start YYYY-MM-DD (default: today - 14 days)")
    parser.add_argument("--end_date", default=None,
                        help="Sim end YYYY-MM-DD (default: start_date + 30 days)")
    parser.add_argument("--model", default="deepseek-v4-flash",
                        help="Model (default: deepseek-v4-flash)")
    parser.add_argument("--matching", default="exact",
                        help="Matching mode (default: exact)")
    parser.add_argument("--max_actions", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Subprocess timeout in seconds (default: 3600 = 60 min)")
    parser.add_argument("--output", default=None,
                        help="Output HTML path (default: dashboard_<slug>.html)")
    parser.add_argument("--summary-only", action="store_true",
                        help="Skip simulation; regenerate summary + dashboard from existing output")

    args = parser.parse_args()

    # ── 1. Fetch market ─────────────────────────────────────────────
    print("=" * 60)
    print("Step 1/6: Fetching Polymarket market...")
    market = PolymarketClient.resolve_market(slug=args.slug, keyword=args.search)
    if not market:
        sys.exit(1)

    slug = market.get("slug", args.slug or "unknown")
    title = market.get("question", "Unknown")
    current_prob = PolymarketClient.get_current_probability(market) or 0
    print(f"  {title}")
    print(f"  Current probability: {current_prob:.4f}")

    work_dir = REPO_ROOT / "logs" / "dashboard" / slug
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        # Skip simulation — find existing output directory
        output_dir = _find_existing_output(slug)
        if not output_dir:
            print("[Dashboard] No existing simulation output found. Run without --summary-only first.")
            sys.exit(1)
        print(f"\nStep 2/6: Using existing output: {output_dir}")
        print("Step 3/6: Skipped (--summary-only).")

        # Read config to get start/end dates used in the original run
        config_path = Path(output_dir) / "config.json"
        start_date = end_date = None
        if config_path.exists():
            try:
                cfg = json.loads(config_path.read_text())
                start_date = cfg.get("start_date", "") or cfg.get("sim_start_date", "")
                end_date = cfg.get("end_date", "") or cfg.get("sim_end_date", "")
            except Exception:
                pass
        if not start_date:
            today = date.today()
            start_date = (today - timedelta(days=14)).isoformat()
            end_date = today.isoformat()
    else:
        # ── 2. Build question ───────────────────────────────────────
        print("\nStep 2/6: Building custom question...")
        jsonl_path = work_dir / "question.jsonl"
        q = build_custom_question_jsonl(market, jsonl_path)

        # ── 3. Determine dates & run futuresim ──────────────────────
        today = date.today()
        start_date = args.start_date or (today - timedelta(days=14)).isoformat()
        if args.end_date:
            end_date = args.end_date
        else:
            end_date = (date.fromisoformat(start_date) + timedelta(days=14)).isoformat() if start_date else today.isoformat()

        res_date_str = q.get("resolution_date", "")
        resolution_end = None
        if res_date_str and not args.end_date:
            try:
                res_dt = date.fromisoformat(res_date_str)
                sim_end_dt = date.fromisoformat(end_date) if isinstance(end_date, str) else end_date
                if res_dt > sim_end_dt:
                    resolution_end = (res_dt + timedelta(days=7)).isoformat()
            except ValueError:
                pass

        print(f"\nStep 3/6: Running futuresim ({start_date} -> {end_date})...")
        output_dir = run_futuresim(jsonl_path, start_date, end_date,
                                   provider="deepseek", model=args.model,
                                   matching=args.matching, max_actions=args.max_actions,
                                   force_submit=True, timeout=args.timeout,
                                   resolution_end=resolution_end)
        if not output_dir:
            print("\n[Dashboard] Simulation failed. Check errors above.")
            sys.exit(1)

    # ── 4. Extract data & fetch real-time Polymarket history ─────────
    print("\nStep 4/6: Extracting predictions and reasoning...")
    daily_data = extract_daily_data(output_dir)
    if not daily_data:
        print("[Dashboard] No predictions found. Agent made 0 submissions.")
        print("[Dashboard] Try a longer date range or --model deepseek-v4-pro.")
        daily_data = []  # Continue with empty, dashboard will show empty state

    for d in daily_data:
        print(f"  {d['date']}: {d['prob_yes']*100:.1f}% Yes | "
              f"{'searches: '+str(len(d.get('searches',[]))) if d.get('searches') else 'no research'}")

    # Fetch Gamma API trend data for approximate daily history
    gamma_history = fetch_pm_price_history(market, date.fromisoformat(start_date), date.fromisoformat(end_date))
    if gamma_history:
        print(f"\n  Gamma trend history: {len(gamma_history)} daily prices (from priceChange deltas)")

    # Record today's snapshot (updates csv_history for dashboard display)
    record_pm_snapshot(slug)
    csv_history = load_pm_history(slug)

    if csv_history:
        snapshot_dates_list = sorted(csv_history.keys())
        print(f"  CSV snapshots: {len(csv_history)} points ({snapshot_dates_list[0]} -> {snapshot_dates_list[-1]})")

    # Show all agent predictions — PM snapshots overlay as dots on matching dates

    if len(csv_history) >= 2:
        sdates = sorted(csv_history.keys())
        print(f"  CSV history: {len(csv_history)} data points ({sdates[0]} -> {sdates[-1]})")
    elif len(csv_history) == 1:
        print(f"  CSV history: 1 data point. Run daily for more.")
    else:
        print("  No CSV history. Snapshot will be recorded each run.")

    # ── 5. Generate AI narrative summary ─────────────────────────────
    print("\nStep 5/6: Generating AI narrative summary...")
    ai_summary = generate_ai_summary(market, daily_data, csv_history, model=args.model)
    if ai_summary:
        preview = ai_summary[:120] + "..." if len(ai_summary) > 120 else ai_summary
        print(f"  Summary: {preview}")
    else:
        print("  No summary generated (LLM call failed or no data).")

    # ── 5b. Generate sentiment analysis ──────────────────────────────
    sentiment_analysis = generate_sentiment_analysis(daily_data, csv_history)
    if sentiment_analysis:
        preview = sentiment_analysis[:120] + "..." if len(sentiment_analysis) > 120 else sentiment_analysis
        print(f"  Sentiment: {preview}")

    # ── 6. Generate dashboard ───────────────────────────────────────
    print("\nStep 6/6: Generating dashboard HTML...")
    output_path = Path(args.output) if args.output else (work_dir / f"dashboard_{slug}.html")
    generate_dashboard(market, daily_data, csv_history, gamma_history, output_path, current_prob,
                       ai_summary, sentiment_analysis)
    print("\n" + "=" * 60)
    print("Dashboard ready! Open the HTML file in any browser.")


if __name__ == "__main__":
    main()
