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
                  force_submit: bool = True, timeout: int = 1800,
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

    # Extract reasoning from raw daily log
    if agent_dir:
        raw_log = os.path.join(agent_dir, "model_raw_daily.jsonl")
        if os.path.exists(raw_log):
            _extract_reasoning(raw_log, predictions)

    # Build daily list
    result = []
    for d in sorted(predictions):
        entry = predictions[d]
        result.append({
            "date": d,
            "prob_yes": entry.get("prob_yes", 0),
            "reasoning": entry.get("reasoning", ""),
            "searches": entry.get("searches", []),
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


def build_pm_datasets(history: Dict[str, float], dates: List[str],
                       current_prob: float) -> Tuple[List[Optional[float]], List[float]]:
    """Build two Polymarket chart series:
    - snapshots: dot markers only at dates with recorded data (nulls elsewhere)
    - current_line: a flat reference line at the current probability
    """
    sorted_dates = sorted(set(dates))
    snapshots: List[Optional[float]] = []
    for d in sorted_dates:
        if d in history:
            snapshots.append(history[d] * 100)
        else:
            snapshots.append(None)  # null = no marker on Chart.js
    current_line = [current_prob * 100] * len(sorted_dates)
    return snapshots, current_line


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
# HTML Dashboard generator
# ===================================================================

def generate_dashboard(market: dict, daily_data: List[dict],
                       pm_history: Dict[str, float], output_path: Path,
                       current_pm_prob: float):
    """Build a standalone HTML dashboard file."""
    title = market.get("question", market.get("title", "Polymarket Market"))
    slug = market.get("slug", "unknown")

    # Prepare chart data
    labels = [d["date"] for d in daily_data]
    agent_data = [d["prob_yes"] * 100 for d in daily_data]
    pm_snapshots, pm_current_line = build_pm_datasets(pm_history, labels, current_pm_prob)

    # Build a lookup for the detail table
    pm_lookup = {}
    for i, d in enumerate(labels):
        if pm_snapshots[i] is not None:
            pm_lookup[d] = pm_snapshots[i] / 100
        else:
            pm_lookup[d] = current_pm_prob

    # Build reasoning rows
    reasoning_rows = ""
    for d in daily_data:
        agent_p = d["prob_yes"] * 100
        pm_p = (pm_lookup.get(d["date"], current_pm_prob) or 0) * 100
        diff = agent_p - pm_p
        searches = d.get("searches", [])
        search_str = ", ".join(searches[:3]) if searches else ""
        reasoning = d.get("reasoning", "")
        if not reasoning and searches:
            reasoning = f"Searched: {search_str}"
        elif not reasoning:
            reasoning = "No reasoning recorded."
        pm_is_snapshot = d["date"] in pm_history
        pm_label = f"{pm_p:.1f}%" + (" *" if pm_is_snapshot else "")
        diff_class = "positive" if diff > 0 else "negative"
        reasoning_rows += f"""
        <tr>
            <td>{d['date']}</td>
            <td>{pm_label}</td>
            <td>{agent_p:.1f}%</td>
            <td class=\"{diff_class}\">{diff:+.1f}%</td>
            <td class=\"reasoning\">{reasoning[:120]}{'...' if len(reasoning) > 120 else ''}</td>
        </tr>"""

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
.chart-container {{ height: 360px; position: relative; }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
th {{ text-align: left; padding: 10px 12px; border-bottom: 2px solid #475569;
      color: #94a3b8; font-weight: 500; }}
td {{ padding: 10px 12px; border-bottom: 1px solid #334155; }}
.reasoning {{ max-width: 340px; font-size: 0.8rem; color: #94a3b8; cursor: pointer; }}
.positive {{ color: #4ade80; font-weight: 600; }}
.negative {{ color: #f87171; font-weight: 600; }}
.tooltip-box {{ position: fixed; background: #0f172a; color: #e2e8f0; border: 1px solid #475569;
    border-radius: 8px; padding: 16px 20px; max-width: 480px; font-size: 0.85rem;
    line-height: 1.5; box-shadow: 0 8px 24px rgba(0,0,0,0.6); z-index: 1000;
    pointer-events: none; opacity: 0; transition: opacity 0.15s; }}
.tooltip-box.visible {{ opacity: 1; }}
.legend {{ display: flex; gap: 20px; font-size: 0.8rem; color: #94a3b8; margin-top: 8px; }}
.legend span {{ display: flex; align-items: center; gap: 6px; }}
.legend .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
.legend .dash {{ width: 16px; height: 0; border-top: 2px dashed #38bdf8; display: inline-block; }}
.stats {{ display: flex; gap: 24px; flex-wrap: wrap; }}
.stat {{ background: #0f172a; border-radius: 8px; padding: 16px 20px; min-width: 140px; }}
.stat .value {{ font-size: 1.6rem; font-weight: 700; margin-top: 4px; }}
.stat .label {{ font-size: 0.8rem; color: #94a3b8; }}
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
        <span><span class="dot" style="background:#38bdf8"></span> PM snapshot (recorded daily)</span>
        <span><span class="dash"></span> PM trend (from Gamma priceChange)</span>
    </div>
    <div style="color:#64748b;font-size:0.75rem;margin-top:4px">
        Polymarket does not expose public historical price data (CLOB /prices-history always empty).
        The dashed line is estimated from Gamma API's oneDay/oneWeek/oneMonth priceChange deltas
        applied to the current lastTradePrice. Blue dots are snapshots recorded each run.
    </div>
    <div class="chart-container">
        <canvas id="probabilityChart"></canvas>
    </div>
</div>

<div class="card">
    <h2>Daily Details &amp; AI Reasoning</h2>
    <table>
        <thead>
            <tr><th>Date</th><th>Polymarket</th><th>AI Agent</th><th>Diff</th><th>AI Reasoning / Evidence</th></tr>
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
                pointRadius: 5,
                pointStyle: 'circle',
                showLine: false,
                spanGaps: false,
                fill: false,
            }},
            {{
                label: 'Polymarket (trend)',
                data: {json.dumps(pm_current_line)},
                borderColor: '#38bdf8',
                backgroundColor: 'transparent',
                borderDash: [6, 4],
                borderWidth: 1,
                tension: 0,
                pointRadius: 0,
                fill: false,
            }},
            {{
                label: 'AI Agent',
                data: {json.dumps(agent_data)},
                borderColor: '#f59e0b',
                backgroundColor: 'rgba(245,158,11,0.1)',
                tension: 0.3,
                pointRadius: 4,
                borderWidth: 2,
                fill: true,
            }}
        ]
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: false,
        scales: {{
            y: {{ min: 0, max: 100, ticks: {{ color: '#94a3b8', callback: v => v + '%' }},
                  grid: {{ color: '#334155' }} }},
            x: {{ ticks: {{ color: '#94a3b8', maxTicksLimit: 14 }},
                  grid: {{ color: '#334155' }} }}
        }},
        plugins: {{
            legend: {{ labels: {{ color: '#e2e8f0' }} }},
            tooltip: {{
                callbacks: {{
                    afterBody: function(items) {{
                        if (items.length && items[0].dataset.label === 'AI Agent') {{
                            var idx = items[0].dataIndex;
                            var reasons = {json.dumps([d.get('reasoning','')[:200] for d in daily_data])};
                            if (idx < reasons.length && reasons[idx]) return 'Evidence: ' + reasons[idx];
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
// Hover tooltip on reasoning cells
const tooltip = document.getElementById('tooltip');
document.querySelectorAll('.reasoning').forEach(el => {{
    let tip = '';
    el.addEventListener('mouseenter', e => {{
        tip = el.textContent.trim();
        if (tip) {{
            tooltip.textContent = tip;
            tooltip.classList.add('visible');
            tooltip.style.left = Math.min(e.clientX + 16, window.innerWidth - 500) + 'px';
            tooltip.style.top = Math.min(e.clientY + 12, window.innerHeight - 120) + 'px';
        }}
    }});
    el.addEventListener('mousemove', e => {{
        tooltip.style.left = Math.min(e.clientX + 16, window.innerWidth - 500) + 'px';
        tooltip.style.top = Math.min(e.clientY + 12, window.innerHeight - 120) + 'px';
    }});
    el.addEventListener('mouseleave', () => tooltip.classList.remove('visible'));
}});
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

    args = parser.parse_args()

    # ── 1. Fetch market ─────────────────────────────────────────────
    print("=" * 60)
    print("Step 1/5: Fetching Polymarket market...")
    market = PolymarketClient.resolve_market(slug=args.slug, keyword=args.search)
    if not market:
        sys.exit(1)

    slug = market.get("slug", args.slug or "unknown")
    title = market.get("question", "Unknown")
    current_prob = PolymarketClient.get_current_probability(market) or 0
    print(f"  {title}")
    print(f"  Current probability: {current_prob:.4f}")

    # ── 2. Build question ───────────────────────────────────────────
    print("\nStep 2/5: Building custom question...")
    work_dir = REPO_ROOT / "logs" / "dashboard" / slug
    work_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = work_dir / "question.jsonl"
    q = build_custom_question_jsonl(market, jsonl_path)

    # ── 3. Determine dates & run futuresim ──────────────────────────
    today = date.today()
    start_date = args.start_date or (today - timedelta(days=14)).isoformat()
    if args.end_date:
        end_date = args.end_date
    else:
        end_date = (date.fromisoformat(start_date) + timedelta(days=14)).isoformat() if start_date else today.isoformat()
    # Ensure the resolution filter includes the market's resolution date
    res_date_str = q.get("resolution_date", "")
    resolution_end = None
    if res_date_str and not args.end_date:
        # Extend resolution_end to cover the question, but keep sim window short
        try:
            res_dt = date.fromisoformat(res_date_str)
            sim_end_dt = date.fromisoformat(end_date) if isinstance(end_date, str) else end_date
            if res_dt > sim_end_dt:
                resolution_end = (res_dt + timedelta(days=7)).isoformat()
        except ValueError:
            pass

    print(f"\nStep 3/5: Running futuresim ({start_date} -> {end_date})...")
    output_dir = run_futuresim(jsonl_path, start_date, end_date,
                               provider="deepseek", model=args.model,
                               matching=args.matching, max_actions=args.max_actions,
                               timeout=args.timeout, resolution_end=resolution_end)
    if not output_dir:
        print("\n[Dashboard] Simulation failed. Check errors above.")
        sys.exit(1)

    # ── 4. Extract data & history ───────────────────────────────────
    print("\nStep 4/5: Extracting predictions and reasoning...")
    daily_data = extract_daily_data(output_dir)
    if not daily_data:
        print("[Dashboard] No predictions found. Agent made 0 submissions.")
        print("[Dashboard] Try a longer date range or --model deepseek-v4-pro.")
        daily_data = []  # Continue with empty, dashboard will show empty state

    for d in daily_data:
        print(f"  {d['date']}: {d['prob_yes']*100:.1f}% Yes | "
              f"{'searches: '+str(len(d.get('searches',[]))) if d.get('searches') else 'no research'}")

    # Fetch Polymarket historical data
    pm_history: Dict[str, float] = {}

    # Try Gamma API trend data for approximate history
    gamma_history = fetch_pm_price_history(market, date.fromisoformat(start_date), date.fromisoformat(end_date))
    if gamma_history:
        print(f"\n  Gamma trend history: {len(gamma_history)} daily prices (from priceChange deltas)")

    # Load CSV snapshots (ground truth anchor points)
    csv_history = load_pm_history(slug)

    # Merge: CSV snapshots override Gamma estimates for their dates
    pm_history = gamma_history
    pm_history.update(csv_history)

    if csv_history:
        snapshot_dates = sorted(csv_history.keys())
        print(f"  CSV snapshots: {len(csv_history)} points ({snapshot_dates[0]} -> {snapshot_dates[-1]})")

    # Record today's snapshot (adds today to pm_history for filtering)
    record_pm_snapshot(slug)
    csv_history = load_pm_history(slug)
    pm_history.update(csv_history)

    # Filter agent predictions to only dates with real CSV snapshot data
    if csv_history and daily_data:
        snapshot_dates_set = set(csv_history.keys())
        daily_data = [d for d in daily_data if d["date"] in snapshot_dates_set]
        if daily_data:
            print(f"\n  Filtered to {len(daily_data)} days with CSV snapshot data")
        else:
            print("\n  No predictions on CSV snapshot dates. Showing empty.")

    snapshot_dates = sorted(pm_history.keys())
    if len(pm_history) >= 2:
        print(f"  Polymarket history: {len(pm_history)} data points ({snapshot_dates[0]} -> {snapshot_dates[-1]})")
    elif len(pm_history) == 1:
        print(f"  Polymarket history: 1 data point ({snapshot_dates[0]}). Run daily for more.")
    else:
        print("  No Polymarket history. Snapshot will be recorded each run.")

    # ── 5. Generate dashboard ───────────────────────────────────────
    print("\nStep 5/5: Generating dashboard HTML...")
    output_path = Path(args.output) if args.output else (work_dir / f"dashboard_{slug}.html")
    generate_dashboard(market, daily_data, pm_history, output_path, current_prob)
    print("\n" + "=" * 60)
    print("Dashboard ready! Open the HTML file in any browser.")


if __name__ == "__main__":
    main()
