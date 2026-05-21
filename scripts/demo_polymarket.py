#!/usr/bin/env python
"""
Polymarket + Futuresim Demo Pipeline
=====================================
1. Fetch a Polymarket question (by slug or keyword search).
2. Convert it into a custom JSONL question file for futuresim.
3. Run futuresim on that question over a date range.
4. Extract the agent's daily probability predictions.
5. Compare with Polymarket's current (or recorded) probability.
6. Output a comparison CSV.

Usage:
    # By slug
    python scripts/demo_polymarket.py --slug will-ukraine-join-nato-before-2030

    # By keyword search (picks first match)
    python scripts/demo_polymarket.py --search "Ukraine NATO"

    # With custom date range and model
    python scripts/demo_polymarket.py --slug some-market \\
        --start_date 2026-01-01 --end_date 2026-01-07 \\
        --model deepseek-v4-flash --matching deepseek

    # Record today's Polymarket snapshot for historical tracking
    python scripts/demo_polymarket.py --slug some-market --record-snapshot

Dependencies: requests (already in pyproject.toml)
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
GAMMA_API = "https://gamma-api.polymarket.com"
HISTORY_CSV = REPO_ROOT / "polymarket_history.csv"
RUN_FORECAST_SCRIPT = REPO_ROOT / "scripts" / "run_forecast_sim.py"


# ---------------------------------------------------------------------------
# Polymarket API client
# ---------------------------------------------------------------------------

class PolymarketClient:
    """Thin wrapper around the Polymarket Gamma Markets API (no key required)."""

    @staticmethod
    def _parse_outcomes(market: dict) -> Tuple[List[str], List[float]]:
        """Parse outcomes and outcomePrices from a market dict."""
        outcomes_raw = market.get("outcomes") or "[]"
        prices_raw = market.get("outcomePrices") or "[]"
        if isinstance(outcomes_raw, str):
            outcomes_raw = json.loads(outcomes_raw)
        if isinstance(prices_raw, str):
            prices_raw = json.loads(prices_raw)
        prices_float = [float(p) for p in prices_raw]
        return list(outcomes_raw), prices_float

    @staticmethod
    def search(query: str, limit: int = 10) -> List[dict]:
        """Search for markets by keyword. Returns list of market summary dicts
        (does NOT include outcomes/prices — use fetch_detail for those)."""
        url = f"{GAMMA_API}/markets"
        params = {"limit": limit, "closed": "false", "keyword": query}
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            print(f"[Polymarket] Search error: {e}")
            return []

    @staticmethod
    def lookup_by_slug(slug: str) -> Optional[dict]:
        """Fetch a market summary by slug (includes outcomes/prices)."""
        url = f"{GAMMA_API}/markets"
        params = {"limit": 1, "slug": slug}
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            return data[0] if data else None
        except requests.RequestException as e:
            print(f"[Polymarket] Lookup error for '{slug}': {e}")
            return None

    @staticmethod
    def fetch_detail(market_id: str) -> Optional[dict]:
        """Fetch full market detail by numeric ID (includes outcomes/prices)."""
        url = f"{GAMMA_API}/markets/{market_id}"
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            print(f"[Polymarket] Detail error for '{market_id}': {e}")
            return None

    @staticmethod
    def get_current_probability(market: dict) -> Optional[float]:
        """Extract the Yes/primary-outcome probability from a market dict."""
        outcomes, prices = PolymarketClient._parse_outcomes(market)
        if not prices:
            return None
        # For binary markets, "Yes" is index 0; for multi-outcome, first outcome
        if outcomes and outcomes[0].lower() == "yes":
            return prices[0]
        # Otherwise, return the highest-price outcome
        return max(prices) if prices else None

    @staticmethod
    def pick_market(slug: Optional[str] = None,
                    search: Optional[str] = None) -> Optional[dict]:
        """Resolve a market by slug or search keyword.
        Returns a full-detail market dict (with outcomes/prices)."""
        if slug:
            summary = PolymarketClient.lookup_by_slug(slug)
            if not summary:
                print(f"[Polymarket] Market not found: {slug}")
                return None
            market_id = summary.get("id", "")
            detail = PolymarketClient.fetch_detail(market_id) if market_id else None
            market = detail or summary
            print(f"[Polymarket] Loaded: {market.get('question', slug)}")
            return market

        if search:
            results = PolymarketClient.search(search, limit=10)
            if not results:
                print(f"[Polymarket] No results for '{search}'.")
                return None
            summary = results[0]
            market_id = summary.get("id", "")
            detail = PolymarketClient.fetch_detail(market_id) if market_id else None
            market = detail or summary
            print(f"[Polymarket] Matched ({len(results)} results): {market.get('question', slug or '?')}")
            return market

        return None


# ---------------------------------------------------------------------------
# Custom question builder (JSONL for futuresim custom dataset)
# ---------------------------------------------------------------------------

def build_custom_question_jsonl(
    market: dict,
    output_path: Path,
    qid: Optional[str] = None,
    resolution_date_override: Optional[str] = None,
) -> dict:
    """
    Convert a Polymarket market dict into a single-question JSONL file
    compatible with `--dataset custom --dataset_path <path>`.

    Returns the question dict for reference.
    """
    title = market.get("question", market.get("title", "Untitled"))
    # Parse endDate (ISO) or use override
    end_date_raw = market.get("endDate", market.get("endDateIso", ""))
    if resolution_date_override:
        res_date = resolution_date_override
    elif end_date_raw:
        res_date = end_date_raw[:10]  # "2026-06-15T00:00:00Z" -> "2026-06-15"
    else:
        res_date = (date.today() + timedelta(days=30)).isoformat()

    # Parse outcomes
    outcomes, prices = PolymarketClient._parse_outcomes(market)
    if len(outcomes) == 2 and outcomes[0].lower() == "yes":
        answer_type = "binary"
        options = ["Yes", "No"]
    elif len(outcomes) > 2:
        answer_type = "multichoice"
        options = list(outcomes)
    else:
        answer_type = "binary"
        options = ["Yes", "No"]

    # Build a description/background from market data
    description = market.get("description", "") or ""
    # Truncate very long descriptions
    if len(description) > 2000:
        description = description[:2000] + "..."

    question = {
        "qid": qid or _slug_to_qid(market.get("slug", "pm")),
        "title": title,
        "resolution_date": res_date,
        "ground_truth_answer": outcomes[0] if outcomes else "",
        "background": description,
        "answer_type": answer_type,
        "options": json.dumps(options) if options else None,
        "resolution_criteria": market.get("resolutionSource", "") or "",
        "source": "polymarket",
        "source_split": "demo",
        "prompt": (
            f"This is a Polymarket forecasting question. "
            f"Current market probability: {prices[0] if prices else 'N/A'}. "
            f"Volume: ${float(market.get('volume', 0)):,.0f}. "
            f"Research the topic and submit your probability estimate."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(question, f, ensure_ascii=False)
        f.write("\n")

    print(f"[Builder] Custom question written: {output_path}")
    return question


def _slug_to_qid(slug: str) -> str:
    """Create a stable QID from a market slug."""
    import hashlib
    return "PM" + hashlib.sha1(slug.encode()).hexdigest()[:6].upper()


# ---------------------------------------------------------------------------
# Futuresim runner
# ---------------------------------------------------------------------------

def run_futuresim(
    dataset_path: Path,
    start_date: str,
    end_date: str,
    *,
    provider: str = "deepseek",
    model: str = "deepseek-v4-flash",
    matching: str = "exact",
    sim_name: str = "polymarket_demo",
    max_actions: int = 5,
    temperature: float = 0.7,
    no_inference: bool = False,
    force_submit: bool = False,
    run_timeout: int = 1800,  # 30 min default
) -> Optional[str]:
    """
    Run futuresim via subprocess and return the output directory path.
    Parses stdout for "Output directory: <path>".
    """
    cmd = [
        sys.executable,
        str(RUN_FORECAST_SCRIPT),
        "--provider", provider,
        "--deepseek_model", model,
        "--matching", matching,
        "--dataset", "custom",
        "--dataset_path", str(dataset_path),
        "--start_date", start_date,
        "--end_date", end_date,
        "--sim_name", sim_name,
        "--max_actions", str(max_actions),
        "--temperature", str(temperature),
    ]
    if no_inference:
        cmd.append("--no_inference")
    if force_submit:
        cmd.extend([
            "--force_submit_threshold_tokens", "0",
            "--submit_reserve_tokens", "0",
            "--max_total_tokens", "32000",
        ])

    print(f"\n[Runner] Launching futuresim...")
    print(f"[Runner] {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=run_timeout,
        )
    except subprocess.TimeoutExpired:
        print("[Runner] Error: Simulation timed out after 10 minutes.")
        return None
    except FileNotFoundError:
        print("[Runner] Error: Python executable or run_forecast_sim.py not found.")
        return None

    if result.returncode != 0:
        print(f"[Runner] Simulation failed (exit code {result.returncode}).")
        print(f"[Runner] STDERR:\n{result.stderr[-1000:]}")
        return None

    # Parse output directory from stdout
    output_dir = None
    for line in result.stdout.splitlines():
        line = line.strip()
        if "Output directory:" in line:
            output_dir = line.split("Output directory:", 1)[-1].strip()
        # Also capture the last few lines for the user
    print(result.stdout[-500:])

    if not output_dir or not os.path.isdir(output_dir):
        print("[Runner] Could not determine output directory from simulation output.")
        return None

    print(f"[Runner] Output: {output_dir}")
    return output_dir


# ---------------------------------------------------------------------------
# Prediction extractor
# ---------------------------------------------------------------------------

def extract_agent_predictions(output_dir: str) -> Dict[str, float]:
    """
    Parse actions.jsonl in the output directory and return
    a dict mapping sim_date -> agent's probability (0-1) for the
    primary/first outcome.

    For binary markets, this is the "Yes" probability.
    For multichoice, it's the highest-probability outcome.
    """
    actions_path = os.path.join(output_dir, "actions.jsonl")
    if not os.path.exists(actions_path):
        print("[Extractor] No actions.jsonl found.")
        return {}

    predictions: Dict[str, float] = {}

    with open(actions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if entry.get("type") != "prediction":
                continue

            sim_date = entry.get("sim_date", "")
            outcomes = entry.get("outcomes", {})

            if not sim_date or not outcomes:
                continue

            # For binary: pick "Yes". For multi-choice: pick the max-prob outcome.
            prob = outcomes.get("Yes", outcomes.get("yes", None))
            if prob is None:
                prob = max(outcomes.values()) if outcomes else 0.0

            predictions[sim_date] = float(prob)

    return predictions


# ---------------------------------------------------------------------------
# Polymarket history
# ---------------------------------------------------------------------------

def record_pm_snapshot(slug: str) -> Optional[float]:
    """Fetch current probability and append to polymarket_history.csv."""
    market = PolymarketClient.lookup_by_slug(slug)
    if not market:
        return None
    prob = PolymarketClient.get_current_probability(market)
    if prob is None:
        return None

    today_str = date.today().isoformat()
    write_header = not HISTORY_CSV.exists()

    with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["date", "slug", "probability"])
        writer.writerow([today_str, slug, prob])

    print(f"[Snapshot] Recorded {slug} = {prob:.4f} for {today_str}")
    return prob


def load_pm_history(slug: str) -> Dict[str, float]:
    """Load historical probabilities for a given slug from the CSV."""
    if not HISTORY_CSV.exists():
        return {}

    history: Dict[str, float] = {}
    with open(HISTORY_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("slug", "").strip() == slug:
                date_str = row.get("date", "").strip()
                prob_str = row.get("probability", "").strip()
                if date_str and prob_str:
                    try:
                        history[date_str] = float(prob_str)
                    except ValueError:
                        continue
    return history


def get_pm_probability_for_date(
    slug: str, target_date: str, current_prob: float, history: Dict[str, float]
) -> Optional[float]:
    """
    Get the Polymarket probability for a specific date.
    Uses historical snapshot if available, otherwise falls back
    to current probability.
    """
    if target_date in history:
        return history[target_date]
    # Fallback to current (best effort)
    return current_prob


# ---------------------------------------------------------------------------
# Comparison output
# ---------------------------------------------------------------------------

def write_comparison_csv(
    predictions: Dict[str, float],
    slug: str,
    current_pm_prob: float,
    history: Dict[str, float],
    output_path: Path,
):
    """Write demo_comparison.csv with date, agent_prob, pm_prob, diff."""
    # Sort by date
    rows = []
    for d in sorted(predictions):
        agent_p = predictions[d]
        pm_p = get_pm_probability_for_date(slug, d, current_pm_prob, history)
        diff = agent_p - (pm_p or 0.0)
        rows.append([d, f"{agent_p:.4f}", f"{pm_p:.4f}" if pm_p else "N/A", f"{diff:+.4f}"])

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "agent_prob", "pm_prob", "diff"])
        writer.writerows(rows)

    print(f"\n[Output] Comparison written: {output_path}")
    # Print summary table
    print(f"  {'Date':<12} {'Agent':>8} {'Polymarket':>12} {'Diff':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*12} {'-'*8}")
    for row in rows:
        print(f"  {row[0]:<12} {row[1]:>8} {row[2]:>12} {row[3]:>8}")

    # Summary stats
    if rows:
        diffs = [abs(float(r[3])) for r in rows if r[3] != "N/A"]
        if diffs:
            print(f"\n  Mean absolute diff: {sum(diffs)/len(diffs):.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Polymarket + Futuresim Demo Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python scripts/demo_polymarket.py --slug will-ukraine-join-nato-before-2030
              python scripts/demo_polymarket.py --search "Ukraine NATO" --start_date 2026-01-01
              python scripts/demo_polymarket.py --slug some-market --record-snapshot
              python scripts/demo_polymarket.py --slug some-market --no-inference
        """),
    )
    # Market selection (one required)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--slug", help="Polymarket market slug (e.g., will-ukraine-join-nato-before-2030)")
    group.add_argument("--search", help="Keyword search to find a Polymarket market")

    # Date range
    parser.add_argument("--start_date", default=None,
                        help="Simulation start date YYYY-MM-DD (default: today - 7 days)")
    parser.add_argument("--end_date", default=None,
                        help="Simulation end date YYYY-MM-DD (default: market resolution + 7 days)")
    parser.add_argument("--resolution_date", default=None,
                        help="Override question resolution date YYYY-MM-DD (default: market endDate)")

    # Futuresim settings
    parser.add_argument("--provider", default="deepseek", choices=["deepseek", "openrouter"],
                        help="Inference provider (default: deepseek)")
    parser.add_argument("--model", default="deepseek-v4-flash",
                        help="Model ID for agent (default: deepseek-v4-flash for speed)")
    parser.add_argument("--matching", default="exact", choices=["exact", "deepseek", "openrouter"],
                        help="Answer matching mode (default: exact for speed)")
    parser.add_argument("--sim_name", default="polymarket_demo",
                        help="Simulation name for output directory")
    parser.add_argument("--max_actions", type=int, default=5,
                        help="Max actions per day (default: 5)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")
    parser.add_argument("--no_inference", action="store_true",
                        help="Run without LLM inference (dry-run test)")
    parser.add_argument("--force_submit", action="store_true",
                        help="Force earlier submissions (sets force_submit_threshold_tokens=0)")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Max runtime for futuresim subprocess in seconds (default: 1800 = 30 min)")

    # Output
    parser.add_argument("--output_dir", default=None,
                        help="Directory for output files (default: logs/current_sim/polymarket_demo_<ts>)")
    parser.add_argument("--record_snapshot", action="store_true",
                        help="Record today's Polymarket probability to history CSV")
    parser.add_argument("--comparison_csv", default=None,
                        help="Path for comparison CSV (default: <output_dir>/demo_comparison.csv)")

    args = parser.parse_args()

    # ── 1. Fetch Polymarket market ──────────────────────────────────
    print("=" * 60)
    print("Step 1/6: Fetching Polymarket market...")
    market = PolymarketClient.pick_market(slug=args.slug, search=args.search)
    if not market:
        sys.exit(1)

    slug = market.get("slug", args.slug or "unknown")
    title = market.get("question", market.get("title", "Unknown"))
    current_prob = PolymarketClient.get_current_probability(market)
    print(f"  Title: {title}")
    print(f"  Current probability: {current_prob}")
    print(f"  End date: {market.get('endDate', market.get('endDateIso', 'N/A'))}")

    # ── 2. Build custom question JSONL ──────────────────────────────
    print("\nStep 2/6: Building custom question...")
    demo_dir = Path(args.output_dir) if args.output_dir else (
        REPO_ROOT / "logs" / "current_sim" / f"{args.sim_name}_{date.today().isoformat()}"
    )
    demo_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = demo_dir / "question.jsonl"

    q = build_custom_question_jsonl(
        market,
        jsonl_path,
        resolution_date_override=args.resolution_date,
    )

    # ── 3. Run futuresim ────────────────────────────────────────────
    today = date.today()
    start_date = args.start_date or (today - timedelta(days=7)).isoformat()

    # Default end_date: market resolution date + 7 days (so agent has time to
    # submit before resolution). Override with --end_date if provided.
    if args.end_date:
        end_date = args.end_date
    else:
        res_date_str = q.get("resolution_date", "")
        if res_date_str:
            try:
                res_dt = date.fromisoformat(res_date_str)
                end_date = (res_dt + timedelta(days=7)).isoformat()
            except ValueError:
                end_date = today.isoformat()
        else:
            end_date = today.isoformat()

    print(f"\nStep 3/6: Running futuresim ({start_date} → {end_date})...")
    output_dir = run_futuresim(
        jsonl_path,
        start_date=start_date,
        end_date=end_date,
        provider=args.provider,
        model=args.model,
        matching=args.matching,
        sim_name=args.sim_name,
        max_actions=args.max_actions,
        temperature=args.temperature,
        no_inference=args.no_inference,
        force_submit=args.force_submit,
        run_timeout=args.timeout,
    )

    if not output_dir:
        print("\n[Demo] Simulation failed. Check the error output above.")
        sys.exit(1)

    # ── 4. Extract predictions ──────────────────────────────────────
    print("\nStep 4/6: Extracting agent predictions...")
    predictions = extract_agent_predictions(output_dir)
    if not predictions:
        print("[Demo] No predictions found. Agent did not submit any forecasts.")
    else:
        for d, p in sorted(predictions.items()):
            print(f"  {d}: {p:.4f}")

    # ── 5. Polymarket comparison ────────────────────────────────────
    print("\nStep 5/6: Polymarket comparison...")
    history = load_pm_history(slug)
    if args.record_snapshot:
        record_pm_snapshot(slug)
        # Reload history after recording
        history = load_pm_history(slug)

    if history:
        print(f"  Loaded {len(history)} historical snapshots for {slug}")
    else:
        print(f"  No historical snapshots. Using current probability ({current_prob}) for all days.")
        print(f"  Tip: run with --record-snapshot daily to build history.")

    # ── 6. Write comparison CSV ─────────────────────────────────────
    print("\nStep 6/6: Writing comparison...")
    comparison_path = Path(args.comparison_csv) if args.comparison_csv else (demo_dir / "demo_comparison.csv")
    write_comparison_csv(
        predictions,
        slug,
        current_pm_prob=current_prob or 0.0,
        history=history,
        output_path=comparison_path,
    )

    print("\n" + "=" * 60)
    print("Demo complete!")
    print(f"  Question: {title}")
    print(f"  Output dir: {output_dir}")
    print(f"  Comparison: {comparison_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
