"""
Polymarket market-data tool using the public Gamma API (no key required).

Provides real-time market probabilities, outcome prices, and market URLs
for forecasting agents to compare their predictions against crowd odds.
"""

import json
import hashlib
from datetime import date
from typing import List, Optional

try:
    import requests
except ImportError:
    raise ImportError("requests module not found. Install with: pip install requests")

from .base import BaseSearchTool, SearchResult, Article

GAMMA_API = "https://gamma-api.polymarket.com"


class PolymarketTool(BaseSearchTool):
    """Query Polymarket's Gamma API for real-time market probabilities.

    Usage:
        tool = PolymarketTool()
        results = tool.search("will-google-have-the-best-ai-model-at-the-end-of-may-2026")
        # or keyword search:
        results = tool.search("Fed rate hike 2026")
    """

    def __init__(self):
        self._available = True  # No API key needed — public API

    # ── BaseSearchTool interface ──────────────────────────────────────

    @property
    def is_available(self) -> bool:
        return self._available

    def search(
        self,
        query: str,
        max_results: int = 10,
        max_date: Optional[date] = None,
        search_type: str = "hybrid",
        min_date: Optional[date] = None,
        current_date: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search Polymarket markets by slug or keyword.

        If *query* looks like a slug (no spaces, contains hyphens), fetch the
        specific market by slug.  Otherwise perform a keyword text search
        across the Gamma /markets endpoint.

        Returns one SearchResult per outcome so the agent sees full market
        structure (Yes/No prices, multi-outcome odds, etc.)
        """
        # Detect slug vs keyword: slugs are hyphenated, no spaces
        is_slug = " " not in query and "-" in query and len(query) > 3

        if is_slug:
            markets = self._fetch_by_slug(query)
        else:
            markets = self._search_markets(query, max_results)

        if not markets:
            return []

        results: List[SearchResult] = []
        for m in markets[:max_results]:
            title = m.get("question") or m.get("title", "Unknown")
            slug = m.get("slug", "")
            volume = float(m.get("volume", 0) or 0)
            liquidity = float(m.get("liquidity", 0) or 0)
            end_date = (m.get("endDateIso") or m.get("endDate") or "")[:10]

            # Parse outcomes & prices (may be JSON strings)
            outcomes_raw = m.get("outcomes") or "[]"
            prices_raw = m.get("outcomePrices") or "[]"
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

            # Build a rich snippet with all outcomes + market metadata
            parts = [f"Market: {title}"]
            if volume:
                parts.append(f"Volume: ${volume:,.0f}")
            if liquidity:
                parts.append(f"Liquidity: ${liquidity:,.0f}")
            if end_date:
                parts.append(f"Ends: {end_date}")

            if outcomes and prices and len(outcomes) == len(prices):
                parts.append("Current odds:")
                for i, (name, price) in enumerate(zip(outcomes, prices)):
                    parts.append(f"  {name}: {price:.4f} ({price*100:.1f}%)")
                # The first outcome price is the "Yes" for binary markets
                yes_price = prices[0] if prices else None
            else:
                yes_price = None
                if outcomes:
                    parts.append(f"Outcomes: {', '.join(outcomes)}")

            snippet = "\n".join(parts)
            market_url = f"https://polymarket.com/event/{slug}" if slug else ""

            # Use one SearchResult per market (with all outcomes in snippet)
            market_id = hashlib.sha1(
                (m.get("id", "") + title).encode()
            ).hexdigest()[:16]

            results.append(SearchResult(
                article_id=market_id,
                title=title,
                source="Polymarket",
                date=date.today(),
                date_publish=date.today(),
                snippet=snippet,
                score=float(yes_price) if yes_price is not None else 0.0,
                url=market_url,
            ))

        return results

    def get_article(self, article_id: str) -> Optional[Article]:
        """Polymarket markets don't have full article bodies."""
        return None

    def count_articles(self, min_date=None, max_date=None) -> Optional[int]:
        """Polymarket markets are not dated — count is not supported."""
        return None

    # ── Gamma API helpers ─────────────────────────────────────────────

    @staticmethod
    def _fetch_by_slug(slug: str) -> List[dict]:
        """Fetch a single market by its slug."""
        try:
            resp = requests.get(
                f"{GAMMA_API}/markets",
                params={"slug": slug, "limit": 1},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else [data] if isinstance(data, dict) else []
        except requests.RequestException as e:
            print(f"  [Polymarket] Slug lookup error for '{slug}': {e}")
            return []
        except Exception as e:
            print(f"  [Polymarket] Unexpected error for slug '{slug}': {e}")
            return []

    @staticmethod
    def _search_markets(keyword: str, limit: int = 10) -> List[dict]:
        """Keyword search across Polymarket markets."""
        try:
            resp = requests.get(
                f"{GAMMA_API}/markets",
                params={"search": keyword, "limit": min(limit, 50), "closed": "false"},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else []
        except requests.RequestException as e:
            print(f"  [Polymarket] Search error for '{keyword}': {e}")
            return []
        except Exception as e:
            print(f"  [Polymarket] Unexpected search error: {e}")
            return []
