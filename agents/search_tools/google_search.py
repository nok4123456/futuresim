"""
Google News search tool using the Serper.dev API.

Provides real-time web search without requiring a local LanceDB corpus.
Set SERPER_API_KEY in your environment or .env file to enable.
"""

import os
import hashlib
from datetime import date
from typing import List, Optional

try:
    import requests
except ImportError:
    raise ImportError("requests module not found. Install with: pip install requests")

from .base import BaseSearchTool, SearchResult, Article


class GoogleNewsSearchTool(BaseSearchTool):
    """Search Google News via Serper.dev API for real-time forecasting evidence.

    Environment variable:
        SERPER_API_KEY: Your Serper.dev API key (required)

    Usage:
        tool = GoogleNewsSearchTool()
        results = tool.search("Fed rate cut 2026", max_results=5)
    """

    API_URL = "https://google.serper.dev/news"

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.environ.get("SERPER_API_KEY", "")
        self._available = bool(self._api_key)

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
        """Search Google News via Serper.dev.

        Note: max_date / min_date / search_type are accepted for interface
        compatibility but Serper.dev does not support date-range filtering
        natively. Date filters are best-effort via the query string.

        current_date: ISO date string of the simulation date (for logging / context).
        """
        del search_type  # not applicable — Serper always uses keyword match

        # Parse simulation date for accurate relative date parsing
        reference_date: Optional[date] = None
        if current_date:
            try:
                reference_date = date.fromisoformat(current_date)
            except (ValueError, TypeError):
                pass

        if not self._available:
            return []

        payload: dict = {"q": query, "num": min(max_results, 100)}
        headers = {
            "X-API-KEY": self._api_key,
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                self.API_URL,
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                print(f"  [GoogleSearch] Rate limited by Serper.dev. Check quota.")
            else:
                print(f"  [GoogleSearch] HTTP error: {e}")
            return []
        except requests.exceptions.RequestException as e:
            print(f"  [GoogleSearch] Network error: {e}")
            return []
        except Exception as e:
            print(f"  [GoogleSearch] Unexpected error: {e}")
            return []

        results: List[SearchResult] = []
        for item in data.get("news", [])[:max_results]:
            pub_date = self._parse_serper_date(item.get("date", ""), reference_date=reference_date)
            # Apply best-effort date filters
            if min_date and pub_date and pub_date < min_date:
                continue
            if max_date and pub_date and pub_date > max_date:
                continue

            article_id = hashlib.sha1(
                (item.get("link", "") + item.get("title", "")).encode()
            ).hexdigest()[:16]

            results.append(SearchResult(
                article_id=article_id,
                title=item.get("title", ""),
                source=item.get("source", ""),
                date=pub_date,
                date_publish=pub_date,
                snippet=item.get("snippet", ""),
                score=1.0,
                url=item.get("link", ""),
            ))

        return results

    def get_article(self, article_id: str) -> Optional[Article]:
        """Serper.dev only returns snippets, not full articles."""
        return None

    @staticmethod
    def _parse_serper_date(date_str: str, reference_date: Optional[date] = None) -> Optional[date]:
        """Parse Serper.dev relative dates like '2 days ago' or absolute dates."""
        if not date_str:
            return None
        # Serper returns relative strings like "2 days ago", "3 hours ago", "1 week ago"
        import re
        from datetime import date as date_type, timedelta

        today = reference_date if reference_date is not None else date_type.today()
        date_str_lower = date_str.strip().lower()

        # "X days ago"
        m = re.match(r"(\d+)\s+day[s]?\s+ago", date_str_lower)
        if m:
            return today - timedelta(days=int(m.group(1)))

        # "X hours ago" or "X hour ago" → today
        if re.match(r"\d+\s+hour[s]?\s+ago", date_str_lower):
            return today

        # "X minutes ago" → today
        if re.match(r"\d+\s+minute[s]?\s+ago", date_str_lower):
            return today

        # "X weeks ago"
        m = re.match(r"(\d+)\s+week[s]?\s+ago", date_str_lower)
        if m:
            return today - timedelta(weeks=int(m.group(1)))

        # "X months ago"
        m = re.match(r"(\d+)\s+month[s]?\s+ago", date_str_lower)
        if m:
            return today - timedelta(days=int(m.group(1)) * 30)

        return None


def create_search_tool(search_db: str = "", embedding_model=None,
                       search_tool_type: str = "") -> Optional[BaseSearchTool]:
    """Factory: return the appropriate search tool based on configuration.

    Resolution order:
    1. Environment variable FSIM_SEARCH_TOOL ("google" / "polymarket" / "lancedb")
    2. Explicit search_tool_type argument
    3. Default: LanceDB if search_db is set, otherwise None
    """
    env_tool = os.environ.get("FSIM_SEARCH_TOOL", "").strip().lower()
    effective = env_tool or search_tool_type.strip().lower()

    if effective == "google":
        print("  Search tool: Google News (Serper.dev API)")
        tool = GoogleNewsSearchTool()
        if not tool.is_available:
            print("  Warning: SERPER_API_KEY not set — Google search disabled.")
            return None
        return tool

    if effective == "polymarket":
        from agents.search_tools.polymarket import PolymarketTool
        print("  Search tool: Polymarket (Gamma API)")
        tool = PolymarketTool()
        return tool

    # Default: LanceDB
    if search_db:
        from agents.search_tools.lancedb import LanceDBSearchTool
        print(f"  Search tool: LanceDB")
        tool = LanceDBSearchTool(search_db, embedding_model=embedding_model)
        if tool.is_available:
            print(f"  LanceDB connected: {search_db}")
        else:
            print(f"  Warning: LanceDB not available at {search_db}")
            return None
        return tool

    return None
