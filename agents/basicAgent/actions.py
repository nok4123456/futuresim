"""Action handler methods for BasicAgent."""

from datetime import date
from typing import Any, Dict, List, Optional

from agents.utils.budget import BudgetTracker
from agents.utils.memory import ActiveMemory, StructuredMemory
from environment.interfaces import PredictionSubmission

from .tools import execute_news_search, optional_search_dates_from_parsed


class BasicActionHandlers:
    def _handle_query(
        self,
        messages,
        forecast_interface,
        response,
        parsed,
        budget: BudgetTracker,
        qid: str = None,
        reasoning=None,
        raw_stream: Optional[str] = None,
        tool_call: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Handle query action."""
        budget.consume_action()

        if parsed.code:
            extra_ctx = None
            if isinstance(self._memory, ActiveMemory):
                extra_ctx = {"mem_df": self._memory.get_mem_df()}
            with self._timer.track("df_query"):
                result, error = self._query_handler.execute(parsed.code, extra_context=extra_ctx)

            if error:
                feedback = f"QUERY ERROR: {error}"
            else:
                feedback = f"QUERY RESULT:\n{result}"
        else:
            feedback = f"ERROR: {parsed.error}"

        self._append_feedback_message(messages, budget, feedback, tool_call=tool_call, tool_name="query_df")

    def _handle_search(
        self,
        messages,
        forecast_interface,
        response,
        parsed,
        budget: BudgetTracker,
        qid: str = None,
        reasoning=None,
        raw_stream: Optional[str] = None,
        tool_call: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Handle search action. Captures top-3 result snippets for logging/memory."""
        budget.consume_action()

        if not self._search_handler.is_available:
            feedback = "SEARCH ERROR: Search is not available."
        elif parsed.query:
            min_date, max_date = optional_search_dates_from_parsed(parsed)
            with self._timer.track("search"):
                effect = execute_news_search(
                    parsed,
                    self._search_handler,
                    max_results=self.config.max_search_results,
                    search_type="hybrid",
                    min_date=min_date,
                    max_date=max_date,
                )
            feedback = effect.feedback

            # Capture top-3 search result snippets for daily evidence log
            raw_results = list(effect.raw_results) if effect.raw_results else []
            if raw_results and effect.successful_hit:
                _store_search_evidence(self, raw_results[:3], qid, parsed.query)
        else:
            feedback = "SEARCH ERROR: No query provided."

        self._append_feedback_message(messages, budget, feedback, tool_call=tool_call, tool_name="search_news")

    def _handle_memory_action(
        self,
        messages,
        forecast_interface,
        response,
        parsed,
        budget: BudgetTracker,
        reasoning=None,
        tool_call: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Handle memory tool calls (retrieve/add/update/delete). Follows search handler pattern."""
        budget.consume_action()
        tool_name = {
            "memory_retrieve": "memory_retrieve",
            "memory_new": "memory_new",
            "memory_add": "memory_new",
            "memory_update": "memory_update",
            "memory_delete": "memory_delete",
        }.get(parsed.action_type, "memory_retrieve")

        if not isinstance(self._memory, (StructuredMemory, ActiveMemory)):
            feedback = "MEMORY ERROR: Structured memory is not enabled."
        elif parsed.error:
            feedback = f"MEMORY ERROR: {parsed.error}"
        elif parsed.action_type == "memory_retrieve":
            entry = self._memory.retrieve(parsed.memory_entry_name)
            if entry is None:
                feedback = f"MEMORY ERROR: No entry with name '{parsed.memory_entry_name}'."
            else:
                feedback = f"MEMORY ENTRY:\n{entry}"
        elif parsed.action_type in ("memory_new", "memory_add"):
            data = parsed.memory_new_data
            try:
                entry_name = self._memory.add_entry(data["name"], data["description"], data["content"])
                feedback = f"MEMORY: Added entry [{entry_name}]. Total entries: {self._memory.entry_count}/{self._memory._max_entries if hasattr(self._memory, '_max_entries') else '?'}."
            except ValueError as exc:
                feedback = f"MEMORY ERROR: {exc}"
        elif parsed.action_type == "memory_update":
            ok = self._memory.update_entry(parsed.memory_entry_name, **parsed.memory_update_data)
            if ok:
                feedback = f"MEMORY: Updated [{parsed.memory_entry_name}]."
            else:
                feedback = f"MEMORY ERROR: No entry with name '{parsed.memory_entry_name}'."
        elif parsed.action_type == "memory_delete":
            ok = self._memory.delete_entry(parsed.memory_entry_name)
            if ok:
                feedback = f"MEMORY: Deleted [{parsed.memory_entry_name}]. Remaining: {self._memory.entry_count}."
            else:
                feedback = f"MEMORY ERROR: No entry with name '{parsed.memory_entry_name}'."
        else:
            feedback = f"MEMORY ERROR: Unknown memory action '{parsed.action_type}'."

        self._append_feedback_message(messages, budget, feedback, tool_call=tool_call, tool_name=tool_name)

    def _handle_mem_action(
        self,
        messages,
        forecast_interface,
        response,
        parsed,
        budget: BudgetTracker,
        reasoning=None,
        tool_call: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Handle mem_df tool calls (mem_add/update/delete) for ActiveMemory."""
        budget.consume_action()
        tool_name = {
            "mem_add": "mem_add",
            "mem_update": "mem_update",
            "mem_delete": "mem_delete",
        }.get(parsed.action_type, "mem_add")

        if not isinstance(self._memory, ActiveMemory):
            feedback = "MEM ERROR: Active memory is not enabled."
        elif parsed.error:
            feedback = f"MEM ERROR: {parsed.error}"
        elif parsed.action_type == "mem_add":
            data = parsed.mem_data
            self._memory.mem_add(
                qid=data["qid"], question=data.get("question", ""),
                memory=data["memory"], category=data.get("category", ""),
            )
            feedback = f"MEM: Added entry for Q{data['qid']}. Total: {self._memory.mem_count} entries."
        elif parsed.action_type == "mem_update":
            self._memory.mem_update(
                qid=parsed.mem_qid, memory=parsed.mem_data["memory"],
                category=parsed.mem_data.get("category"),
            )
            feedback = f"MEM: Updated entry for Q{parsed.mem_qid}."
        elif parsed.action_type == "mem_delete":
            ok = self._memory.mem_delete(parsed.mem_qid)
            if ok:
                feedback = f"MEM: Deleted Q{parsed.mem_qid}. Remaining: {self._memory.mem_count}."
            else:
                feedback = f"MEM ERROR: No entry for Q{parsed.mem_qid}."
        else:
            feedback = f"MEM ERROR: Unknown mem action '{parsed.action_type}'."

        self._append_feedback_message(messages, budget, feedback, tool_call=tool_call, tool_name=tool_name)

    def _handle_submit(
        self,
        messages,
        forecast_interface,
        response,
        parsed,
        budget: BudgetTracker,
        qid: str = None,
        reasoning=None,
        raw_stream: Optional[str] = None,
        tool_call: Optional[Dict[str, Any]] = None,
    ) -> List:
        """Handle submit action. Returns list of submitted forecasts."""
        submitted = []
        budget.consume_action()
        dropped_forecasts = 0

        # For logging: use provided qid, or infer from forecasts
        log_qid = qid
        if not log_qid and parsed.forecasts and len(parsed.forecasts) == 1:
            log_qid = parsed.forecasts[0]['qid']

        if parsed.forecasts:
            # Enforce single-qid submit: one <forecast ...> block per submit action.
            if len(parsed.forecasts) > 1:
                dropped_forecasts = len(parsed.forecasts) - 1
                parsed.forecasts = [parsed.forecasts[0]]

            for f in parsed.forecasts:
                try:
                    pred = PredictionSubmission(question_id=f['qid'], outcomes=f['outcomes'])
                    forecast_interface.submit_prediction(pred)
                    submitted.append(f)
                    outcomes_str = ", ".join(f"{k}: {v:.2f}" for k, v in f['outcomes'].items())
                    print(f"  [{self.agent_id}] Forecast {f['qid']}: {outcomes_str}")
                except Exception as e:
                    print(f"  [{self.agent_id}] Failed to submit {f['qid']}: {e}")

            if submitted:
                # Ensure later same-day df queries reflect newly submitted predictions.
                self._query_handler.invalidate_cache()

            # Include submitted qids in log metadata
            submitted_qids = [f['qid'] for f in submitted]
            if hasattr(self, '_day_qids'):
                self._day_qids.update(str(q) for q in submitted_qids)
            if submitted:
                sub = submitted[0]
                outcomes_str = ", ".join(f"{k}: {v:.2f}" for k, v in sub['outcomes'].items())
                title = self._query_handler.get_question_title(sub['qid'])
                title_str = f" ({title})" if title else ""
                feedback = f"Submitted forecast for qid={sub['qid']}{title_str}: {outcomes_str}."

                # Attach submission reasoning if provided by the model
                sub_reasoning = parsed.submit_reasoning
                if sub_reasoning:
                    feedback += f"\nReasoning: {sub_reasoning[:500]}"
                    # Store in daily evidence log
                    _store_submit_evidence(self, sub['qid'], sub['outcomes'], sub_reasoning)

                # Log counterfactual if provided
                counterfactual = parsed.submit_counterfactual
                if counterfactual:
                    feedback += f"\nCounterfactual: {counterfactual[:300]}"
                    _store_submit_counterfactual(self, sub['qid'], counterfactual)

                # Log evidence diversity if provided
                evidence_div = parsed.evidence_diversity
                if evidence_div is not None:
                    feedback += f"\nEvidence Diversity: {evidence_div} distinct source(s)"
                    _store_submit_evidence_diversity(self, sub['qid'], evidence_div)

                # Log base rate estimate if provided
                base_rate = parsed.base_rate_estimate
                if base_rate:
                    feedback += f"\nBase Rate: {base_rate[:300]}"
                    _store_submit_base_rate(self, sub['qid'], base_rate)

                # Log market sentiment score if provided
                sentiment = parsed.market_sentiment_score
                if sentiment is not None:
                    feedback += f"\nMarket Sentiment Score: {sentiment:+.2f}"
                    _store_submit_sentiment(self, sub['qid'], sentiment)

                if dropped_forecasts > 0:
                    feedback += f"\nIgnored {dropped_forecasts} extra forecast block(s); submit exactly one qid per action."
            else:
                feedback = "SUBMIT ERROR: No valid forecast submitted."
        else:
            # Parse error - still consumed action
            feedback = f"SUBMIT ERROR: {parsed.error}"

        self._append_feedback_message(messages, budget, feedback, tool_call=tool_call, tool_name="submit_forecasts")
        return submitted

    def _handle_invalid(
        self,
        messages,
        forecast_interface,
        response,
        parsed,
        budget: BudgetTracker,
        qid: str = None,
        reasoning=None,
        raw_stream: Optional[str] = None,
        tool_call: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Handle invalid/unknown action."""
        budget.consume_action()

        error_msg = parsed.error or "Call exactly one function tool."
        feedback = f"No valid action found. {error_msg}"
        tool_name = tool_call.get("name") if isinstance(tool_call, dict) else None
        self._append_feedback_message(messages, budget, feedback, tool_call=tool_call, tool_name=tool_name)


# ── Daily evidence helpers ───────────────────────────────────────────────

def _store_search_evidence(agent, results, qid, query):
    """Store top search result snippets in the agent's daily evidence log and memory."""
    if not results:
        return
    lines = ["Today's research:"]
    for r in results:
        title = getattr(r, 'title', '') or ''
        snippet = getattr(r, 'snippet', '') or ''
        source = getattr(r, 'source', '') or ''
        date_pub = getattr(r, 'date_publish', None)
        date_str = f" ({date_pub})" if date_pub else ""
        url_str = f" [{source}]" if source else ""
        lines.append(f"- {title}{url_str}{date_str}: {snippet[:200]}")
    evidence_text = "\n".join(lines)

    # Store in daily evidence list (for logging at end of day)
    if not hasattr(agent, '_day_evidence'):
        agent._day_evidence = []
    agent._day_evidence.append({
        "type": "search",
        "query": query,
        "qid": qid,
        "evidence": evidence_text,
        "results": [{"title": getattr(r, 'title', ''), "snippet": getattr(r, 'snippet', ''),
                       "source": getattr(r, 'source', ''), "url": getattr(r, 'url', ''),
                       "date_publish": str(getattr(r, 'date_publish', ''))}
                      for r in results],
    })

    # Store in memory under "latest_evidence" if ActiveMemory is available
    memory = getattr(agent, '_memory', None)
    if isinstance(memory, ActiveMemory) and qid:
        try:
            memory.mem_add(
                qid=qid, question="",
                memory=f"[Search query: {query}]\n{evidence_text}",
                category="evidence",
            )
        except Exception:
            pass  # Memory storage is best-effort


def _store_submit_evidence(agent, qid, outcomes, reasoning_text):
    """Store structured prediction reasoning in the daily evidence log."""
    if not reasoning_text:
        return
    outcomes_str = ", ".join(f"{k}={v:.1%}" for k, v in (outcomes or {}).items())

    if not hasattr(agent, '_day_evidence'):
        agent._day_evidence = []
    agent._day_evidence.append({
        "type": "prediction",
        "qid": qid,
        "outcomes": outcomes_str,
        "reasoning": reasoning_text,
    })


def _store_submit_sentiment(agent, qid, sentiment_score):
    """Store market sentiment score in the daily evidence log."""
    if not hasattr(agent, '_day_evidence'):
        agent._day_evidence = []
    agent._day_evidence.append({
        "type": "sentiment",
        "qid": qid,
        "market_sentiment_score": sentiment_score,
    })


def _store_submit_counterfactual(agent, qid, counterfactual):
    """Store counterfactual reasoning in the daily evidence log."""
    if not hasattr(agent, '_day_evidence'):
        agent._day_evidence = []
    agent._day_evidence.append({
        "type": "counterfactual",
        "qid": qid,
        "counterfactual": counterfactual,
    })


def _store_submit_evidence_diversity(agent, qid, diversity_count):
    """Store evidence diversity count in the daily evidence log."""
    if not hasattr(agent, '_day_evidence'):
        agent._day_evidence = []
    agent._day_evidence.append({
        "type": "evidence_diversity",
        "qid": qid,
        "evidence_diversity": diversity_count,
    })


def _store_submit_base_rate(agent, qid, base_rate):
    """Store base rate estimate in the daily evidence log."""
    if not hasattr(agent, '_day_evidence'):
        agent._day_evidence = []
    agent._day_evidence.append({
        "type": "base_rate",
        "qid": qid,
        "base_rate_estimate": base_rate,
    })
