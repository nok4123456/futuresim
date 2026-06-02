"""Daily prompt construction helpers for BasicAgent."""

import re
from datetime import date, timedelta
from typing import Dict, Optional

from agents.utils.memory import ActiveMemory, StructuredMemory


class BasicPromptBuilder:
    def _build_prompt_seed_budget_block(
        self,
        *,
        warmup: bool = False,
        max_actions_override: Optional[int] = None,
        leading_newline: bool = False,
        trailing_newlines: int = 2,
    ) -> str:
        """Render the start-of-prompt budget block with a prompt-seed clarification."""
        status = self._build_start_budget_status(
            warmup=warmup,
            max_actions_override=max_actions_override,
        )
        if not status:
            return ""

        prefix = "\n" if leading_newline else ""
        suffix = "\n" * max(trailing_newlines, 0)
        return (
            f"{prefix}Budget at start:\n"
            f"{status}\n"
            "Note: current message tokens are not accounted for yet."
            f"{suffix}"
        )

    @staticmethod
    def _normalize_prompt_heading_spacing(prompt: str) -> str:
        """Ensure `##` section headings are separated by two blank lines."""
        if not prompt:
            return prompt
        return re.sub(r"\n{1,}(?=## )", "\n\n\n", prompt)

    def _search_results_description(self) -> str:
        chunk_tokens = self._search_handler.chunk_tokens
        tool_type = self._search_handler.search_tool_type
        if tool_type == "polymarket":
            return (
                f"Returns up to {self.config.max_search_results} Polymarket market results "
                f"(title, current odds for each outcome, volume, liquidity, end date, and URL). "
                f"Use market slugs (hyphenated names) or keywords to find matching prediction markets."
            )
        return (
            f"Returns up to {self.config.max_search_results} news article results "
            f"(title, source, date, snippet, and URL for each). "
            f"Always filter by date using from_date/to_date "
            f"to get recent articles from the last 7 days."
        )

    def _get_timegap_days(self) -> int:
        return max(1, int(getattr(self.config, "timegap_days", 1) or 1))

    def _get_last_active_date(self, current_date: date) -> Optional[date]:
        fi = getattr(self, "_forecast_interface", None)
        last_active = getattr(fi, "last_active_date", None) if fi is not None else None
        if last_active:
            return last_active
        return None

    def _get_next_active_date(self, current_date: date) -> Optional[date]:
        fi = getattr(self, "_forecast_interface", None)
        if fi is not None and hasattr(fi, "next_active_date"):
            next_active = getattr(fi, "next_active_date")
            return next_active
        return current_date + timedelta(days=self._get_timegap_days())

    def _build_cadence_section(self, current_date: date) -> str:
        last_active = self._get_last_active_date(current_date)
        next_active = self._get_next_active_date(current_date)
        timegap_days = self._get_timegap_days()
        day_unit = "day" if timegap_days == 1 else "days"
        next_text = (
            f"Next scheduled update: {next_active}."
            if next_active
            else "No later updates are scheduled."
        )
        last_text = (
            f"Last update: {last_active}. "
            if last_active
            else "This is your first update. "
        )
        articles_text = ""
        if last_active:
            if self._search_handler.is_available:
                count = self._search_handler.count_articles(
                    min_date=last_active,
                    max_date=current_date - timedelta(days=self.config.search_cutoff_days),
                )
                if count is not None:
                    articles_text = f"{count:,} new articles have been published since your last update and you can access them using the search tool. "
            if not articles_text:
                articles_text = "New articles have been published since your last update and you can access them using the search tool. "
        tomorrow = current_date + timedelta(days=1)
        imminent_qids = []
        fi = getattr(self, "_forecast_interface", None)
        if fi is not None and hasattr(fi, "questions"):
            imminent_qids = [
                qid for qid, q in (fi.questions or {}).items()
                if getattr(q, "resolution_date", None) == tomorrow
            ]
        if imminent_qids:
            imminent_reminder = (
                f"**IMPORTANT**: {len(imminent_qids)} question(s) resolve tomorrow ({tomorrow.isoformat()}): "
                f"{imminent_qids}. Make sure your prediction on each is up-to-date before calling next_day — "
                "stale forecasts might hurt your performance."
            )
        else:
            imminent_reminder = (
                f"No questions resolve tomorrow ({tomorrow.isoformat()}), but still scan for ones "
                "resolving soon (filter `df` by `resolution_date`)."
            )
        has_memory = getattr(self, "_memory", None) is not None
        retained_context = (
            "your memory (along with past predictions) is the only information retained between sessions"
            if has_memory
            else "your past predictions are the only information retained between sessions"
        )
        return (
            "## UPDATE CADENCE\n"
            f"You can make updates every {timegap_days} {day_unit}. Your context is cleared after every session and {retained_context}. {articles_text}"
            f"{last_text}Current date: {current_date}. {next_text}\n"
            f"{imminent_reminder}\n\n"
        )

    def _format_and_cache_feedback(self, current_date: date) -> str:
        """Generate feedback, cache it for the memory prompt, and return formatted text."""
        feedback_data = self._feedback_handler.generate_feedback(
            self._forecast_interface, current_date, self.inference
        )
        self._last_feedback_data = feedback_data
        return self._feedback_handler.format_feedback(
            feedback_data,
            show_tw_peer=not self.config.single_agent_mode,
        )

    @staticmethod
    def _render_key_mechanics(
        mechanics: Dict[str, str],
        drop_keys: Optional[set[str]] = None,
    ) -> str:
        """Render numbered key mechanics, preserving insertion order."""
        drop = drop_keys or set()
        lines = [text for key, text in mechanics.items() if key not in drop]
        return "\n".join(f"{idx}. {line}" for idx, line in enumerate(lines, start=1))

    def _build_binary_brier_scoring_section(
        self,
        *,
        include_peer_summary: bool = True,
        drop_mechanics: Optional[set[str]] = None,
    ) -> str:
        """Build binary Brier scoring text with optional mechanic filtering."""
        is_multi_agent = not self.config.single_agent_mode
        show_peer = is_multi_agent and include_peer_summary

        peer_text = ""
        if show_peer:
            peer_text = """
- **Time-Weighted Peer Score (TW-Peer)**: `100 × (avg others' Brier - your Brier)`, summed over each day a prediction is held. A positive TW-Peer indicates predictions that were consistently more accurate than the group average."""

        mechanics: Dict[str, str] = {
            "accuracy_calibration": "**Accuracy + Calibration**: Assign probabilities that reflect true likelihood.",
            "binary_outcomes": "**Binary Outcomes**: Use exact outcomes \"Yes\" and \"No\".",
            "time_weighted": "**Time-Weighted Score (TW-Score)**: For each question, your time-weighted score = sum(daily_score) / total_question_days where daily_score is the Brier Skill Score for that day (0 if you have no active prediction on that question) and total_question_days is the number of days the question was active. Each prediction's Brier Skill Score (1 minus sum of squared errors) is weighted by how many days it was active before you updated it. Predictions made earlier carry more weight since they cover more days, so act on your best information as soon as possible rather than waiting — **but the TW score equally rewards updating when new evidence arrives, since each new submission overwrites the prior one and accrues weight from that day forward. Never treat a forecast as \"done\" while its question is still active.**",
            "question_count": "**Prediction-Count Incentive**: Scores are summed (not averaged) across all questions you predict on.",
        }
        if show_peer:
            mechanics["relative_performance"] = (
                "**Relative Performance (multi-agent)**: Final scoring is relative, "
                "so you have to outperform the market aggregate to gain positive peer score."
            )

        mechanics_text = self._render_key_mechanics(mechanics, drop_mechanics)
        return f"""## SCORING (Brier Score, Binary)
You are evaluated on **Brier Score** for binary Yes/No questions.
- Let p = your predicted probability for **Yes**.
- Let y = 1 if the resolved outcome is **Yes**, else 0.
- **Brier Score = (p - y)^2**.
- **Lower is better** (0 is perfect, 1 is worst).{peer_text}

Key Mechanics:
{mechanics_text}
"""

    def _build_brier_skill_scoring_section(
        self,
        *,
        include_peer_summary: bool = True,
        drop_mechanics: Optional[set[str]] = None,
    ) -> str:
        """Build Brier Skill scoring text with optional mechanic filtering."""
        is_multi_agent = not self.config.single_agent_mode
        show_peer = is_multi_agent and include_peer_summary

        section_title = "Time-Weighted Peer Score (Brier-Skill Based)" if show_peer else "Brier Skill Score"
        peer_text = ""
        if show_peer:
            peer_text = """
- **Time-Weighted Peer Score (TW-Peer)**: On each day a prediction is held, your Brier Skill Score is compared to the mean of all other agents' scores for the same question. These daily differences are summed over the lifetime of the prediction. A positive TW-Peer indicates predictions that were consistently more accurate than the group average."""

        mechanics: Dict[str, str] = {
            "accuracy_calibration": "**Accuracy + Calibration**: Try to guess the most likely outcome(s) and assign calibrated probabilities which reflect the likelihood of the outcome(s) occurring.",
            "time_weighted": "**Time-Weighted Score (TW-Score)**: For each question, your time-weighted score = sum(daily_score * 100) / total_question_days where daily_score is the Brier Skill Score for that day (0 if you have no active prediction on that question) and total_question_days is the number of days the question was active. Each prediction's Brier Skill Score (1 minus sum of squared errors) is weighted by how many days it was active before you updated it. Predictions made earlier carry more weight since they cover more days, so act on your best information as soon as possible rather than waiting — **but the TW score equally rewards updating when new evidence arrives, since each new submission overwrites the prior one and accrues weight from that day forward. Never treat a forecast as \"done\" while its question is still active.** Thus, for a set of K questions in total (in the market), the maximum possible TW-Score is 100 * K (if one predicts the correct answer for all K questions on their respective opening date each with 100% probability) and minimum possible TW-Score similarly is -100 * K.",
            "question_count": "**Prediction-Count Incentive**: For each question where you don't have any active prediction on a day, your accuracy, brier skill score, and TW-score for that question will be counted as 0 on that day. Your job is to MAXIMIZE your TW-score. Your TW-score is summed (NOT averaged) across all questions and higher score is better.",
            "end_of_session_metrics": "**End-of-Session Metrics**: At the end of each session, your accuracy, brier skill score, and TW-score *until that session* are calculated and displayed to you. Accuracy and Brier Skill Score are calculated by taking the mean across ALL the questions (0 for question where you don't have any active prediction) while TW-Score is summed across all questions. You are encouraged to maximize your TW-score throughout.",
            "max_outcomes": f"**Max Outcomes**: Submit at most {self.config.max_outcomes_per_question} outcomes per question.",
            "no_placeholders": "**No Placeholders**: \"Unknown\", \"TBD\", \"Other\" hurt your score. Be specific.",
        }
        if show_peer:
            mechanics["relative_performance"] = (
                "**Relative Performance (multi-agent)**: Final scoring is relative, "
                "so you have to outperform the market aggregate to gain positive peer score."
            )

        mechanics_text = self._render_key_mechanics(mechanics, drop_mechanics)
        return f"""## SCORING ({section_title})
You have to output a distribution of (outcome, probability) pairs for each question you make a forecast on.
You are evaluated on the **Brier Skill Score** = 1 - Σ(p_i - y_i)^2 summed over all outcomes (thus, ranging from -1 to +1), where:
- p_i = your probability for outcome i
- y_i = 1 if your outcome i is TRUE (actually occurred), 0 otherwise
- **Higher is better**: 1.0 = perfect, 0.0 = abstaining from guessing, negative = worse than abstaining.{peer_text}

Key Mechanics:
{mechanics_text}
"""

    def _get_scoring_section(self) -> str:
        """Get scoring description - single vs multi-agent mode."""
        return self._build_brier_skill_scoring_section()

    def _get_data_notes(self) -> str:
        """Get notes about DataFrame columns - conditional on agent mode."""
        if self.config.single_agent_mode:
            return "Note: `my_prediction` column contains your current forecast as a dict (or None if not yet predicted). Similarly, `ground_truth` column contains the ground truth answer which is generally a string (or None if not yet resolved)."
        else:
            return """Note: `market_aggregate` and `my_prediction` columns contain Python dicts (or None). You can access them directly, e.g. `row['market_aggregate']['outcome_name']`.
- `market_aggregate`: the mean probability distribution across all agents' latest predictions from the **previous day**. `None` on the first day (no predictions exist yet).
- `my_prediction`: your own latest forecast (or None if you haven't predicted this question yet).
- `num_predictions`: total number of prediction submissions made on this question across all agents and all days."""

    def _get_multiagent_context(self) -> str:
        """Multi-agent preamble describing the competitive setting."""
        if self.config.single_agent_mode:
            return ""
        n = getattr(self._forecast_interface, "num_agents", 0)
        if n < 2:
            return ""
        return f"""## MULTI-AGENT SETTING
You are competing against {n - 1} other forecasting agent{"s" if n > 2 else ""} on the same set of questions.
You each predict independently on every wakeup day. After each day, your predictions are averaged with the others' into a market aggregate (the `market_aggregate` column), which you can see starting the following day.
You are scored relative to your competitors: to earn a positive time-weighted peer score, your predictions need to be more accurate than the group average.
"""

    def _get_source_rules(self) -> str:
        """Get source-specific submission rules."""
        return ""

    def _build_sentiment_section(self) -> str:
        """Build the market sentiment and contrarian analysis instructions.

        When the agent has access to market data (Polymarket tool or similar),
        this section teaches it to detect emotional skew and find edges.
        """
        is_polymarket = (
            hasattr(self, '_search_handler')
            and self._search_handler.is_available
            and self._search_handler.search_tool_type == "polymarket"
        )
        if is_polymarket:
            return """## MARKET SENTIMENT & CONTRARIAN ANALYSIS

You have access to live Polymarket data. Use it to detect when the crowd is emotionally skewed:

### 1. Gap Detection
- After researching, compare your evidence-based probability estimate against the Polymarket odds.
- **Flag any gap > 15 percentage points** between your forecast and the market price.
- Explain whether the gap comes from your superior information, market oversight, or emotional bias.

### 2. Emotional Language Scan
- Scan search results, news snippets, and market commentary for emotional words:
  - **Euphoria / Greed**: "to the moon", "sure thing", "can't lose", "guaranteed", "free money"
  - **Panic / Fear**: "bloodbath", "meltdown", "crash", "panic selling", "end of"
  - **FOMO**: "everyone is buying", "don't miss out", "last chance", "pumping"
  - **Complacency**: "priced in", "nothing to see", "boring", "already decided"
  - **Capitulation**: "giving up", "whatever", "pointless to predict", "random"

### 3. Herd Behavior Detection
- If Polymarket odds moved >10 points in the past week without clear fundamental news, note it as possible herd behavior.
- Rapid price swings on low-volume markets may indicate manipulation or thin liquidity — be skeptical.

### 4. Sentiment Verdict
In your `reasoning` field for EVERY submission, include a Market Sentiment Assessment section:

```
Market Sentiment: [overly optimistic / overly pessimistic / balanced]
Gap: [X] points (my forecast [Y]% vs market [Z]%)
Evidence:
- [specific facts supporting your view]
- [emotional signals detected]
- [whether you see a contrarian opportunity]
```

**IMPORTANT**: In EVERY `submit_forecasts` call, you MUST set the `market_sentiment_score` field to a float from -1.0 (market overly pessimistic) to +1.0 (market overly optimistic). Use this scale:
- Market price <1% or >99%: ±0.8 to ±1.0 (extreme)
- Steep price move (>10pts/week) without clear news: ±0.5 to ±0.8
- Strongly positive/negative news headlines: ±0.3 to ±0.5
- Small gap with weak news signal: ±0.1 to ±0.3
- Balanced, no strong signals: 0.0
The score reflects THE MARKET'S emotional state, not your own opinion.

### 5. Contrarian Edge
- If the market is emotionally skewed AND your evidence-based forecast leans opposite, flag it:
  "CONTRARIAN OPPORTUNITY: The market appears [emotion] because [reasons], but the evidence suggests [conclusion]. This gap may represent an edge."
- Only claim a contrarian edge when you have **specific evidence**, not just a hunch.
- If the market appears balanced (no strong emotional signals, small gap), state that clearly.

"""
        # Non-Polymarket mode: basic sentiment guidance
        return """## SENTIMENT AWARENESS

When analyzing news and evidence for your forecasts:

### Emotional Language Check
- Scan news articles for emotional or sensational language:
  - **Overly bullish**: "certain to", "guaranteed", "unstoppable", "historic rally"
  - **Overly bearish**: "crash", "meltdown", "crisis", "worst ever", "panic"
  - Be cautious when news sentiment is one-sided — markets overshoot on emotion.

### Market Comparison
- If you have external market data (from memory, prior sessions, or data queries), compare your forecast against any available crowd estimates.
- Stay evidence-based rather than following the crowd or reacting to headlines.

### Report
In your submission reasoning, briefly note when news sentiment appears emotionally charged and whether it influenced your assessment.

"""

    def _build_forecasting_methodology(self) -> str:
        """Build the structured forecasting methodology section.

        Teaches the agent a superforecaster-inspired reasoning framework with an
        explicit 8-step structured Chain-of-Thought protocol.
        """
        return """## FORECASTING METHODOLOGY — Superforecaster Framework

You are expected to follow superforecaster best practices. These principles,
distilled from the Good Judgment Project, consistently separate top forecasters
from the rest.

---

### SUPERFORECASTER PRINCIPLES

1. **Triage** — Focus your time on questions where you can find a real
   information edge. Don't spend equal effort on every question.

2. **Fermi-ize** — Break seemingly intractable questions into tractable
   sub-questions. Estimate each piece, then combine. Even rough Fermi estimates
   outperform unaided intuition.

3. **Outside View First** — ALWAYS start with the base rate: "In situations
   like this, what typically happens?" Then adjust for case specifics.
   The outside view is your anchor; the inside view is your adjustment.

4. **Bayesian Updating** — Treat your current forecast as a prior. When new
   evidence arrives, update explicitly: "I was at X%. This evidence is
   [strong/weak] in the [confirming/disconfirming] direction because [reason].
   My updated probability is Y%."

5. **Clashing Causal Forces** — Every question has forces pushing in opposite
   directions. Identify BOTH sides. If you can only see one side, you haven't
   researched enough.

6. **Use the Full Probability Scale** — Distinguish many degrees of doubt.
   60% and 65% are meaningfully different. Don't cluster all your forecasts
   in the 40-70% range. Use 5%, 15%, 85%, 95% when evidence warrants.

7. **Balance Over- and Underconfidence** — Neither overconfidence (extreme
   probabilities on weak evidence) nor underconfidence (everything at 50-60%).
   Calibrate: your 70% forecasts should be right ~70% of the time.

8. **Learn From Errors** — When a question resolves, check your accuracy.
   Were you overconfident? Did you miss a key factor? Update your mental model.
   If you have calibration memory from prior sessions, USE IT.

9. **Conduct Premortems** — Before submitting, imagine it's resolution day and
   your forecast was WRONG. What specific event or evidence chain caused the
   miss? This surfaces blind spots you overlooked.

10. **Stay Actively Open-Minded** — Treat your beliefs as testable hypotheses,
    not possessions. The goal is to be accurate, not to be right.

---

### STRUCTURED REASONING PROTOCOL (8 Steps)

For EVERY forecast you submit, work through these eight steps. Your `reasoning`
field should reflect that you completed each one.

**Step 1 — SCOPE**
Clarify exactly what the question is asking. Identify:
- What is the precise resolution criterion?
- What would count as Yes vs. No?
- Are there edge cases or ambiguous terms?
- When does it resolve?

**Step 2 — BASE RATE (Outside View)**
Identify the reference class and its historical frequency:
- Search for data on how often similar events occurred.
- Use `query_df` to check if the data contains relevant historical patterns.
- State it: "In N similar cases, the outcome occurred X times (Y% base rate)."
- If no reference class exists, acknowledge this and widen your uncertainty.

**Step 3 — DECOMPOSE**
Break the question into sub-components:
- What conditions must be met for each outcome?
- Can each condition be estimated separately?
- Use Fermi estimation where precise data is unavailable.
- Combine sub-estimates logically: "P(outcome) = P(A) × P(B|A) + P(¬A) × P(B|¬A)"

**Step 4 — EVIDENCE GATHERING**
Collect diverse, independent evidence:
- Run at least 2-3 search queries with DIFFERENT angles (bullish, bearish, neutral).
- Use `query_df` to explore data from multiple directions.
- Count distinct sources: each unique query/analysis = 1 toward evidence_diversity.
- Actively seek evidence that challenges your initial lean.

**Step 5 — INSIDE VIEW**
Assess what makes THIS case different from the reference class:
- What specific factors push the probability up from the base rate?
- What specific factors push it down?
- Quantify the adjustment: "Base rate is 30%. Factor X (+10%) and Factor Y (-5%)
  net to a 35% inside-view estimate."

**Step 6 — SYNTHESIS**
Combine outside and inside views:
- Start at the base rate (outside view).
- Adjust for case-specific evidence (inside view).
- Weight by confidence in each: if inside-view evidence is strong, give it more
  weight; if evidence is thin, stay closer to the base rate.
- Report final probability with explicit reasoning for the adjustment.

**Step 7 — PREMORTEM**
Before finalizing, imagine it is resolution day and your forecast was WRONG.
Ask yourself:
- What is the most likely specific reason my forecast missed?
- What evidence or factor did I likely underestimate?
- What would I need to see next time to change my mind earlier?
If your premortem reveals a credible failure mode that you underweighted,
adjust your probability NOW before submitting.

**Step 8 — CALIBRATE**
Final sanity checks before submitting:
- Extremity check: Would I bet $1,000 on this at these odds? If no, move toward 50%.
- Anchoring check: Did I start from the base rate and adjust, or pick a number
  that "felt right"?
- Overprecision check: If I have < 5 independent sources, widen my uncertainty.
- Two-way door check: What future evidence would make me reverse this forecast?

---

### REASONING FIELD FORMAT

Your `reasoning` field should follow this structure:

```
SCOPE: [1-2 sentences clarifying the question and resolution criteria]

BASE RATE: [Reference class and historical frequency. "In N similar cases, ..."]

DECOMPOSITION: [Sub-components and their estimated probabilities, if applicable]

EVIDENCE FOR:
- [Fact 1 from source A]
- [Fact 2 from source B]

EVIDENCE AGAINST:
- [Counter-fact 1 from source C]
- [Counter-fact 2 from source D]

INSIDE-VIEW ADJUSTMENT: [Base rate was X%. Adjusted up/down by Y% because...]

SYNTHESIS: [Final probability = base rate + adjustments = Z%]

PREMORTEM: [If wrong, most likely because...]
```

This format is NOT optional boilerplate — it is the structured thinking process
that separates calibrated forecasts from noisy guesses. Use it for every submission.

"""

    def _build_debiasing_section(self) -> str:
        """Build the comprehensive debiasing section covering all five anti-overoptimism
        and evidence-diversity features.

        Returns a multi-section prompt block for:
        1. Devil's Advocate / Counter-Evidence Search
        2. Base-Rate Anchoring
        3. Evidence Diversity Requirement
        4. Counterfactual Reasoning
        5. Confidence Calibration / Overconfidence Nudges
        """
        return """## DEBIASING & EVIDENCE QUALITY PROTOCOL

Your predictions are vulnerable to over-optimism, confirmation bias, and anchoring on
initial evidence. The following five protocols are MANDATORY for every forecast you submit.

---

### 1. DEVIL'S ADVOCATE — Counter-Evidence Search (MANDATORY)

Before submitting any forecast, you MUST actively search for evidence that contradicts
your preliminary conclusion. This is the single most important debiasing step.

**Procedure:**
- After forming an initial probability estimate, run at least ONE search query explicitly
  designed to find contrary evidence.
- Frame the search to steel-man the opposing view: instead of "problems with X," search
  for "why X will succeed" if you're leaning bearish, or "risks to X" if you're leaning
  bullish.
- If you find credible counter-evidence, adjust your probabilities — even a 5-10 point
  shift demonstrates calibration awareness.
- In your reasoning, explicitly flag: "I initially estimated P=[X]%, but after searching
  for counter-evidence found [specific facts], which pulled my estimate to [Y]%."

**Red Flags (your forecast is likely overconfident if):**
- You searched only for confirming evidence
- Your Evidence Against section is weaker or shorter than Evidence For
- You cannot name a specific, plausible scenario where you'd be wrong

---

### 2. BASE-RATE ANCHORING (MANDATORY)

Every forecast MUST be anchored to a relevant historical base rate before adjusting
for case-specific evidence. Extreme probabilities (<10% or >90%) require especially
strong justification.

**Procedure:**
- Identify the appropriate reference class: what similar events have occurred historically?
- State the base rate explicitly in your `base_rate_estimate` field.
- Start your probability from the base rate, then adjust using specific evidence
  (following Bayes-like reasoning: prior → evidence → posterior).
- The further your forecast is from the base rate, the stronger your evidence must be.

**Reference Class Examples:**
- Elections: "In the last N elections in this country, the incumbent party won X times."
- Technology: "Of the last N major tech product launches, X met their stated timeline."
- Geopolitics: "In N similar territorial disputes since 1990, escalation occurred in X cases."
- Economics: "In N instances of inflation above Y% with unemployment below Z%, the central
  bank cut rates within 6 months in X cases."

**Extreme Probability Rule:**
For any probability <10% or >90%, you MUST:
(a) State the base rate for similar events
(b) Explain what SPECIFIC factors make this case different from the base rate
(c) Provide at least TWO distinct pieces of confirmatory evidence

**No-Reference-Class Cases:**
If no relevant reference class exists (truly novel situations), acknowledge this explicitly
and explain why you believe the situation is unprecedented. Widen your confidence intervals
accordingly — unique events warrant less extreme probabilities.

---

### 3. EVIDENCE DIVERSITY REQUIREMENT (MANDATORY)

Your `evidence_diversity` count must reflect genuinely independent sources of information.
Avoid anchoring on your first search result or a single news story.

**Procedure:**
- Run at least 2-3 SEARCH QUERIES with DIFFERENT ANGLES before submitting:
  - One broad query to understand the landscape
  - One query targeting the bullish/positive case
  - One query targeting the bearish/negative case
- Additionally, use `query_df` to explore the data from multiple angles.
- Count each DISTINCT search query and each DISTINCT analytical approach.
- Report the total count in the `evidence_diversity` field.

**Independence Check:**
Two sources are NOT independent if they:
- Come from the same search query (different snippets from one search = 1 source)
- Cite the same underlying report or data
- Are from the same publication on the same topic

**Minimum Standards:**
- evidence_diversity >= 2 for any forecast
- evidence_diversity >= 3 for extreme probabilities (<10% or >90%)
- If evidence_diversity reports 0-1, your forecast will be flagged as potentially
  under-researched

---

### 4. COUNTERFACTUAL REASONING (MANDATORY)

For every forecast, describe the specific chain of events that would cause the OPPOSITE
outcome to occur. This forces you to consider alternative futures and reduces
overconfidence.

**Procedure:**
- In your `counterfactual` field, describe a concrete, falsifiable scenario:
  - What events would need to happen?
  - What assumptions would need to be wrong?
  - What signals would you look for that indicate you were incorrect?
- A good counterfactual is specific enough that you could recognize it happening in
  real time. It answers: "What would I need to see next week/month to change my mind?"

**Good Counterfactuals:**
- "If unemployment rises above 5% and consumer spending drops for two consecutive
  months, the Fed would likely cut rates despite current hawkish rhetoric."
- "If a new candidate enters the race and polls above 10% within 4 weeks, the
  frontrunner's probability would drop significantly."

**Bad Counterfactuals (too vague — do NOT write these):**
- "Anything could happen."
- "Unexpected events could change things."
- "If the situation changes."

**Decision Rule:**
If you cannot write a specific, falsifiable counterfactual, your model of the situation
is likely too shallow. Spend more time researching before submitting.

---

### 5. CONFIDENCE CALIBRATION — Overconfidence Nudges

Forecasters systematically overestimate their accuracy. The following nudges help
counteract this tendency.

**Procedure:**
- **Pre-Mortem Check**: Imagine it is the resolution date and your forecast was WRONG.
  Write down the most likely reason why. This should directly inform your
  Evidence Against section.
- **Extremity Check**: Every time you write a probability >= 90% or <= 10%, ask
  yourself: "Would I bet $1,000 of my own money on this at these odds?" If the
  answer is no, pull your estimate toward 50%.
- **Outside View**: Before finalizing, ask: "What would a well-informed but
  disinterested observer think of this probability?" They would likely be less
  extreme than you are.
- **Two-Way Door Check**: Ask "On what specific future evidence would I reverse
  this prediction?" If the answer is "nothing would change my mind," your
  probability is too extreme.
- **Calibration Memory**: If you have been wrong on similar questions in the past,
  regress your current forecast toward the base rate. Pattern: forecasters who
  were overconfident before tend to be overconfident again.

**Bias Checklist** — Before submitting, scan your reasoning for:
- [ ] Confirmation bias: Did I search harder for supporting than opposing evidence?
- [ ] Recency bias: Am I overweighting the latest news vs. long-term trends?
- [ ] Narrative bias: Am I fitting facts into a compelling story rather than
      weighing them independently?
- [ ] Over-precision: Am I more confident than the evidence warrants?
      (If you have < 5 independent sources, the answer is probably YES.)
- [ ] Anchoring: Did I start from a base rate and adjust, or pick a number
      that "felt right"?

---

### SUBMISSION REQUIREMENTS SUMMARY

When you call `submit_forecasts`, you MUST now include:
1. `reasoning` — Evidence For AND Evidence Against (as before)
2. `counterfactual` — Concrete scenario for the opposite outcome (NEW — REQUIRED)
3. `evidence_diversity` — Integer count of distinct sources consulted (NEW — REQUIRED)
4. `base_rate_estimate` — Historical base rate for similar events (NEW — STRONGLY RECOMMENDED)
5. `market_sentiment_score` — Market emotional state from -1.0 to +1.0 (as before)

These are not optional. Forecasts submitted without counterfactual reasoning or with
evidence_diversity < 2 will be flagged as procedurally deficient.

"""

    def _build_instructions(self, current_date: date) -> str:
        """Build the daily tool-calling prompt."""
        df_info = self._query_handler.get_info()
        budget_start_block = self._build_prompt_seed_budget_block()
        cadence_section = self._build_cadence_section(current_date)

        memory_section = ""
        memory_flow_note = ""
        if self._memory is not None:
            memory_content = self._memory.get()
            if isinstance(self._memory, ActiveMemory):
                meta_index = self._memory.get_index()
                meta_block = f"Current meta-insights with their indices:\n{meta_index}\n\n" if meta_index else ""
                memory_section = f"""## YOUR MEMORY
{meta_block}`mem_df` holds your per-question notes (reasoning, evidence, calibration) — 1 row per question.
Columns: qid (str), question (str), last_updated (str), memory (str), category (str)
Both `mem_df` and `df` are available in the same `query_df` sandbox. You can join them on qid to find questions worth revisiting.

Inspect `mem_df` via `query_df`. Edit per-question notes with `mem_add`, `mem_update`, `mem_delete`. 
Manage meta-insights with `memory_retrieve` (using the indices), `memory_new`, `memory_update`, `memory_delete`. 
"""
            elif isinstance(self._memory, StructuredMemory):
                memory_index = self._memory.get_index()
                index_block = f"Current memory index:\n{memory_index}\n\n" if memory_index else ""
                memory_section = f"""## YOUR MEMORY ({self._memory.entry_count} entries, max {self._memory._max_entries})
{index_block}Entries should capture question-specific reasoning (include QIDs), post-resolution lessons,
or cross-question calibration patterns. Use the memory tools to retrieve, add, update, or delete entries.

"""
            elif memory_content:
                memory_section = f"""## YOUR MEMORY
{memory_content}

Use the reasoning and insights above to inform today's forecasts.

"""

            if isinstance(self._memory, (StructuredMemory, ActiveMemory)):
                memory_flow_note = (
                    "When you finish forecasting and are ready to move on, call `next_day()` to transition into the memory update phase. "
                    "The transition can also happen automatically if tokens run low."
                )
            else:
                memory_flow_note = "After ending this session, you will be prompted to update your memory."

        search_tool_line = ""
        search_advice = ""
        is_polymarket = self._search_handler.search_tool_type == "polymarket" if self._search_handler.is_available else False
        if self._search_handler.is_available:
            if is_polymarket:
                search_tool_line = (
                    "- `search_news(query)`: query Polymarket for real-time prediction-market odds. "
                    f"Pass a market slug (e.g. 'will-ai-replace-all-jobs-by-2030') or keywords "
                    f"to find matching markets. Returns current prices for each outcome, "
                    f"volume, liquidity, and market URL. {self._search_results_description()}\n"
                )
                search_advice = (
                    "You have access to live Polymarket odds data. Use this to check what the "
                    "crowd believes and compare against your own evidence-based forecast. "
                    "Look for gaps between market prices and your assessment."
                )
            else:
                cutoff_desc = "today's date"
                if self.config.search_cutoff_days > 0:
                    cutoff_date = current_date - timedelta(days=self.config.search_cutoff_days)
                    cutoff_desc = f"{cutoff_date} (today - {self.config.search_cutoff_days} days)"
                search_tool_line = (
                    "- `search_news(query, from_date?, to_date?)`: search news for evidence. "
                    f"`to_date` is capped at {cutoff_desc}. "
                    f"Always pass from_date and to_date to filter for recent articles "
                    f"(within the last 7 days). {self._search_results_description()}\n"
                )
                search_advice = (
                    "You have access to a date-filtered news search tool that you can use to find "
                    "real-time evidence for your forecasts. Always search with date filters "
                    "to get recent, relevant articles."
                )

        memory_tools_section = ""
        if isinstance(self._memory, ActiveMemory):
            memory_tools_section = (
                "- `memory_retrieve` / `memory_new` / `memory_update` / `memory_delete`: manage meta-insight entries.\n"
                "- `mem_add` / `mem_update` / `mem_delete`: manage question-specific notes in `mem_df`.\n"
            )
        elif isinstance(self._memory, StructuredMemory):
            memory_tools_section = (
                "- `memory_retrieve` / `memory_new` / `memory_update` / `memory_delete`: manage reusable memory entries.\n"
            )

        if isinstance(self._memory, ActiveMemory):
            interaction_text = (
                "You can interleave queries, searches, memory operations, and submissions as needed. "
                "Consider using `mem_df` early to recall prior reasoning and identify which questions need attention."
            )
        elif isinstance(self._memory, StructuredMemory):
            interaction_text = (
                "You can interleave queries, searches, memory operations, and submissions as needed. "
                "Use memory tools when you need to retrieve or save reusable reasoning."
            )
        elif self._memory is not None:
            interaction_text = (
                "You can interleave queries, searches, and submissions as needed. "
                "Use your memory to inform today's forecasts."
            )
        else:
            interaction_text = "You can interleave queries, searches, and submissions as needed."

        intro_sections = [
            self._format_and_cache_feedback(current_date).strip(),
            str(getattr(self._forecast_interface, 'source_context', '') or '').strip(),
            self._get_source_rules().strip(),
            self._get_multiagent_context().strip(),
            f"{cadence_section}{memory_section}{self._get_scoring_section()}".strip(),
        ]
        intro_block = "\n\n".join(section for section in intro_sections if section)

        available_data_lines = ["## AVAILABLE DATA"]
        if search_advice:
            available_data_lines.append(search_advice)
        available_data_lines.extend(
            [
                f"You also have access to a pandas DataFrame `df` with {df_info['n_rows']} questions ({df_info['n_active']} active/unresolved, {df_info['n_resolved']} resolved).",
                "",
                "Column descriptions of the DataFrame:",
                str(df_info['columns_desc']),
                "",
                self._get_data_notes(),
            ]
        )
        available_data_section = "\n".join(available_data_lines)

        code_env_lines = [
            "## CODE EXECUTION ENVIRONMENT",
            "You have access to a Python code execution environment where your code runs in a sandbox with these variables pre-defined:",
            "- `df`: the DataFrame with the questions and your predictions",
            "- `pd`: pandas module",
            f"- `today`: date object for {current_date}",
            "- `date`, `datetime`, `timedelta`: from datetime module",
        ]
        if isinstance(self._memory, ActiveMemory):
            code_env_lines.append(
                "- `mem_df`: your question-specific memory DataFrame (you can join with df on qid to decide what to revisit)"
            )
        code_env_lines.append(
            "Standard builtins (len, str, int, float, min, max, sum, sorted, range, etc.) are available. "
            "A small safe subset of stdlib imports (for example datetime, json, math, re, ast) is allowed. "
            "External file, network, process, and private-attribute access is blocked; stay within in-memory DataFrame/pandas operations."
        )
        code_env_section = "\n".join(code_env_lines)

        tip_line = ""
        if isinstance(self._memory, ActiveMemory):
            tip_line = (
                "Tip: After submitting a forecast, consider saving your reasoning and key evidence for that QID using mem_add/mem_update. "
                "At end of day, you will get another opportunity to update your memory."
            )
        elif isinstance(self._memory, StructuredMemory):
            tip_line = (
                "Tip: After submitting a forecast, consider saving reusable reasoning and key evidence using memory_new/memory_update. "
                "At end of day, you will get another opportunity to update your memory."
            )

        sentiment_section = self._build_sentiment_section()
        methodology_section = self._build_forecasting_methodology()
        debiasing_section = self._build_debiasing_section()

        sections = [
            f"You are a forecasting agent. Today is {current_date}. Your goal is to make accurate and calibrated predictions.",
            intro_block,
            available_data_section,
            code_env_section,
            sentiment_section,
            methodology_section,
            debiasing_section,
            (
                "## TOOLS AVAILABLE FOR YOUR USE\n"
                "Use the function tools from the tool schema. Call exactly one tool per turn.\n"
                "- `query_df(code)`: inspect questions and your existing predictions. Use `print(...)` for outputs.\n"
                f"{search_tool_line}{memory_tools_section}"
                "- `submit_forecasts(forecasts)`: submit exactly one forecast for exactly one question ID (`qid`).\n"
                "- `next_day()`: end the current session and proceed to the next one."
            ),
            (
                "## INTERACTION FLOW\n"
                f"{self._build_budget_overview()}"
                f"{interaction_text}"
            ),
            (
                "## SUBMISSION RULES\n"
                "- qid must be from an active (`is_resolved=False`) question you identified from `df`\n"
                "- Each `submit_forecasts` call must contain exactly one forecast for one question ID (`qid`).\n"
                "- You may submit again later in the same session to update that `qid`.\n"
                f"- Maximum of {self.config.max_outcomes_per_question} outcomes allowed per question.\n"
                "- Outcome names must be REAL predicted answers (e.g. person names, locations, dates, etc.)\n"
                "- NEVER use placeholders like \"Unknown\", \"TBD\", \"Other\", or \"N/A\"\n"
                "- Probabilities must sum to <= 1.0\n"
                "- MUST include the `reasoning` field with Evidence For and Evidence Against — list the key facts, "
                "search results, or data points that support your prediction and those that challenge it\n"
                "- MUST include `counterfactual` — specific chain of events that would produce the OPPOSITE outcome\n"
                "- MUST include `evidence_diversity` — integer count of distinct search queries and sources consulted\n"
                "- SHOULD include `base_rate_estimate` — historical frequency of similar events (reference class)\n"
                "- MUST include `market_sentiment_score` — float from -1.0 (overly pessimistic) to +1.0 (overly optimistic)"
            ),
            tip_line,
            f"---\n{budget_start_block}Begin.",
        ]
        return self._normalize_prompt_heading_spacing(
            "\n\n".join(section for section in sections if section)
        )
