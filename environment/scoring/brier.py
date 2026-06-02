"""
Brier Skill Score for free-form outcome prediction.

Score = 1 - Σ(p_i - y_i)²

Where:
- Sum is over {named outcomes} ∪ {truth}
- p_i = probability assigned (0 if outcome not named)
- y_i = 1 if outcome is truth, 0 otherwise

Properties:
- Higher is better. **Skill = 1 − Brier** lies in **[-1, 1]** for a proper simplex distribution
  (non‑negative probs, sum ≤ 1) over the scored outcomes; worst calibrated cases often approach **−1**.
- 0 = no information baseline (abstainer)
- Proper scoring rule
- No overconfidence incentive
"""

from typing import Optional

from .base import BaseScorer, DailyPrediction


class BrierScorer(BaseScorer):
    """
    Brier Skill Score. Default scorer for free-form prediction.
    
    Standard multi-class Brier over {named outcomes} ∪ {truth},
    converted to skill score (1 - Brier) so higher is better.
    
    Key: if truth not in named outcomes, we include it with p=0.
    """
    
    higher_is_better = True
    
    def score_prediction(
        self, 
        pred: DailyPrediction, 
        ground_truth: str, 
        matcher=None,
        question_id: str = None,
        question_title: str = None
    ) -> Optional[float]:
        """
        Compute Brier Skill Score.
        
        Score = 1 - Σ(p_i - y_i)² over {named outcomes} ∪ {truth}
        """
        brier = 0.0
        
        # Collect ALL outcomes that match the ground truth
        matched_outcomes = set()
        matcher_ambiguous = False

        # Helper for normalized comparison (lowercase, no spaces)
        def normalize(s: str) -> str:
            return s.lower().replace(" ", "").strip()

        truth_norm = normalize(ground_truth)

        if ground_truth in pred.outcomes:
            matched_outcomes.add(ground_truth)
        elif matcher:
            # LLM-based semantic matching — collect all equivalent outcomes
            for outcome in pred.outcomes:
                result = matcher.is_equivalent(outcome, ground_truth,
                                               question_id=question_id, question_title=question_title,
                                               match_type="check_guess")
                if result is True:
                    matched_outcomes.add(outcome)
                elif result is None:
                    matcher_ambiguous = True
        else:
            # Exact matching with normalization (lowercase + no spaces)
            for outcome in pred.outcomes:
                if normalize(outcome) == truth_norm:
                    matched_outcomes.add(outcome)

        # If the matcher was ambiguous and we found no matches, exclude from scoring
        if not matched_outcomes and matcher_ambiguous:
            return None

        # Sum over named outcomes — all matched outcomes get y=1
        for outcome, prob in pred.outcomes.items():
            y = 1.0 if outcome in matched_outcomes else 0.0
            brier += (prob - y) ** 2

        # If truth not in named outcomes, include it with p=0
        if not matched_outcomes:
            brier += (0.0 - 1.0) ** 2  # = 1

        return 1.0 - brier


default_scorer = BrierScorer()

