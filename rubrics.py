
"""
Composable Rubric definitions for the Predictive Maintenance Arena.

Uses OpenEnv's Rubric base class and WeightedSum container.
Four rubrics combine into the final score:
  - UptimeRubric (40%): Production uptime percentage
  - CostRubric (25%): Cost efficiency
  - PredictionRubric (20%): Accuracy of maintenance decisions
  - CascadeRubric (15%): Cascade failure prevention
"""

from __future__ import annotations

from typing import Any

from openenv.core.rubrics.base import Rubric
from openenv.core.rubrics.containers import WeightedSum


class UptimeRubric(Rubric):
    """Scores production uptime — how much of the target was produced."""

    def forward(self, action: Any, observation: Any) -> float:
        md = getattr(observation, "metadata", {}) or {}
        produced = md.get("units_produced", 0)
        target = md.get("production_target", 800)
        if target <= 0:
            return 1.0
        ratio = produced / target
        return min(1.0, max(0.0, ratio))


class CostRubric(Rubric):
    """Scores cost efficiency — lower cost relative to maximum is better."""

    def forward(self, action: Any, observation: Any) -> float:
        md = getattr(observation, "metadata", {}) or {}
        cost = md.get("total_cost", 0)
        max_cost = 50000  # 3 machines failing ≈ ~45K
        if cost <= 0:
            return 1.0
        ratio = cost / max_cost
        return max(0.0, 1.0 - ratio)


class PredictionRubric(Rubric):
    """Scores prediction accuracy — correct vs unnecessary maintenance.

    An agent that never attempts maintenance gets 0.0 because failing
    to maintain is not a neutral outcome — machines will degrade and fail.
    The agent must both attempt maintenance AND be accurate.
    """

    def forward(self, action: Any, observation: Any) -> float:
        md = getattr(observation, "metadata", {}) or {}
        correct = md.get("correct_preventive", 0)
        unnecessary = md.get("unnecessary_preventive", 0)
        total = correct + unnecessary
        if total == 0:
            return 0.0  # FIX #6: Never attempted maintenance — worst outcome
        accuracy = correct / total
        return accuracy


class CascadeRubric(Rubric):
    """Scores cascade failure prevention.

    Measures how many potential cascades were prevented vs how many occurred.
    If no cascade risk arose during the episode, returns 1.0 (no cascades
    happened and none needed preventing).
    """

    def forward(self, action: Any, observation: Any) -> float:
        md = getattr(observation, "metadata", {}) or {}
        cascades = md.get("cascade_failures", 0)
        prevented = md.get("prevented_cascades", 0)
        total_risk = cascades + prevented
        if total_risk == 0:
            return 1.0
        return prevented / total_risk


def create_maintenance_rubric() -> WeightedSum:
    """Create the composite rubric for the maintenance arena."""
    return WeightedSum(
        rubrics=[
            UptimeRubric(),
            CostRubric(),
            PredictionRubric(),
            CascadeRubric(),
        ],
        weights=[0.40, 0.25, 0.20, 0.15],
    )
