

"""
Typed Pydantic models for the Predictive Maintenance Arena.
Uses OpenEnv's base Action and Observation classes.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field

from openenv.core.env_server.types import Action, Observation


# ── Action Space ─────────────────────────────────────────────────────────────

class ActionType(str, Enum):
    """All actions the maintenance agent can take."""

    MONITOR = "monitor"                          # Observe all machines, take no action
    RUN_DIAGNOSTIC = "run_diagnostic"            # Deep test on one machine
    SCHEDULE_MAINTENANCE = "schedule_maintenance" # Preventive maintenance
    EMERGENCY_SHUTDOWN = "emergency_shutdown"     # Immediate shutdown
    ORDER_PARTS = "order_parts"                  # Order spare parts
    ADJUST_SPEED = "adjust_speed"                # Change line speed (params: speed_pct 50-100)


class MaintenanceAction(Action):
    """An action taken by the maintenance agent."""

    action_type: ActionType = Field(
        ..., description="The type of action to perform"
    )
    machine_id: int = Field(
        default=-1,
        description="Target machine index (0-4). -1 for actions that affect all.",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Action-specific parameters. E.g. {'speed_pct': 80}.",
    )


# ── Observation Space ────────────────────────────────────────────────────────

class MaintenanceObservation(Observation):
    """What the agent sees after each action."""

    # Per-machine observable state
    machines: list[dict] = Field(
        default_factory=list,
        description=(
            "List of 5 machine dicts, each with: name, status, sensors "
            "(dict of sensor_name -> value), health_bar (coarse: good/warn/crit), "
            "is_under_maintenance, maintenance_hours_left"
        ),
    )

    # Production metrics
    hour: int = Field(0, description="Current hour (0-167)")
    max_hours: int = Field(168, description="Episode length in hours")
    units_produced: int = Field(0, description="Total units produced so far")
    production_target: int = Field(0, description="Target units for the week")
    line_speed_pct: int = Field(100, description="Current production line speed (50-100%)")

    # Resource state
    crew_available: bool = Field(True, description="Is maintenance crew free?")
    crew_busy_on: int = Field(-1, description="Machine ID crew is working on (-1 if free)")
    crew_hours_left: int = Field(0, description="Hours until crew finishes current job")
    spare_parts: dict[str, int] = Field(
        default_factory=dict,
        description="Spare parts inventory: machine_name -> count",
    )
    parts_on_order: dict[str, int] = Field(
        default_factory=dict,
        description="Parts ordered but not arrived: machine_name -> hours_to_arrival",
    )

    # Action result
    action_result: str = Field(
        "", description="Textual result of the last action"
    )
    action_success: bool = Field(True, description="Whether the action executed OK")

    # Event log (last 5 events)
    recent_events: list[str] = Field(
        default_factory=list,
        description="Recent events in reverse chronological order",
    )

    # Reward breakdown (for transparency)
    reward_breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Per-component reward breakdown for this step",
    )

    # Difficulty level
    difficulty_level: int = Field(1, description="Current adaptive difficulty (1-5)")

    # Available actions
    available_actions: list[str] = Field(
        default_factory=list,
        description="List of valid action types",
    )
