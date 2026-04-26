

"""
Core environment for the Predictive Maintenance Arena.

Extends OpenEnv's Environment base class with proper reset/step/state.
"""

from __future__ import annotations

import uuid
from typing import Any

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from .models import (
        ActionType,
        MaintenanceAction,
        MaintenanceObservation,
    )
    from .machines import FactorySimulator
    from .rubrics import create_maintenance_rubric
except ImportError:
    from models import (
        ActionType,
        MaintenanceAction,
        MaintenanceObservation,
    )
    from machines import FactorySimulator
    from rubrics import create_maintenance_rubric


MAX_HOURS = 168  # One week


class MaintenanceArenaEnvironment(Environment):
    """
    Predictive Maintenance Arena — an OpenEnv environment where AI agents
    manage a factory floor of 5 interconnected machines.

    Themes: World Modeling (3.1) + Self-Improvement (4)
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        rubric = create_maintenance_rubric()
        super().__init__(rubric=rubric)
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)
        self._factory = FactorySimulator()
        self._done = False
        self._cumulative_reward = 0.0
        self._step_rewards: list[float] = []
        self._difficulty_level = 1
        # FIX #7: Difficulty history lives here, not in factory,
        # so it persists across reset() calls within the same env instance
        self._difficulty_history: list[float] = []

    # ── OpenEnv Interface ────────────────────────────────────────────────

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> MaintenanceObservation:
        """Initialize a new episode (one week of factory operation)."""
        self._state = State(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
        )
        self._done = False
        self._cumulative_reward = 0.0
        self._step_rewards = []

        difficulty = kwargs.get("difficulty", self._difficulty_level)
        self._factory.reset(seed=seed, difficulty=difficulty)

        return self._build_observation(
            action_result="New week started. 5 machines online. Begin monitoring.",
            action_success=True,
            step_reward=0.0,
        )

    def step(self, action: MaintenanceAction) -> MaintenanceObservation:
        """Execute one action and advance the factory by one hour."""
        if self._done:
            return self._build_observation(
                action_result="Episode already ended.",
                action_success=False,
                step_reward=0.0,
            )

        self._state.step_count += 1

        # Process the agent's action
        result_text, action_reward = self._process_action(action)

        # Advance the factory by one hour
        step_result = self._factory.step()

        # Compute step reward
        step_reward = self._compute_step_reward(action, action_reward)

        # Add factory events to result
        if step_result["events"]:
            result_text += "\n\nEVENTS THIS HOUR:\n" + "\n".join(
                f"  - {e}" for e in step_result["events"]
            )

        # Check episode end
        if self._factory.hour >= MAX_HOURS:
            self._done = True
            result_text += f"\n\nWEEK COMPLETE. Total units: {self._factory.units_produced}"

        # Check total line failure
        all_failed = all(
            m.status == "failed" for m in self._factory.machines
        )
        if all_failed:
            self._done = True
            result_text += "\n\nTOTAL LINE SHUTDOWN — All machines failed!"
            step_reward -= 5.0  # Extra penalty for total shutdown

        # FIX #13: Track cumulative AFTER all adjustments (including -5.0 penalty)
        self._cumulative_reward += step_reward
        self._step_rewards.append(step_reward)

        obs = self._build_observation(
            action_result=result_text,
            action_success=True,
            step_reward=round(step_reward, 4),
        )

        # If episode ended, compute final rubric score
        if self._done:
            obs.metadata = obs.metadata or {}
            obs.metadata.update(self._get_rubric_metadata())
            if self.rubric:
                final_score = self.rubric(action, obs)
                obs.metadata["final_score"] = round(final_score, 4)

                # FIX #7: Adjust difficulty at environment level (persists across episodes)
                self._difficulty_history.append(final_score)
                if len(self._difficulty_history) >= 3:
                    recent_avg = sum(self._difficulty_history[-3:]) / 3
                    if recent_avg > 0.7 and self._difficulty_level < 5:
                        self._difficulty_level += 1
                    elif recent_avg < 0.3 and self._difficulty_level > 1:
                        self._difficulty_level -= 1
                obs.metadata["difficulty_level"] = self._difficulty_level

        return obs

    @property
    def state(self) -> State:
        """Return current environment state."""
        return self._state

    # ── Action Processing ────────────────────────────────────────────────

    def _process_action(self, action: MaintenanceAction) -> tuple[str, float]:
        """
        Process an agent action.
        Returns (result_text, action_specific_reward).
        """
        at = action.action_type
        mid = action.machine_id

        if at == ActionType.MONITOR:
            return self._factory.do_monitor(), 0.0

        elif at == ActionType.RUN_DIAGNOSTIC:
            result = self._factory.do_diagnostic(mid)
            return result, -0.05

        elif at == ActionType.SCHEDULE_MAINTENANCE:
            result, was_correct = self._factory.do_schedule_maintenance(mid)
            reward = 0.8 if was_correct else -0.3
            return result, reward

        elif at == ActionType.EMERGENCY_SHUTDOWN:
            result = self._factory.do_emergency_shutdown(mid)
            return result, -0.2

        elif at == ActionType.ORDER_PARTS:
            result = self._factory.do_order_parts(mid)
            return result, -0.05

        elif at == ActionType.ADJUST_SPEED:
            speed = action.parameters.get("speed_pct", 100)
            result = self._factory.do_adjust_speed(int(speed))
            return result, 0.0

        return "Unknown action.", -0.1

    def _compute_step_reward(
        self, action: MaintenanceAction, action_reward: float
    ) -> float:
        """
        Compute the dense per-step reward.

        Components:
          1. Production reward: +0.2 per running machine (max +1.0)
          2. Action-specific reward (from _process_action)
          3. Failure penalty: -1.5 per machine that NEWLY failed this step
        """
        reward = 0.0

        # Production reward
        running = sum(
            1 for m in self._factory.machines
            if m.status in ("running", "warning")
        )
        reward += running * 0.2

        # Action-specific
        reward += action_reward

        # FIX #1: Only penalize NEWLY failed machines, not already-failed ones
        reward -= len(self._factory._newly_failed) * 1.5

        return reward

    # ── Observation Building ─────────────────────────────────────────────

    def _build_observation(
        self,
        action_result: str,
        action_success: bool,
        step_reward: float,
    ) -> MaintenanceObservation:
        """Build the full observation for the agent."""
        factory = self._factory

        return MaintenanceObservation(
            done=self._done,
            reward=step_reward,
            machines=factory.get_observable_machines(),
            hour=factory.hour,
            max_hours=MAX_HOURS,
            units_produced=factory.units_produced,
            production_target=factory.production_target,
            line_speed_pct=factory.line_speed_pct,
            crew_available=factory.crew_available,
            crew_busy_on=factory.crew_busy_machine,
            crew_hours_left=factory.crew_hours_left,
            spare_parts=dict(factory.spare_parts),
            parts_on_order=dict(factory.parts_on_order),
            action_result=action_result,
            action_success=action_success,
            recent_events=factory.events[:5],
            reward_breakdown={
                "cumulative_reward": round(self._cumulative_reward, 3),
                "step_reward": round(step_reward, 3),
                "steps_taken": self._state.step_count,
            },
            difficulty_level=factory.difficulty_level,
            available_actions=[a.value for a in ActionType],
            metadata={
                "episode_id": self._state.episode_id,
                "step_count": self._state.step_count,
            },
        )

    def _get_rubric_metadata(self) -> dict:
        """Get metadata needed by rubrics for final scoring."""
        f = self._factory
        return {
            "units_produced": f.units_produced,
            "production_target": f.production_target,
            "total_cost": f.total_cost,
            "correct_preventive": f.correct_preventive,
            "unnecessary_preventive": f.unnecessary_preventive,
            "cascade_failures": f.cascade_failures,
            "prevented_cascades": f.prevented_cascades,
            "total_downtime_hours": f.total_downtime_hours,
            "difficulty_level": f.difficulty_level,
        }
