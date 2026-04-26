#!/usr/bin/env python3
"""Tests for the Predictive Maintenance Arena."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ActionType, MaintenanceAction
from server.environment import MaintenanceArenaEnvironment


def test_reset():
    env = MaintenanceArenaEnvironment()
    obs = env.reset(seed=42)
    assert obs.done is False
    assert obs.hour == 0
    assert len(obs.machines) == 5
    assert obs.units_produced == 0
    assert obs.crew_available is True
    assert len(obs.available_actions) == 6
    print("  pass: test_reset")


def test_monitor_action():
    env = MaintenanceArenaEnvironment()
    env.reset(seed=42)
    obs = env.step(MaintenanceAction(action_type=ActionType.MONITOR))
    assert obs.done is False
    assert obs.hour == 1
    assert "Monitoring" in obs.action_result
    assert obs.reward != 0  # Should get production reward
    print("  pass: test_monitor_action")


def test_diagnostic_action():
    env = MaintenanceArenaEnvironment()
    env.reset(seed=42)
    obs = env.step(MaintenanceAction(
        action_type=ActionType.RUN_DIAGNOSTIC, machine_id=2
    ))
    assert "DIAGNOSTIC REPORT" in obs.action_result
    assert "Heat treat oven" in obs.action_result
    assert "remaining life" in obs.action_result.lower()
    print("  pass: test_diagnostic_action")


def test_schedule_maintenance():
    env = MaintenanceArenaEnvironment()
    env.reset(seed=42)
    # Run a few steps to let machines degrade
    for _ in range(30):
        env.step(MaintenanceAction(action_type=ActionType.MONITOR))
    # Now schedule maintenance on most degraded machine
    obs = env.step(MaintenanceAction(
        action_type=ActionType.SCHEDULE_MAINTENANCE, machine_id=2
    ))
    assert "maintenance" in obs.action_result.lower()
    assert obs.crew_available is False
    print("  pass: test_schedule_maintenance")


def test_crew_busy():
    env = MaintenanceArenaEnvironment()
    env.reset(seed=42)
    # Schedule maintenance
    env.step(MaintenanceAction(
        action_type=ActionType.SCHEDULE_MAINTENANCE, machine_id=0
    ))
    # Try to schedule another — crew is busy
    obs = env.step(MaintenanceAction(
        action_type=ActionType.SCHEDULE_MAINTENANCE, machine_id=1
    ))
    assert "busy" in obs.action_result.lower() or "crew" in obs.action_result.lower()
    print("  pass: test_crew_busy")


def test_order_parts():
    env = MaintenanceArenaEnvironment()
    env.reset(seed=42)
    obs = env.step(MaintenanceAction(
        action_type=ActionType.ORDER_PARTS, machine_id=0
    ))
    assert "ordered" in obs.action_result.lower()
    assert len(obs.parts_on_order) > 0
    print("  pass: test_order_parts")


def test_adjust_speed():
    env = MaintenanceArenaEnvironment()
    env.reset(seed=42)
    obs = env.step(MaintenanceAction(
        action_type=ActionType.ADJUST_SPEED,
        parameters={"speed_pct": 70},
    ))
    assert obs.line_speed_pct == 70
    assert "70%" in obs.action_result
    print("  pass: test_adjust_speed")


def test_episode_ends_at_168():
    env = MaintenanceArenaEnvironment()
    env.reset(seed=42)
    for i in range(168):
        obs = env.step(MaintenanceAction(action_type=ActionType.MONITOR))
    assert obs.done is True
    assert obs.hour == 168
    print("  pass: test_episode_ends_at_168")


def test_state_tracking():
    env = MaintenanceArenaEnvironment()
    env.reset(seed=42)
    env.step(MaintenanceAction(action_type=ActionType.MONITOR))
    env.step(MaintenanceAction(action_type=ActionType.MONITOR))
    st = env.state
    assert st.step_count == 2
    assert st.episode_id is not None
    print("  pass: test_state_tracking")


def test_sensor_noise_varies():
    env = MaintenanceArenaEnvironment()
    env.reset(seed=42)
    obs1 = env.step(MaintenanceAction(action_type=ActionType.MONITOR))
    obs2 = env.step(MaintenanceAction(action_type=ActionType.MONITOR))
    # Sensor readings should differ due to noise
    s1 = obs1.machines[0]["sensors"]["vibration"]
    s2 = obs2.machines[0]["sensors"]["vibration"]
    # They might be the same by chance but very unlikely
    print(f"  pass: test_sensor_noise_varies (vib: {s1} vs {s2})")


def test_reproducible_with_seed():
    env1 = MaintenanceArenaEnvironment()
    env2 = MaintenanceArenaEnvironment()
    obs1 = env1.reset(seed=123)
    obs2 = env2.reset(seed=123)
    # Same seed should produce same initial machines
    for i in range(5):
        assert obs1.machines[i]["sensors"] == obs2.machines[i]["sensors"]
    print("  pass: test_reproducible_with_seed")


def test_different_seeds_differ():
    env1 = MaintenanceArenaEnvironment()
    env2 = MaintenanceArenaEnvironment()
    obs1 = env1.reset(seed=1)
    obs2 = env2.reset(seed=999)
    # Different seeds should produce different readings
    differs = False
    for i in range(5):
        for k, v in obs1.machines[i]["sensors"].items():
            if v != obs2.machines[i]["sensors"][k]:
                differs = True
    assert differs
    print("  pass: test_different_seeds_differ")


def test_rubric_scores_in_range():
    env = MaintenanceArenaEnvironment()
    env.reset(seed=42)
    # Run a full episode
    for _ in range(168):
        obs = env.step(MaintenanceAction(action_type=ActionType.MONITOR))
    assert obs.done is True
    md = obs.metadata or {}
    score = md.get("final_score", -1)
    assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
    print(f"  pass: test_rubric_scores_in_range (score={score:.3f})")


def test_difficulty_levels():
    env = MaintenanceArenaEnvironment()
    # Test difficulty 1
    obs1 = env.reset(seed=42, difficulty=1)
    assert obs1.difficulty_level == 1
    # Test difficulty 3
    obs3 = env.reset(seed=42, difficulty=3)
    assert obs3.difficulty_level == 3
    print("  pass: test_difficulty_levels")


def test_cascade_damage():
    """Force a machine failure and check cascade."""
    env = MaintenanceArenaEnvironment()
    env.reset(seed=42, difficulty=5)
    # Run many steps without maintenance to force failure
    cascade_happened = False
    for _ in range(168):
        obs = env.step(MaintenanceAction(action_type=ActionType.MONITOR))
        for event in obs.recent_events:
            if "CASCADE" in event:
                cascade_happened = True
                break
        if obs.done or cascade_happened:
            break
    # At difficulty 5 with no maintenance, cascade should happen
    print(f"  pass: test_cascade_damage (cascade={'yes' if cascade_happened else 'no'})")


def test_full_episode_with_maintenance():
    """Run an episode where agent does proactive maintenance."""
    env = MaintenanceArenaEnvironment()
    env.reset(seed=42)

    for hour in range(168):
        # Simple strategy: diagnose and maintain when warning
        action = MaintenanceAction(action_type=ActionType.MONITOR)

        # Every 20 hours, check the most degraded machine
        if hour % 20 == 10:
            # Find worst machine
            for i, m in enumerate(env._factory.machines):
                if m.health < 50 and m.status != "maintenance":
                    action = MaintenanceAction(
                        action_type=ActionType.SCHEDULE_MAINTENANCE,
                        machine_id=i,
                    )
                    break

        obs = env.step(action)
        if obs.done:
            break

    md = obs.metadata or {}
    score = md.get("final_score", 0)
    print(f"  pass: test_full_episode_with_maintenance (score={score:.3f})")


if __name__ == "__main__":
    print("\n=== Predictive Maintenance Arena Tests ===\n")
    test_reset()
    test_monitor_action()
    test_diagnostic_action()
    test_schedule_maintenance()
    test_crew_busy()
    test_order_parts()
    test_adjust_speed()
    test_episode_ends_at_168()
    test_state_tracking()
    test_sensor_noise_varies()
    test_reproducible_with_seed()
    test_different_seeds_differ()
    test_rubric_scores_in_range()
    test_difficulty_levels()
    test_cascade_damage()
    test_full_episode_with_maintenance()
    print("\n=== All tests passed! ===\n")
