#!/usr/bin/env python3
"""
Demonstrates two key environment features:
  1. Adaptive difficulty 
  2. Cascade dependency mechanics — shows how one failure propagates

No GPU needed. 


OUTPUT:
    adaptive_difficulty.png  — difficulty escalation 
    cascade_demo.png         — cascade failure propagation visualization
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from server.environment import MaintenanceArenaEnvironment
from models import MaintenanceAction, ActionType


# ============================================================================
# PART 1: Adaptive Difficulty — 30 episodes, force escalation through all levels
# ============================================================================

def run_smart_episode(env, seed):
    """Run one 168-step episode with smart baseline."""
    obs = env.reset(seed=seed)
    rewards = []
    actions_taken = {"monitor": 0, "run_diagnostic": 0, "schedule_maintenance": 0, "other": 0}

    for step in range(168):
        at = ActionType.MONITOR
        mid = -1
        for i, m in enumerate(obs.machines):
            if isinstance(m, dict):
                if m.get("status") == "critical" and not m.get("is_under_maintenance"):
                    at = ActionType.SCHEDULE_MAINTENANCE
                    mid = i
                    break
                elif m.get("status") == "warning":
                    at = ActionType.RUN_DIAGNOSTIC
                    mid = i
                    break

        action = MaintenanceAction(action_type=at, machine_id=mid, parameters={})
        obs = env.step(action)
        rewards.append(obs.reward)
        actions_taken[at.value] = actions_taken.get(at.value, 0) + 1
        if obs.done:
            break

    total = sum(rewards)
    score = obs.metadata.get("final_score", None) if obs.metadata else None
    difficulty = env._difficulty_level
    return total, score, difficulty, actions_taken, rewards


def demo_adaptive_difficulty():
    print("=" * 60)
    print("  DEMO 1: Adaptive Difficulty System")
    print("  30 episodes, same smart baseline, same env instance")
    print("  Difficulty adjusts based on rolling 3-episode average")
    print("  Threshold: >0.7 → harder, <0.3 → easier")
    print("=" * 60)

    env = MaintenanceArenaEnvironment()

    episodes = 30
    ep_rewards = []
    ep_scores = []
    ep_difficulties = []
    ep_maintenance_count = []
    ep_all_rewards = []

    for ep in range(episodes):
        seed = ep * 13 + 7
        total, score, difficulty, actions, rewards = run_smart_episode(env, seed)
        ep_rewards.append(total)
        ep_scores.append(score if score else 0)
        ep_difficulties.append(difficulty)
        ep_maintenance_count.append(actions.get("schedule_maintenance", 0))
        ep_all_rewards.append(rewards)
        print(f"  Ep {ep+1:2d} | reward={total:7.1f} | rubric={score:.4f} | "
              f"difficulty={difficulty} | maintenance={actions.get('schedule_maintenance', 0)}")

    # ── Plot: 3-panel adaptive difficulty visualization ────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [2, 1, 1.5]})

    x = range(1, episodes + 1)

    # Panel 1: Rubric score + difficulty level (dual axis)
    color_score = "#378ADD"
    color_diff = "#E24B4A"

    ax1 = axes[0]
    ax2 = ax1.twinx()

    ax1.plot(x, ep_scores, color=color_score, linewidth=2, marker="o", markersize=4,
             label="Rubric Score", zorder=3)
    ax1.fill_between(x, ep_scores, alpha=0.1, color=color_score)
    ax1.axhline(y=0.7, color="#1D9E75", linestyle="--", linewidth=1.5, alpha=0.7,
                label="Difficulty-up threshold (0.7)")
    ax1.axhline(y=0.3, color="#BA7517", linestyle="--", linewidth=1.5, alpha=0.7,
                label="Difficulty-down threshold (0.3)")
    ax1.set_ylabel("Rubric Score", fontsize=11, color=color_score)
    ax1.tick_params(axis="y", labelcolor=color_score)
    ax1.set_ylim(0, 1.1)

    ax2.step(x, ep_difficulties, color=color_diff, linewidth=3, where="mid",
             label="Difficulty Level", zorder=2)
    ax2.fill_between(x, ep_difficulties, step="mid", alpha=0.08, color=color_diff)
    ax2.set_ylabel("Difficulty Level", fontsize=11, color=color_diff)
    ax2.tick_params(axis="y", labelcolor=color_diff)
    ax2.set_ylim(0.5, 5.5)
    ax2.set_yticks([1, 2, 3, 4, 5])
    ax2.set_yticklabels(["1\nBasic", "2\nNoisy", "3\nCorrelated", "4\nAdversarial", "5\nNovel"])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=9,
               framealpha=0.9)

    ax1.set_title("Adaptive Difficulty — Environment Co-evolves With Agent Performance",
                  fontsize=13, fontweight="bold", pad=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("")

    # Panel 2: Maintenance actions per episode (shows agent adapting)
    colors_by_diff = {1: "#1D9E75", 2: "#378ADD", 3: "#BA7517", 4: "#E24B4A", 5: "#7F77DD"}
    bar_colors = [colors_by_diff.get(d, "#999") for d in ep_difficulties]
    axes[1].bar(x, ep_maintenance_count, color=bar_colors, width=0.7,
                edgecolor="white", linewidth=0.5)
    axes[1].set_ylabel("Maintenance Actions", fontsize=11)
    axes[1].set_title("Maintenance Actions Per Episode (colored by difficulty level)",
                      fontsize=11, fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y")

    legend_patches = [mpatches.Patch(facecolor=colors_by_diff[d], label=f"Level {d}")
                      for d in sorted(set(ep_difficulties))]
    axes[1].legend(handles=legend_patches, loc="upper right", fontsize=9)

    # Panel 3: Reward heatmap across episodes and steps
    # Pad rewards to same length
    max_len = max(len(r) for r in ep_all_rewards)
    reward_matrix = []
    for r in ep_all_rewards:
        padded = r + [0.0] * (max_len - len(r))
        reward_matrix.append(padded)
    reward_matrix = list(reversed(reward_matrix))  # Episode 1 at top

    im = axes[2].imshow(reward_matrix, aspect="auto", cmap="RdYlGn",
                        extent=[0, max_len, 0.5, episodes + 0.5],
                        vmin=-2, vmax=1.5)
    axes[2].set_xlabel("Step (simulated hour)", fontsize=11)
    axes[2].set_ylabel("Episode", fontsize=11)
    axes[2].set_title("Per-Step Reward Heatmap (green = good, red = failure/penalty)",
                      fontsize=11, fontweight="bold")
    cbar = plt.colorbar(im, ax=axes[2], shrink=0.8)
    cbar.set_label("Step Reward", fontsize=10)

    plt.tight_layout()
    plt.savefig("adaptive_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: adaptive_difficulty.png")

    # Print summary
    print(f"\n  Difficulty progression: {ep_difficulties}")
    max_diff = max(ep_difficulties)
    if max_diff > 1:
        first_change = next(i for i, d in enumerate(ep_difficulties) if d > 1) + 1
        print(f"  First difficulty increase at episode {first_change}")
        print(f"  Reached max difficulty level {max_diff}")
    else:
        print(f"  Stayed at level 1 (agent did not consistently exceed 0.7 threshold)")

    return ep_scores, ep_difficulties


# ============================================================================
# PART 2: Cascade Dependency Demo — show exactly how failures propagate
# ============================================================================

def demo_cascade():
    print(f"\n{'=' * 60}")
    print("  DEMO 2: Cascade Dependency Mechanics")
    print("  Run factory until a failure occurs, track the cascade")
    print("=" * 60)

    from machines import FactorySimulator, DEPENDENCIES

    # Run multiple seeds to find episodes with cascades
    cascade_episodes = []

    for seed in range(50):
        sim = FactorySimulator(seed=seed)
        sim.reset(seed=seed, difficulty=2)  # Level 2 for more failures

        health_history = {i: [] for i in range(5)}
        status_history = {i: [] for i in range(5)}
        events_by_step = []
        cascade_steps = []

        for step in range(168):
            # Record health BEFORE step
            for i, m in enumerate(sim.machines):
                health_history[i].append(m.health)
                status_history[i].append(m.status)

            result = sim.step()
            events_by_step.append(result["events"])

            # Check for cascade events
            cascade_events = [e for e in result["events"] if "CASCADE" in e]
            if cascade_events:
                cascade_steps.append(step)

        if cascade_steps:
            # Record final health
            for i, m in enumerate(sim.machines):
                health_history[i].append(m.health)
                status_history[i].append(m.status)

            cascade_episodes.append({
                "seed": seed,
                "health_history": health_history,
                "status_history": status_history,
                "events": events_by_step,
                "cascade_steps": cascade_steps,
                "cascade_count": sim.cascade_failures,
            })
            if len(cascade_episodes) >= 3:
                break

    if not cascade_episodes:
        print("  No cascades found in 50 seeds. Trying higher difficulty...")
        return

    # Pick the most dramatic cascade episode
    best = max(cascade_episodes, key=lambda e: e["cascade_count"])
    print(f"  Using seed {best['seed']} — {best['cascade_count']} cascade events")
    print(f"  Cascade steps: {best['cascade_steps']}")

    # Print cascade event details
    for step in best['cascade_steps'][:5]:
        events = best['events'][step]
        for e in events:
            if "CASCADE" in e or "FAILURE" in e:
                print(f"    Step {step}: {e}")

    # ── Plot: Machine health over time with cascade markers ──────────
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [3, 1]})

    machine_names = ["Conveyor", "CNC Mill", "Oven", "Press", "Packaging"]
    machine_colors = ["#1D9E75", "#378ADD", "#E24B4A", "#BA7517", "#7F77DD"]

    # Panel 1: Health curves
    for i in range(5):
        h = best["health_history"][i]
        axes[0].plot(range(len(h)), h, color=machine_colors[i], linewidth=2,
                     label=machine_names[i], alpha=0.9)

    # Mark cascade events
    for step in best["cascade_steps"]:
        axes[0].axvline(x=step, color="#E24B4A", linestyle=":", linewidth=1, alpha=0.5)

    # Mark the first cascade with annotation
    if best["cascade_steps"]:
        first_cascade = best["cascade_steps"][0]
        axes[0].axvline(x=first_cascade, color="#E24B4A", linestyle="-", linewidth=2, alpha=0.8)
        axes[0].annotate("Cascade begins",
                        xy=(first_cascade, 50), xytext=(first_cascade + 15, 75),
                        fontsize=10, fontweight="bold", color="#E24B4A",
                        arrowprops=dict(arrowstyle="->", color="#E24B4A", linewidth=1.5))

    # Health thresholds
    axes[0].axhline(y=60, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
    axes[0].axhline(y=30, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
    axes[0].axhline(y=15, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)

    axes[0].text(170, 62, "good", fontsize=8, color="gray")
    axes[0].text(170, 32, "warn", fontsize=8, color="gray")
    axes[0].text(170, 17, "crit", fontsize=8, color="gray")

    axes[0].set_ylabel("Machine Health (hidden from agent)", fontsize=11)
    axes[0].set_title(f"Cascade Failure Propagation — Seed {best['seed']}, Difficulty Level 2\n"
                      f"{best['cascade_count']} cascade events observed",
                      fontsize=13, fontweight="bold")
    axes[0].legend(loc="lower left", fontsize=10, ncol=5)
    axes[0].set_ylim(-5, 105)
    axes[0].set_xlim(0, 175)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Dependency graph visualization
    axes[1].set_xlim(-0.5, 5.5)
    axes[1].set_ylim(-1.5, 2)
    axes[1].axis("off")
    axes[1].set_title("Machine Dependency Graph — Failures Cascade Along These Links",
                      fontsize=12, fontweight="bold", pad=10)

    # Draw machines as boxes
    positions = [(0.5, 0.5), (1.5, 0.5), (2.5, 0.5), (3.5, 0.5), (4.5, 0.5)]
    for i, (px, py) in enumerate(positions):
        final_health = best["health_history"][i][-1]
        if final_health <= 0:
            fc = "#E24B4A"
            status = "FAILED"
        elif final_health < 30:
            fc = "#BA7517"
            status = f"CRIT ({final_health:.0f})"
        elif final_health < 60:
            fc = "#FFD700"
            status = f"WARN ({final_health:.0f})"
        else:
            fc = "#1D9E75"
            status = f"OK ({final_health:.0f})"

        rect = plt.Rectangle((px - 0.35, py - 0.35), 0.7, 0.7, linewidth=2,
                             edgecolor="black", facecolor=fc, alpha=0.8)
        axes[1].add_patch(rect)
        axes[1].text(px, py + 0.05, machine_names[i], ha="center", va="center",
                    fontsize=8, fontweight="bold", color="white")
        axes[1].text(px, py - 0.15, status, ha="center", va="center",
                    fontsize=7, color="white")

    # Draw forward arrows
    for i in range(4):
        axes[1].annotate("", xy=(positions[i+1][0] - 0.35, 0.5),
                        xytext=(positions[i][0] + 0.35, 0.5),
                        arrowprops=dict(arrowstyle="->", color="black", linewidth=1.5))

    # Draw backward arrows (oven ← press, CNC ← oven)
    # Oven (2) ← Press (3)
    axes[1].annotate("", xy=(positions[2][0] + 0.2, 0.15),
                    xytext=(positions[3][0] - 0.2, 0.15),
                    arrowprops=dict(arrowstyle="->", color="#E24B4A", linewidth=1.5,
                                   linestyle="dashed"))
    axes[1].text(3.0, -0.05, "vibration\n(1.1x)", ha="center", fontsize=7, color="#E24B4A")

    # CNC (1) ← Oven (2)
    axes[1].annotate("", xy=(positions[1][0] + 0.2, 0.15),
                    xytext=(positions[2][0] - 0.2, 0.15),
                    arrowprops=dict(arrowstyle="->", color="#E24B4A", linewidth=1.5,
                                   linestyle="dashed"))
    axes[1].text(2.0, -0.05, "heat\n(1.15x)", ha="center", fontsize=7, color="#E24B4A")

    # Labels
    axes[1].text(2.5, -1.0,
                "Solid arrows: forward production flow (downstream starvation)\n"
                "Dashed red arrows: backward physical effects (heat, vibration)",
                ha="center", fontsize=9, style="italic", color="gray")

    plt.tight_layout()
    plt.savefig("cascade_demo.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: cascade_demo.png")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    demo_adaptive_difficulty()
    demo_cascade()
    print(f"\n{'=' * 60}")
    print("  Both demos complete.")
    print("  Files: adaptive_difficulty.png, cascade_demo.png")
    print(f"{'=' * 60}")
