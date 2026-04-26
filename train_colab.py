#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# 🏭 Predictive Maintenance Arena — GRPO Training Notebook
# Meta PyTorch × HuggingFace OpenEnv Hackathon, April 2026
# Author: Shubhan Kamat (https://linkedin.com/in/shubhankamat)
#
# This notebook trains an LLM agent to manage a factory of 5 interconnected
# machines using GRPO (Group Relative Policy Optimization).
#
# Environment: https://huggingface.co/spaces/ShubhanKamat/pred-maint-arena
# 
#
# HOW TO RUN IN COLAB:
#   1. Runtime → Change runtime type → GPU (A100 recommended)
#   2. Run all cells in order
#   3. Plots are saved and downloadable at the end
"""

# ============================================================================
# Install dependencies and clone environment
# ============================================================================

import subprocess
import sys

print("Installing dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "openenv-core>=0.2.0", "fastapi>=0.104.0", "uvicorn", "pydantic>=2.5.0",
    "websockets>=12.0", "unsloth", "torch", "transformers", "peft",
    "accelerate", "matplotlib", "bitsandbytes"])

print("Cloning environment...")
subprocess.run(["rm", "-rf", "pred-maint-arena"], capture_output=True)
subprocess.check_call(["git", "clone",
    "https://huggingface.co/spaces/ShubhanKamat/pred-maint-arena",
    "pred-maint-arena"])

import os
os.chdir("pred-maint-arena")
sys.path.insert(0, ".")

print("✅ Setup complete!")

# ============================================================================
# Import environment and verify it works
# ============================================================================

from server.environment import MaintenanceArenaEnvironment
from models import MaintenanceAction, ActionType

env = MaintenanceArenaEnvironment()
obs = env.reset(seed=42)
print(f"✅ Environment loaded!")
print(f"   Machines: {len(obs.machines)}")
print(f"   Max hours: {obs.max_hours}")
print(f"   Production target: {obs.production_target}")

# Quick step test
action = MaintenanceAction(action_type=ActionType.MONITOR, machine_id=-1, parameters={})
obs = env.step(action)
print(f"   Step reward: {obs.reward} (should NOT be 1.0)")
print(f"   Hour: {obs.hour}")

# ============================================================================
# Run baselines (passive, random, smart) — 168-step full weeks
# ============================================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import random
import time

MAX_STEPS = 168
SEEDS = [42, 99, 123, 256, 777]

def run_baseline(env, strategy, seed):
    """Run one 168-step episode with a given strategy. Returns (rewards, total, rubric_score)."""
    obs = env.reset(seed=seed)
    rewards = []
    if strategy == "random":
        random.seed(seed)
    actions = [ActionType.MONITOR, ActionType.RUN_DIAGNOSTIC,
               ActionType.SCHEDULE_MAINTENANCE, ActionType.EMERGENCY_SHUTDOWN,
               ActionType.ORDER_PARTS, ActionType.ADJUST_SPEED]

    for step in range(MAX_STEPS):
        at = ActionType.MONITOR
        mid = -1
        params = {}

        if strategy == "passive":
            pass  # always monitor

        elif strategy == "smart":
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

        elif strategy == "random":
            at = random.choice(actions)
            mid = random.randint(0, 4) if at != ActionType.MONITOR else -1
            if at == ActionType.ADJUST_SPEED:
                params = {"speed_pct": random.choice([60, 70, 80, 90, 100])}

        action = MaintenanceAction(action_type=at, machine_id=mid, parameters=params)
        obs = env.step(action)
        rewards.append(obs.reward)
        if obs.done:
            break

    total = sum(rewards)
    score = obs.metadata.get("final_score", None) if obs.metadata else None
    return rewards, total, score


print("Running baselines (3 strategies × 5 seeds = 15 episodes)...")
print("Each episode = 168 steps (one full simulated week)\n")

results = {}
for strategy in ["passive", "random", "smart"]:
    totals = []
    scores = []
    all_rewards = []
    for seed in SEEDS:
        rewards, total, score = run_baseline(env, strategy, seed)
        totals.append(total)
        scores.append(score)
        all_rewards.append(rewards)
        print(f"  {strategy:8s} seed={seed}: reward={total:7.1f}  rubric={score}")

    avg = sum(totals) / len(totals)
    clean_scores = [s for s in scores if s is not None]
    avg_score = sum(clean_scores) / len(clean_scores) if clean_scores else None
    results[strategy] = {
        "totals": totals, "avg": avg, "scores": scores, "avg_score": avg_score,
        "all_rewards": all_rewards,
    }
    print(f"  {'':8s} AVG: reward={avg:7.1f}  rubric={avg_score}\n")

# ── Baseline plots ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: reward per step (first seed)
for strat, color in [("passive", "#E24B4A"), ("random", "#BA7517"), ("smart", "#1D9E75")]:
    r = results[strat]["all_rewards"][0]
    axes[0].plot(r, label=f"{strat.title()} (total={results[strat]['totals'][0]:.0f})",
                 color=color, alpha=0.8, linewidth=1.2)
axes[0].axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
axes[0].set_xlabel("Step (simulated hour)")
axes[0].set_ylabel("Reward per step")
axes[0].set_title("Baseline Comparison — One Full Week (168 hours)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Right: bar chart
avgs = [results[s]["avg"] for s in ["passive", "random", "smart"]]
stds = [
    (sum((x - results[s]["avg"])**2 for x in results[s]["totals"]) / len(SEEDS)) ** 0.5
    for s in ["passive", "random", "smart"]
]
colors = ["#E24B4A", "#BA7517", "#1D9E75"]
bars = axes[1].bar(["Passive", "Random", "Smart"], avgs, color=colors, width=0.4,
                   yerr=stds, capsize=5, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, avgs):
    y = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2, y + max(abs(y)*0.03, 1),
                 f"{val:.1f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Total Episode Reward (avg ± std, 5 seeds)")
axes[1].set_title("Baseline Total Rewards — 168-Step Full Week")
axes[1].axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
axes[1].grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("baseline_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: baseline_comparison.png")

# Rubric scores plot
p_s = results["passive"]["avg_score"]
r_s = results["random"]["avg_score"]
s_s = results["smart"]["avg_score"]
if p_s is not None and s_s is not None:
    fig, ax = plt.subplots(figsize=(8, 5))
    rubric_vals = [p_s, r_s or 0, s_s]
    bars = ax.bar(["Passive", "Random", "Smart"], rubric_vals,
                  color=["#E24B4A", "#BA7517", "#1D9E75"], width=0.4, edgecolor="white")
    for bar, val in zip(bars, rubric_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.set_ylabel("Final Rubric Score (0.0 - 1.0)")
    ax.set_title("Baseline Rubric Scores (avg over 5 seeds)\n"
                 "(Uptime 40% + Cost 25% + Prediction 20% + Cascade 15%)")
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("rubric_scores.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: rubric_scores.png")

# ============================================================================
# Load model with Unsloth (4-bit + LoRA)
# ============================================================================

import copy
import textwrap
import re
import torch

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = textwrap.dedent("""\
You are an AI maintenance engineer managing a factory with 5 machines.
Each step = 1 hour. You see sensor readings and decide what to do.

MACHINES: 0:Conveyor 1:CNC 2:Oven 3:Press 4:Packaging
Connected: 0->1->2->3->4. Failure cascades downstream.

ACTIONS (respond with JSON only):
{"action_type":"monitor","machine_id":-1,"parameters":{}}
{"action_type":"run_diagnostic","machine_id":N,"parameters":{}}
{"action_type":"schedule_maintenance","machine_id":N,"parameters":{}}
{"action_type":"emergency_shutdown","machine_id":N,"parameters":{}}
{"action_type":"order_parts","machine_id":N,"parameters":{}}
{"action_type":"adjust_speed","machine_id":-1,"parameters":{"speed_pct":80}}

RULES:
- Monitor when sensors look normal
- Diagnose machines with elevated readings
- Maintain BEFORE failure (health_bar: good=fine, warn=investigate, crit=act now)
- One crew only — prioritize the most critical machine

Respond with ONLY one JSON object. No explanation.
""").strip()

print(f"Loading {MODEL_NAME} with 4-bit quantization + LoRA...")
t0 = time.time()

from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME, max_seq_length=1024,
    dtype=None, load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16, lora_dropout=0, bias="none",
    use_gradient_checkpointing="unsloth",
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"✅ Model loaded in {time.time()-t0:.1f}s")

# ============================================================================
# Helper functions
# ============================================================================

def format_obs(obs_dict):
    """Format observation dict as text prompt for the LLM."""
    machines = obs_dict.get("machines", [])
    lines = [f"HOUR {obs_dict.get('hour', 0)}/{obs_dict.get('max_hours', 168)} | "
             f"Prod: {obs_dict.get('units_produced', 0)}/{obs_dict.get('production_target', 800)} | "
             f"Crew: {'free' if obs_dict.get('crew_available', True) else 'busy'}"]
    for i, m in enumerate(machines):
        if isinstance(m, dict):
            s = m.get("sensors", {})
            sv = ", ".join(f"{k}={v}" for k, v in s.items())
            flag = " !!" if m.get("status") in ("warning", "critical") else ""
            lines.append(f"M{i} {m.get('name', '?')} [{m.get('health_bar', '?')}]{flag}: {sv}")
    return "\n".join(lines)


def parse_action(text):
    """Parse LLM output into action dict. Falls back to monitor."""
    cleaned = text.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        if len(parts) >= 2:
            cleaned = parts[1].lstrip("json").strip()
    try:
        r = json.loads(cleaned)
        if "action_type" in r:
            return r
    except:
        pass
    match = re.search(r'\{[^{}]*"action_type"[^{}]*\}', cleaned)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    return {"action_type": "monitor", "machine_id": -1, "parameters": {}}


def obs_to_dict(obs):
    """Convert MaintenanceObservation to dict."""
    return {
        "machines": obs.machines if isinstance(obs.machines, list) else [],
        "hour": getattr(obs, "hour", 0),
        "max_hours": getattr(obs, "max_hours", 168),
        "units_produced": getattr(obs, "units_produced", 0),
        "production_target": getattr(obs, "production_target", 800),
        "crew_available": getattr(obs, "crew_available", True),
    }


def do_step(env, action_type_str, machine_id=-1, parameters=None):
    """Take a step using direct Python. Returns (obs_dict, reward, done)."""
    valid = ["monitor", "run_diagnostic", "schedule_maintenance",
             "emergency_shutdown", "order_parts", "adjust_speed"]
    if action_type_str not in valid:
        action_type_str = "monitor"
        machine_id = -1
    if action_type_str in ("run_diagnostic", "schedule_maintenance",
                           "emergency_shutdown", "order_parts"):
        try:
            machine_id = max(0, min(4, int(machine_id)))
        except (ValueError, TypeError):
            machine_id = 0

    action = MaintenanceAction(
        action_type=ActionType(action_type_str),
        machine_id=machine_id,
        parameters=parameters or {},
    )
    obs = env.step(action)
    return obs_to_dict(obs), obs.reward, obs.done


print("✅ Helper functions ready")

# ============================================================================
# GRPO Training Loop
# ============================================================================

NUM_EPISODES = 26
TRAIN_STEPS = 168  # Full week per episode
NUM_GENERATIONS = 3
LR = 2e-5

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
model.train()

episode_rewards = []
episode_scores = []
episode_difficulties = []
train_start = time.time()

print(f"\n{'='*60}")
print(f"  GRPO Training — {NUM_EPISODES} episodes × {TRAIN_STEPS} steps")
print(f"  Model: {MODEL_NAME} (4-bit + LoRA)")
print(f"  Generations per step: {NUM_GENERATIONS}")
print(f"  Direct Python environment — real rewards")
print(f"  Adaptive difficulty enabled (persists across episodes)")
print(f"{'='*60}\n")

# FIX #7: Create ONE environment instance — difficulty history persists across episodes
train_env = MaintenanceArenaEnvironment()

for ep in range(NUM_EPISODES):
    ep_start = time.time()
    seed = ep * 7 + 42

    # Reset the SAME environment (difficulty carries over)
    obs_obj = train_env.reset(seed=seed)
    obs_dict = obs_to_dict(obs_obj)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_obs(obs_dict)},
    ]

    ep_reward = 0.0
    ep_steps = 0

    for step in range(TRAIN_STEPS):
        # Keep context manageable — system + last 4 turns
        recent_messages = [messages[0]]
        recent_messages += messages[max(1, len(messages) - 8):]

        prompt = tokenizer.apply_chat_template(
            recent_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        # Generate all candidates in ONE batched forward pass (~3x faster)
        candidates = []
        candidate_rewards = []

        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=100,
                temperature=0.8, do_sample=True,
                num_return_sequences=NUM_GENERATIONS,
                pad_token_id=tokenizer.pad_token_id,
            )

        for g in range(NUM_GENERATIONS):
            gen_text = tokenizer.decode(out[g][prompt_len:], skip_special_tokens=True)
            action = parse_action(gen_text)
            candidates.append((gen_text, action, out[g]))

            # Evaluate on a copy of current state
            eval_env = copy.deepcopy(train_env)
            _, reward, _ = do_step(eval_env,
                                   action.get("action_type", "monitor"),
                                   action.get("machine_id", -1),
                                   action.get("parameters", {}))
            candidate_rewards.append(reward)

        # GRPO update
        best_idx = candidate_rewards.index(max(candidate_rewards))
        mean_reward = sum(candidate_rewards) / len(candidate_rewards)
        advantage = candidate_rewards[best_idx] - mean_reward

        if advantage > 0.01:
            best_text = candidates[best_idx][0]
            train_text = prompt + best_text
            train_inputs = tokenizer(train_text, return_tensors="pt",
                                     truncation=True, max_length=868)
            train_inputs = {k: v.to(model.device) for k, v in train_inputs.items()}
            labels = train_inputs["input_ids"].clone()
            labels[0, :prompt_len] = -100

            loss = model(**train_inputs, labels=labels).loss
            scaled_loss = loss * advantage
            scaled_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Continue episode with best action
        best_action = candidates[best_idx][1]
        obs_dict, reward, done = do_step(train_env,
                                         best_action.get("action_type", "monitor"),
                                         best_action.get("machine_id", -1),
                                         best_action.get("parameters", {}))
        ep_reward += reward
        ep_steps += 1

        messages.append({"role": "assistant", "content": candidates[best_idx][0]})
        if done:
            break
        messages.append({"role": "user", "content": format_obs(obs_dict)})

    ep_time = time.time() - ep_start
    avg_score = ep_reward / max(ep_steps, 1)
    episode_rewards.append(ep_reward)
    episode_scores.append(avg_score)
    episode_difficulties.append(train_env._difficulty_level)

    elapsed = time.time() - train_start
    eta = (elapsed / (ep + 1)) * (NUM_EPISODES - ep - 1)
    recent = sum(episode_scores[max(0, ep - 4):ep + 1]) / min(5, ep + 1)
    print(f"  Ep {ep+1:2d}/{NUM_EPISODES} | reward={ep_reward:7.1f} | "
          f"avg/step={avg_score:.3f} | recent={recent:.3f} | "
          f"diff={train_env._difficulty_level} | "
          f"time={ep_time:.0f}s | ETA={eta/60:.0f}min")

total_time = time.time() - train_start
print(f"\n✅ Training complete in {total_time/60:.1f} minutes")

# ============================================================================
# Generate training curve plot
# ============================================================================

passive_baseline = results["passive"]["avg"]
smart_baseline = results["smart"]["avg"]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: total episode reward
axes[0].plot(episode_rewards, alpha=0.3, color="#378ADD", linewidth=0.8, label="Per-episode")
w = min(5, len(episode_rewards) // 3 + 1)
if w > 1 and len(episode_rewards) > w:
    sm = [sum(episode_rewards[max(0, i-w+1):i+1]) / (i - max(0, i-w+1) + 1)
          for i in range(len(episode_rewards))]
    axes[0].plot(sm, color="#378ADD", linewidth=2.5, label="Smoothed (moving avg)")
axes[0].axhline(y=passive_baseline, color="#E24B4A", linestyle="--",
                linewidth=1.5, label=f"Passive baseline ({passive_baseline:.0f})")
axes[0].axhline(y=smart_baseline, color="#1D9E75", linestyle="--",
                linewidth=1.5, label=f"Smart baseline ({smart_baseline:.0f})")
axes[0].set_xlabel("Episode", fontsize=11)
axes[0].set_ylabel("Total Episode Reward", fontsize=11)
axes[0].set_title(f"GRPO Training — Episode Rewards ({TRAIN_STEPS}-step episodes)", fontsize=12, fontweight="bold")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Right: avg reward per step
axes[1].plot(episode_scores, alpha=0.3, color="#1D9E75", linewidth=0.8, label="Per-episode")
if w > 1 and len(episode_scores) > w:
    sm2 = [sum(episode_scores[max(0, i-w+1):i+1]) / (i - max(0, i-w+1) + 1)
           for i in range(len(episode_scores))]
    axes[1].plot(sm2, color="#1D9E75", linewidth=2.5, label="Smoothed (moving avg)")
axes[1].set_xlabel("Episode", fontsize=11)
axes[1].set_ylabel("Avg Reward per Step", fontsize=11)
axes[1].set_title("GRPO Training — Per-Step Improvement", fontsize=12, fontweight="bold")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: training_curve.png")

# ============================================================================
# Additional plots for presentation
# ============================================================================

# ── Plot: Adaptive Difficulty during training ─────────────────────────
fig, ax1 = plt.subplots(figsize=(14, 5))
ax2 = ax1.twinx()

x = range(1, len(episode_rewards) + 1)
color_score = "#378ADD"
color_diff = "#E24B4A"

ax1.plot(x, episode_scores, color=color_score, linewidth=2, marker="o", markersize=4,
         label="Avg Reward/Step", zorder=3)
ax1.fill_between(x, episode_scores, alpha=0.1, color=color_score)
ax1.axhline(y=0.7, color="#1D9E75", linestyle="--", linewidth=1.5, alpha=0.7,
            label="Difficulty-up threshold (0.7)")
ax1.axhline(y=0.3, color="#BA7517", linestyle="--", linewidth=1.5, alpha=0.7,
            label="Difficulty-down threshold (0.3)")
ax1.set_xlabel("Episode", fontsize=11)
ax1.set_ylabel("Avg Reward per Step", fontsize=11, color=color_score)
ax1.tick_params(axis="y", labelcolor=color_score)
ax1.set_ylim(0, 1.2)

ax2.step(x, episode_difficulties, color=color_diff, linewidth=3, where="mid",
         label="Difficulty Level", zorder=2)
ax2.fill_between(x, episode_difficulties, step="mid", alpha=0.08, color=color_diff)
ax2.set_ylabel("Difficulty Level", fontsize=11, color=color_diff)
ax2.tick_params(axis="y", labelcolor=color_diff)
ax2.set_ylim(0.5, 5.5)
ax2.set_yticks([1, 2, 3, 4, 5])
ax2.set_yticklabels(["1 Basic", "2 Noisy", "3 Correlated", "4 Adversarial", "5 Novel"])

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=9, framealpha=0.9)

ax1.set_title("Adaptive Difficulty During GRPO Training — Environment Co-evolves With Agent",
              fontsize=13, fontweight="bold", pad=10)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("adaptive_difficulty_training.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: adaptive_difficulty_training.png")
print(f"  Difficulty progression: {episode_difficulties}")

# ── Plot: Trained agent vs all baselines (bar chart) ─────────────────
final_5_avg_reward = sum(episode_rewards[-5:]) / min(5, len(episode_rewards))
best_ep_reward = max(episode_rewards)

fig, ax = plt.subplots(figsize=(10, 6))
agents = ["Passive\n(do nothing)", "Random\n(random)", "Smart\n(rule-based)",
          "Trained\n(GRPO, last 5 avg)", "Trained\n(best episode)"]
values = [results["passive"]["avg"], results["random"]["avg"],
          results["smart"]["avg"], final_5_avg_reward, best_ep_reward]
colors = ["#E24B4A", "#BA7517", "#1D9E75", "#378ADD", "#2855A1"]
bars = ax.bar(agents, values, color=colors, width=0.5, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, values):
    y = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, y + max(abs(y)*0.02, 1),
            f"{val:.1f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.set_ylabel(f"Total Episode Reward ({TRAIN_STEPS} steps)", fontsize=11)
ax.set_title("Trained Agent vs Baselines — Total Episode Reward", fontsize=13, fontweight="bold")
ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("trained_vs_baselines.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: trained_vs_baselines.png")

# ── Plot: Reward distribution across episodes ────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(episode_rewards, bins=min(15, len(episode_rewards)), color="#378ADD",
        edgecolor="white", linewidth=1, alpha=0.8, label="Trained episodes")
ax.axvline(x=results["passive"]["avg"], color="#E24B4A", linestyle="--",
           linewidth=2, label=f"Passive avg ({results['passive']['avg']:.0f})")
ax.axvline(x=results["smart"]["avg"], color="#1D9E75", linestyle="--",
           linewidth=2, label=f"Smart avg ({results['smart']['avg']:.0f})")
ax.set_xlabel("Total Episode Reward", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Distribution of Training Episode Rewards", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("reward_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: reward_distribution.png")

# ── Plot: Improvement over time (% above smart baseline) ─────────────
smart_avg = results["smart"]["avg"]
pct_above = [(r - smart_avg) / smart_avg * 100 for r in episode_rewards]

fig, ax = plt.subplots(figsize=(12, 5))
colors_pct = ["#1D9E75" if p >= 0 else "#E24B4A" for p in pct_above]
ax.bar(range(1, len(pct_above)+1), pct_above, color=colors_pct, width=0.7,
       edgecolor="white", linewidth=0.5)
ax.axhline(y=0, color="black", linewidth=1)
ax.set_xlabel("Episode", fontsize=11)
ax.set_ylabel("% Above Smart Baseline", fontsize=11)
ax.set_title("Training Progress — Percentage Above Smart Baseline Per Episode",
             fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("improvement_over_baseline.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: improvement_over_baseline.png")

# ── Post-training evaluation: run trained model vs baselines ──────────
print("\n--- Post-training evaluation (trained model vs baselines) ---")
model.eval()
eval_env = MaintenanceArenaEnvironment()

trained_eval_rewards = []
trained_eval_scores = []

for eval_seed in SEEDS:
    obs_obj = eval_env.reset(seed=eval_seed)
    obs_dict = obs_to_dict(obs_obj)
    eval_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_obs(obs_dict)},
    ]
    ep_reward = 0.0
    for step in range(168):
        recent = [eval_messages[0]] + eval_messages[max(1, len(eval_messages)-8):]
        prompt = tokenizer.apply_chat_template(recent, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        pl = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=100, temperature=0.3,
                                 do_sample=True, pad_token_id=tokenizer.pad_token_id)
        gen_text = tokenizer.decode(out[0][pl:], skip_special_tokens=True)
        action = parse_action(gen_text)
        obs_dict, reward, done = do_step(eval_env,
                                         action.get("action_type", "monitor"),
                                         action.get("machine_id", -1),
                                         action.get("parameters", {}))
        ep_reward += reward
        eval_messages.append({"role": "assistant", "content": gen_text})
        if done:
            break
        eval_messages.append({"role": "user", "content": format_obs(obs_dict)})

    score = None
    try:
        last_obs = eval_env._factory
        score_val = eval_env.rubric(
            MaintenanceAction(action_type=ActionType.MONITOR, machine_id=-1, parameters={}),
            eval_env._build_observation("eval", True, 0.0)
        )
        score = round(score_val, 4)
    except:
        pass
    trained_eval_rewards.append(ep_reward)
    trained_eval_scores.append(score)
    print(f"  seed={eval_seed}: reward={ep_reward:.1f}  rubric={score}")

trained_avg_reward = sum(trained_eval_rewards) / len(trained_eval_rewards)
clean_eval_scores = [s for s in trained_eval_scores if s is not None]
trained_avg_rubric = sum(clean_eval_scores) / len(clean_eval_scores) if clean_eval_scores else None
print(f"  AVG: reward={trained_avg_reward:.1f}  rubric={trained_avg_rubric}")

# ── Plot: Final comparison with trained model evaluation ──────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Bar chart: rewards
agents_final = ["Passive", "Random", "Smart", "Trained"]
rewards_final = [results["passive"]["avg"], results["random"]["avg"],
                 results["smart"]["avg"], trained_avg_reward]
colors_final = ["#E24B4A", "#BA7517", "#1D9E75", "#378ADD"]
bars = axes[0].bar(agents_final, rewards_final, color=colors_final, width=0.45,
                   edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, rewards_final):
    y = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2, y + max(abs(y)*0.02, 1),
                 f"{val:.1f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Total Episode Reward (avg over 5 seeds)", fontsize=11)
axes[0].set_title("Final Evaluation — Total Reward", fontsize=12, fontweight="bold")
axes[0].grid(True, alpha=0.3, axis="y")

# Bar chart: rubric scores
rubrics_final = [results["passive"]["avg_score"] or 0, results["random"]["avg_score"] or 0,
                 results["smart"]["avg_score"] or 0, trained_avg_rubric or 0]
bars2 = axes[1].bar(agents_final, rubrics_final, color=colors_final, width=0.45,
                    edgecolor="white", linewidth=1.5)
for bar, val in zip(bars2, rubrics_final):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Rubric Score (0.0 — 1.0)", fontsize=11)
axes[1].set_title("Final Evaluation — Rubric Score", fontsize=12, fontweight="bold")
axes[1].set_ylim(0, 1.15)
axes[1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("final_evaluation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: final_evaluation.png")

# ============================================================================
# Save results and model
# ============================================================================

final_avg = sum(episode_scores[-5:]) / min(5, len(episode_scores))
best_score = max(episode_scores)

train_results = {
    "model": MODEL_NAME,
    "method": "GRPO with Unsloth 4-bit LoRA (Direct Python)",
    "episodes": NUM_EPISODES,
    "steps_per_episode": TRAIN_STEPS,
    "generations_per_step": NUM_GENERATIONS,
    "time_minutes": round(total_time / 60, 1),
    "baselines": {
        "passive_avg_reward": round(results["passive"]["avg"], 2),
        "passive_avg_rubric": results["passive"]["avg_score"],
        "random_avg_reward": round(results["random"]["avg"], 2),
        "random_avg_rubric": results["random"]["avg_score"],
        "smart_avg_reward": round(results["smart"]["avg"], 2),
        "smart_avg_rubric": results["smart"]["avg_score"],
    },
    "trained_evaluation": {
        "avg_reward": round(trained_avg_reward, 2),
        "avg_rubric": trained_avg_rubric,
        "per_seed_rewards": [round(r, 2) for r in trained_eval_rewards],
        "per_seed_rubrics": trained_eval_scores,
    },
    "trained_final_avg_per_step": round(final_avg, 4),
    "trained_best_avg_per_step": round(best_score, 4),
    "episode_rewards": [round(r, 2) for r in episode_rewards],
    "episode_scores": [round(s, 4) for s in episode_scores],
    "episode_difficulties": episode_difficulties,
}

with open("training_results.json", "w") as f:
    json.dump(train_results, f, indent=2)
print("Saved: training_results.json")

try:
    model.save_pretrained("./trained_model")
    tokenizer.save_pretrained("./trained_model")
    print("Saved: ./trained_model/")
except Exception as e:
    print(f"Could not save model: {e}")

print(f"\n{'='*60}")
print(f"  FINAL RESULTS")
print(f"  {'─'*56}")
print(f"  Baselines (168-step episodes, avg over 5 seeds):")
print(f"    Passive:  reward={results['passive']['avg']:7.1f}  rubric={results['passive']['avg_score']}")
print(f"    Random:   reward={results['random']['avg']:7.1f}  rubric={results['random']['avg_score']}")
print(f"    Smart:    reward={results['smart']['avg']:7.1f}  rubric={results['smart']['avg_score']}")
print(f"  {'─'*56}")
print(f"  Trained agent ({NUM_EPISODES} episodes × {TRAIN_STEPS} steps):")
print(f"    Final avg/step:  {final_avg:.3f}")
print(f"    Best avg/step:   {best_score:.3f}")
print(f"    Training time:   {total_time/60:.1f} min")
print(f"  {'─'*56}")
print(f"  Files saved:")
print(f"    baseline_comparison.png  — Passive vs Random vs Smart")
print(f"    rubric_scores.png        — Rubric scores by strategy")
print(f"    training_curve.png       — GRPO reward progression")
print(f"    training_results.json    — All numbers")
print(f"{'='*60}")

# ============================================================================
# Download files (Colab only)
# ============================================================================

try:
    from google.colab import files
    print("\nDownloading plots...")
    for f in ["training_curve.png", "baseline_comparison.png", "rubric_scores.png",
              "trained_vs_baselines.png", "reward_distribution.png",
              "improvement_over_baseline.png", "final_evaluation.png",
              "training_results.json"]:
        try:
            files.download(f)
        except:
            pass
    print("✅ All files downloaded!")
except ImportError:
    print("\nNot running in Colab. Files are in the current directory:")
    os.system("ls -la *.png *.json")
