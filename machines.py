"""
Machine simulation engine.

Simulates 5 industrial machines with:
  - Degradation models (gradual + random shocks)
  - Noisy sensor readings
  - Cascading failure dependencies
  - Maintenance and repair mechanics
  - Adaptive difficulty
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


# ── Machine Definitions ─────────────────────────────────────────────────────

@dataclass
class SensorSpec:
    """Specification for one sensor on a machine."""
    name: str
    unit: str
    base_value: float        # Normal operating value
    degradation_coeff: float # How much sensor shifts per health-point lost
    noise_std: float         # Gaussian noise standard deviation
    critical_threshold: float # Value at which alarm triggers


@dataclass
class MachineSpec:
    """Blueprint for one machine type."""
    name: str
    sensors: list[SensorSpec]
    base_degradation_rate: float  # Health lost per hour at full speed
    repair_time: int              # Hours for scheduled maintenance
    repair_cost: int              # Dollar cost of scheduled maintenance
    failure_repair_cost: int      # Dollar cost of emergency repair after failure
    failure_repair_time: int      # Hours for emergency repair


# The 5 machines in our production line
MACHINE_SPECS = [
    MachineSpec(
        name="Conveyor belt",
        sensors=[
            SensorSpec("vibration", "mm/s", 2.0, 0.08, 0.3, 6.0),
            SensorSpec("temperature", "C", 40.0, 0.3, 1.5, 75.0),
            SensorSpec("power_draw", "kW", 3.0, 0.05, 0.2, 5.5),
        ],
        base_degradation_rate=0.3,
        repair_time=4,
        repair_cost=1200,
        failure_repair_cost=8000,
        failure_repair_time=12,
    ),
    MachineSpec(
        name="CNC mill",
        sensors=[
            SensorSpec("vibration", "mm/s", 3.5, 0.12, 0.4, 8.0),
            SensorSpec("temperature", "C", 55.0, 0.5, 2.0, 90.0),
            SensorSpec("pressure", "bar", 4.0, 0.03, 0.15, 6.0),
        ],
        base_degradation_rate=0.5,
        repair_time=6,
        repair_cost=2500,
        failure_repair_cost=15000,
        failure_repair_time=18,
    ),
    MachineSpec(
        name="Heat treat oven",
        sensors=[
            SensorSpec("vibration", "mm/s", 1.5, 0.05, 0.2, 4.0),
            SensorSpec("temperature", "C", 180.0, 1.2, 3.0, 250.0),
            SensorSpec("power_draw", "kW", 11.0, 0.15, 0.5, 16.0),
        ],
        base_degradation_rate=0.7,
        repair_time=8,
        repair_cost=3000,
        failure_repair_cost=20000,
        failure_repair_time=24,
    ),
    MachineSpec(
        name="Hydraulic press",
        sensors=[
            SensorSpec("vibration", "mm/s", 4.0, 0.1, 0.5, 9.0),
            SensorSpec("pressure", "bar", 80.0, 0.8, 2.0, 110.0),
            SensorSpec("acoustic", "dB", 65.0, 0.4, 1.5, 85.0),
        ],
        base_degradation_rate=0.4,
        repair_time=5,
        repair_cost=2000,
        failure_repair_cost=12000,
        failure_repair_time=15,
    ),
    MachineSpec(
        name="Packaging unit",
        sensors=[
            SensorSpec("vibration", "mm/s", 1.0, 0.06, 0.2, 3.5),
            SensorSpec("temperature", "C", 32.0, 0.2, 1.0, 55.0),
            SensorSpec("power_draw", "kW", 2.5, 0.04, 0.15, 4.5),
        ],
        base_degradation_rate=0.25,
        repair_time=3,
        repair_cost=800,
        failure_repair_cost=5000,
        failure_repair_time=10,
    ),
]

# Dependency graph: machine_index -> list of (affected_index, stress_multiplier)
# Includes both forward (downstream) and backward (physical proximity) effects
DEPENDENCIES = {
    0: [(1, 1.3)],              # Conveyor issues stress CNC downstream
    1: [(2, 1.2)],              # CNC issues stress Oven downstream
    2: [(1, 1.15), (3, 1.25)],  # Oven heat affects CNC backward; feeds Press forward
    3: [(2, 1.1), (4, 1.2)],    # Press vibration propagates to Oven backward; feeds Packaging
    4: [],                       # Packaging is end of line
}


# ── Machine State ────────────────────────────────────────────────────────────

@dataclass
class MachineState:
    """Runtime state of one machine (includes hidden health)."""
    spec: MachineSpec
    health: float = 100.0                # 0-100, hidden from agent
    status: str = "running"              # running, warning, critical, maintenance, failed
    degradation_rate: float = 0.0        # Current effective degradation rate
    is_under_maintenance: bool = False
    maintenance_hours_left: int = 0
    total_repairs: int = 0               # Imperfect repair tracking
    # Health to restore AFTER maintenance completes (not immediately)
    pending_restore_health: float = -1.0
    # Sensor noise multiplier (increases with difficulty)
    noise_multiplier: float = 1.0
    # Track health before maintenance for cascade-prevention scoring
    health_at_maintenance: float = -1.0

    def __post_init__(self):
        self.degradation_rate = self.spec.base_degradation_rate


# ── Factory Simulation ───────────────────────────────────────────────────────

class FactorySimulator:
    """
    Simulates the factory floor with 5 interconnected machines.

    Manages degradation, sensor generation, maintenance, and cascading failures.
    """

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.machines: list[MachineState] = []
        self.hour: int = 0
        self.total_cost: int = 0
        self.units_produced: int = 0
        self.production_target: int = 800  # Units expected per week
        self.line_speed_pct: int = 100
        self.events: list[str] = []
        self.cascade_failures: int = 0
        self.prevented_cascades: int = 0
        self.correct_preventive: int = 0
        self.unnecessary_preventive: int = 0
        self.total_downtime_hours: int = 0

        # Track which machines failed THIS step (avoids double-counting in reward)
        self._newly_failed: set[int] = set()

        # Crew state
        self.crew_available: bool = True
        self.crew_busy_machine: int = -1
        self.crew_hours_left: int = 0

        # Queue for machines waiting for crew
        self._crew_queue: list[int] = []

        # Spare parts: machine_name -> count
        self.spare_parts: dict[str, int] = {}
        self.parts_on_order: dict[str, int] = {}

        # Difficulty
        self.difficulty_level: int = 1
        self._difficulty_history: list[float] = []

        self._init_machines()

    def _init_machines(self):
        """Initialize all 5 machines with some random starting health."""
        self.machines = []
        for spec in MACHINE_SPECS:
            m = MachineState(spec=spec)
            m.health = 80.0 + self.rng.random() * 20.0
            m.degradation_rate = spec.base_degradation_rate
            self.machines.append(m)
            self.spare_parts[spec.name] = 2

    def reset(self, seed: int | None = None, difficulty: int = 1):
        """Reset the factory for a new episode."""
        if seed is not None:
            self.rng = random.Random(seed)
        self.hour = 0
        self.total_cost = 0
        self.units_produced = 0
        self.line_speed_pct = 100
        self.events = []
        self.cascade_failures = 0
        self.prevented_cascades = 0
        self.correct_preventive = 0
        self.unnecessary_preventive = 0
        self.total_downtime_hours = 0
        self._newly_failed = set()
        self.crew_available = True
        self.crew_busy_machine = -1
        self.crew_hours_left = 0
        self._crew_queue = []
        self.parts_on_order = {}
        self.difficulty_level = difficulty
        self._difficulty_history = []
        self._init_machines()
        self._apply_difficulty()

    def _apply_difficulty(self):
        """Adjust the simulation based on difficulty level."""
        d = self.difficulty_level

        for m in self.machines:
            m.noise_multiplier = 1.0 + (d - 1) * 0.3
            m.degradation_rate = m.spec.base_degradation_rate * (1.0 + (d - 1) * 0.15)

        if d >= 3:
            for name in self.spare_parts:
                self.spare_parts[name] = max(1, self.spare_parts[name] - 1)

        self.production_target = 800 + (d - 1) * 50

    # ── Stepping ─────────────────────────────────────────────────────────

    def step(self) -> dict:
        """Advance the factory by one hour."""
        self.hour += 1
        step_events = []
        machines_running = 0
        produced = 0  # FIX #4: Initialize before conditional

        # Clear newly-failed tracker for this step
        self._newly_failed = set()

        # Update crew
        if not self.crew_available:
            self.crew_hours_left -= 1
            if self.crew_hours_left <= 0:
                mc = self.machines[self.crew_busy_machine]
                mc.is_under_maintenance = False
                mc.maintenance_hours_left = 0  # FIX #11: Reset hours left
                # FIX #2: Apply health restoration when maintenance completes
                if mc.pending_restore_health > 0:
                    mc.health = mc.pending_restore_health
                    mc.pending_restore_health = -1.0
                mc.status = "running"
                step_events.append(
                    f"Maintenance complete on {mc.spec.name}. Restored to service."
                )
                self.crew_available = True
                self.crew_busy_machine = -1

                # FIX #5: Process crew queue
                if self._crew_queue:
                    next_id = self._crew_queue.pop(0)
                    next_m = self.machines[next_id]
                    if next_m.is_under_maintenance:
                        self.crew_available = False
                        self.crew_busy_machine = next_id
                        self.crew_hours_left = next_m.maintenance_hours_left
                        step_events.append(
                            f"Crew now working on {next_m.spec.name} (was queued)."
                        )

        # FIX #11: Decrement maintenance_hours_left for the machine being worked on
        if not self.crew_available and self.crew_busy_machine >= 0:
            mc = self.machines[self.crew_busy_machine]
            if mc.maintenance_hours_left > 0:
                mc.maintenance_hours_left -= 1

        # Update spare parts delivery
        arrived = []
        for name, hrs in list(self.parts_on_order.items()):
            self.parts_on_order[name] = hrs - 1
            if self.parts_on_order[name] <= 0:
                self.spare_parts[name] = self.spare_parts.get(name, 0) + 1
                step_events.append(f"Spare parts arrived for {name}.")
                arrived.append(name)
        for name in arrived:
            del self.parts_on_order[name]

        # Degrade each machine
        for i, m in enumerate(self.machines):
            if m.status == "failed" or m.is_under_maintenance:
                # FIX #12: Both failed AND maintenance count as downtime
                self.total_downtime_hours += 1
                continue

            # Apply degradation
            speed_factor = self.line_speed_pct / 100.0
            load_factor = speed_factor * (1.0 + m.total_repairs * 0.05)
            degradation = m.degradation_rate * load_factor
            noise = self.rng.gauss(0, 0.1)
            m.health -= max(0, degradation + noise)

            # Random shock events (rare)
            if self.rng.random() < 0.005 * self.difficulty_level:
                shock = self.rng.uniform(5, 15)
                m.health -= shock
                step_events.append(
                    f"Power surge affected {m.spec.name}! Health dropped."
                )

            m.health = max(0.0, min(100.0, m.health))

            # Update status based on health
            if m.health <= 0:
                m.status = "failed"
                m.health = 0
                self._newly_failed.add(i)  # For reward calculation
                self.total_cost += m.spec.failure_repair_cost
                step_events.append(
                    f"FAILURE: {m.spec.name} has failed! "
                    f"Emergency repair cost: ${m.spec.failure_repair_cost:,}"
                )
                self._apply_cascade(i, step_events)
            elif m.health < 15:
                if m.status != "critical":
                    step_events.append(
                        f"CRITICAL: {m.spec.name} health critically low!"
                    )
                m.status = "critical"
            elif m.health < 40:
                if m.status == "running":
                    step_events.append(
                        f"WARNING: {m.spec.name} sensors showing elevated readings."
                    )
                m.status = "warning"
            else:
                m.status = "running"

            if m.status in ("running", "warning"):
                machines_running += 1

        # Production output
        if machines_running > 0:
            production_rate = (machines_running / 5.0) * (self.line_speed_pct / 100.0)
            produced = int(production_rate * 5)
            self.units_produced += produced

        self.events = (step_events + self.events)[:20]

        return {
            "hour": self.hour,
            "events": step_events,
            "machines_running": machines_running,
            "produced_this_hour": produced,
        }

    def _apply_cascade(self, failed_idx: int, events: list[str]):
        """Apply cascade damage when a machine fails."""
        deps = DEPENDENCIES.get(failed_idx, [])
        for affected_idx, stress_mult in deps:
            affected = self.machines[affected_idx]
            if affected.status in ("failed", "maintenance"):
                continue
            damage = self.rng.uniform(5, 20) * stress_mult
            affected.health -= damage
            # FIX #10: Add stress proportional to base rate, don't multiply
            affected.degradation_rate += (stress_mult - 1.0) * affected.spec.base_degradation_rate
            self.cascade_failures += 1
            events.append(
                f"CASCADE: {self.machines[failed_idx].spec.name} failure "
                f"stressed {affected.spec.name}! (-{damage:.0f} health)"
            )

    # ── Agent Actions ────────────────────────────────────────────────────

    def do_monitor(self) -> str:
        """Just observe — no action taken."""
        return "Monitoring all machines. No action taken."

    def do_diagnostic(self, machine_id: int) -> str:
        """Run diagnostic on a specific machine."""
        if not 0 <= machine_id <= 4:
            return f"Invalid machine_id: {machine_id}. Must be 0-4."

        m = self.machines[machine_id]
        if m.status == "failed":
            return f"{m.spec.name} has already failed. Cannot diagnose."
        if m.is_under_maintenance:
            return f"{m.spec.name} is under maintenance. Cannot diagnose."

        est_life = m.health / max(m.degradation_rate, 0.01)
        est_life += self.rng.gauss(0, est_life * 0.15)
        est_life = max(0, est_life)

        if m.health < 30:
            severity = "SEVERE degradation detected"
        elif m.health < 50:
            severity = "Moderate degradation detected"
        elif m.health < 70:
            severity = "Mild wear detected"
        else:
            severity = "Machine in good condition"

        component_details = []
        for sensor in m.spec.sensors:
            reading = self._get_sensor_reading(m, sensor)
            pct_of_threshold = (reading / sensor.critical_threshold) * 100
            component_details.append(
                f"  {sensor.name}: {reading:.1f} {sensor.unit} "
                f"({pct_of_threshold:.0f}% of threshold)"
            )

        return (
            f"DIAGNOSTIC REPORT: {m.spec.name}\n"
            f"  Status: {severity}\n"
            f"  Estimated remaining life: {est_life:.0f} hours\n"
            f"  Degradation rate: {m.degradation_rate:.2f}/hr\n"
            f"  Total past repairs: {m.total_repairs}\n"
            f"  Component readings:\n" + "\n".join(component_details)
        )

    def do_schedule_maintenance(self, machine_id: int) -> tuple[str, bool]:
        """
        Schedule preventive maintenance.
        Returns (result_text, was_correct).
        was_correct = True if machine actually needed it (health < 50).
        """
        if not 0 <= machine_id <= 4:
            return f"Invalid machine_id: {machine_id}.", False

        m = self.machines[machine_id]
        if m.status == "failed":
            return f"{m.spec.name} has already failed. Use emergency repair.", False
        if m.is_under_maintenance:
            return f"{m.spec.name} is already under maintenance.", False
        if not self.crew_available:
            return "Maintenance crew is busy. Wait for them to finish.", False

        if self.spare_parts.get(m.spec.name, 0) < 1:
            return (
                f"No spare parts available for {m.spec.name}. "
                f"Order parts first with ORDER_PARTS action.",
                False,
            )

        # FIX #6 (aligned): health < 50 matches closer to warn threshold (40)
        was_correct = m.health < 50
        m.health_at_maintenance = m.health

        m.is_under_maintenance = True
        m.status = "maintenance"
        m.maintenance_hours_left = m.spec.repair_time

        self.crew_available = False
        self.crew_busy_machine = machine_id
        self.crew_hours_left = m.spec.repair_time

        self.total_cost += m.spec.repair_cost
        self.spare_parts[m.spec.name] -= 1
        m.total_repairs += 1

        # FIX #2: Store pending health — don't restore immediately
        restore_to = 95 - m.total_repairs * 3
        restore_to = max(70, min(98, restore_to))
        m.pending_restore_health = restore_to

        m.degradation_rate = m.spec.base_degradation_rate * (1.0 + m.total_repairs * 0.03)

        if was_correct:
            self.correct_preventive += 1
            # FIX #3: Broader cascade prevention check
            deps = DEPENDENCIES.get(machine_id, [])
            if deps and m.health_at_maintenance < 40:
                self.prevented_cascades += 1

            return (
                f"Preventive maintenance started on {m.spec.name}.\n"
                f"  Cost: ${m.spec.repair_cost:,}\n"
                f"  Duration: {m.spec.repair_time} hours\n"
                f"  Health will be restored to {restore_to}% when complete.\n"
                f"  Good call — machine was showing real degradation.",
                True,
            )
        else:
            self.unnecessary_preventive += 1
            return (
                f"Preventive maintenance started on {m.spec.name}.\n"
                f"  Cost: ${m.spec.repair_cost:,}\n"
                f"  Duration: {m.spec.repair_time} hours\n"
                f"  Note: Machine was in relatively good condition ({m.health:.0f}%).",
                False,
            )

    def do_emergency_shutdown(self, machine_id: int) -> str:
        """Emergency shutdown of a machine."""
        if not 0 <= machine_id <= 4:
            return f"Invalid machine_id: {machine_id}."

        m = self.machines[machine_id]
        if m.status == "failed":
            return f"{m.spec.name} already failed."
        if m.is_under_maintenance:
            return f"{m.spec.name} already under maintenance."

        m.status = "maintenance"
        m.is_under_maintenance = True
        cost = int(m.spec.failure_repair_cost * 0.6)
        repair_time = int(m.spec.failure_repair_time * 0.7)

        self.total_cost += cost
        m.total_repairs += 1
        m.pending_restore_health = max(80, m.health)
        m.maintenance_hours_left = repair_time

        # FIX #5: Proper crew handling with queue
        if self.crew_available:
            self.crew_available = False
            self.crew_busy_machine = machine_id
            self.crew_hours_left = repair_time
        else:
            self._crew_queue.append(machine_id)

        return (
            f"Emergency shutdown on {m.spec.name}.\n"
            f"  Cost: ${cost:,}\n"
            f"  Estimated repair: {repair_time} hours"
            + (" (queued — crew is busy)" if machine_id in self._crew_queue else "")
        )

    def do_order_parts(self, machine_id: int) -> str:
        """Order spare parts for a machine."""
        if not 0 <= machine_id <= 4:
            return f"Invalid machine_id: {machine_id}."

        m = self.machines[machine_id]
        name = m.spec.name

        if name in self.parts_on_order:
            return f"Parts for {name} already on order ({self.parts_on_order[name]} hrs)."

        delivery_time = self.rng.randint(6, 12)
        part_cost = int(m.spec.repair_cost * 0.3)
        self.parts_on_order[name] = delivery_time
        self.total_cost += part_cost

        return (
            f"Spare parts ordered for {name}.\n"
            f"  Cost: ${part_cost:,}\n"
            f"  Estimated delivery: {delivery_time} hours"
        )

    def do_adjust_speed(self, speed_pct: int) -> str:
        """Adjust production line speed."""
        speed_pct = max(50, min(100, speed_pct))
        old = self.line_speed_pct
        self.line_speed_pct = speed_pct

        return (
            f"Line speed adjusted: {old}% -> {speed_pct}%.\n"
            f"  Production rate change: {(speed_pct - old):+d}%\n"
            f"  Machine wear rate: {'decreased' if speed_pct < old else 'increased'}"
        )

    # ── Sensor Readings ──────────────────────────────────────────────────

    def _get_sensor_reading(self, m: MachineState, sensor: SensorSpec) -> float:
        """Generate a noisy sensor reading based on machine health."""
        health_deficit = 100.0 - m.health
        signal = sensor.base_value + sensor.degradation_coeff * health_deficit
        noise = self.rng.gauss(0, sensor.noise_std * m.noise_multiplier)

        if self.difficulty_level >= 3 and self.rng.random() < 0.02:
            noise += self.rng.uniform(1, 3) * sensor.degradation_coeff

        return max(0, signal + noise)

    def get_observable_machines(self) -> list[dict]:
        """Get the observable state of all machines (no hidden health)."""
        result = []
        for m in self.machines:
            sensors = {}
            for sensor in m.spec.sensors:
                if m.status == "failed":
                    sensors[sensor.name] = 0.0
                elif m.is_under_maintenance:
                    sensors[sensor.name] = sensor.base_value
                else:
                    sensors[sensor.name] = round(
                        self._get_sensor_reading(m, sensor), 1
                    )

            if m.health >= 60:
                health_bar = "good"
            elif m.health >= 30:
                health_bar = "warn"
            else:
                health_bar = "crit"

            result.append({
                "name": m.spec.name,
                "status": m.status,
                "sensors": sensors,
                "health_bar": health_bar,
                "is_under_maintenance": m.is_under_maintenance,
                "maintenance_hours_left": m.maintenance_hours_left,
            })
        return result

    # ── Adaptive Difficulty ──────────────────────────────────────────────

    def evaluate_and_adjust_difficulty(self, episode_score: float):
        """Adjust difficulty based on agent performance."""
        self._difficulty_history.append(episode_score)

        if len(self._difficulty_history) < 3:
            return

        recent_avg = sum(self._difficulty_history[-3:]) / 3

        if recent_avg > 0.7 and self.difficulty_level < 5:
            self.difficulty_level += 1
        elif recent_avg < 0.3 and self.difficulty_level > 1:
            self.difficulty_level -= 1
