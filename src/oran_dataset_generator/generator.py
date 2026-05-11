from __future__ import annotations

import random
import uuid
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SLICES: Tuple[str, str, str] = ("URLLC", "eMBB", "IoT")
EDGE_CASE_SCENARIOS: Tuple[str, ...] = (
    "full_congestion",
    "single_slice_overload",
    "burst_spike",
    "idle_network",
    "prb_scarcity",
    "excess_resources",
    "near_threshold",
    "slight_violation",
    "severe_violation",
    "conflicting_intents",
    "multi_priority",
    "starvation",
    "equal_load",
    "noisy_metrics",
    "inconsistent_state",
    "retransmission_failure",
    "slice_failure",
)
ALL_SCENARIOS: Tuple[str, ...] = ("NORMAL",) + EDGE_CASE_SCENARIOS
MIN_PRBS_PER_SLICE = 5
DEFAULT_SEED = 42


def set_random_seed(seed: int = DEFAULT_SEED) -> None:
    """Set deterministic seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def _clip_float(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    filtered = {slice_name: max(0.0001, float(weights[slice_name])) for slice_name in SLICES}
    total = sum(filtered.values())
    return {slice_name: filtered[slice_name] / total for slice_name in SLICES}


def _allocate_prbs(
    total_prbs: int,
    weights: Dict[str, float],
    min_prbs: int = MIN_PRBS_PER_SLICE,
) -> Dict[str, int]:
    if total_prbs < min_prbs * len(SLICES):
        raise ValueError("total_prbs is too small to satisfy per-slice minimum PRB requirements")

    norm = _normalize_weights(weights)
    distributable = total_prbs - min_prbs * len(SLICES)

    raw_extra = np.array([norm[s] * distributable for s in SLICES], dtype=float)
    floor_extra = np.floor(raw_extra).astype(int)
    allocations = {slice_name: min_prbs + int(floor_extra[idx]) for idx, slice_name in enumerate(SLICES)}

    remainder = distributable - int(floor_extra.sum())
    if remainder > 0:
        fractional = raw_extra - floor_extra
        order = np.argsort(-fractional)
        for i in range(remainder):
            allocations[SLICES[int(order[i % len(SLICES)])]] += 1

    return allocations


def _enforce_total_and_min_prbs(
    allocations: Dict[str, int],
    total_prbs: int,
    min_prbs: int = MIN_PRBS_PER_SLICE,
) -> Dict[str, int]:
    adjusted = {slice_name: max(min_prbs, int(allocations[slice_name])) for slice_name in SLICES}
    current_sum = sum(adjusted.values())

    while current_sum > total_prbs:
        donor = max(SLICES, key=lambda s: adjusted[s])
        if adjusted[donor] <= min_prbs:
            break
        adjusted[donor] -= 1
        current_sum -= 1

    while current_sum < total_prbs:
        receiver = min(SLICES, key=lambda s: adjusted[s])
        adjusted[receiver] += 1
        current_sum += 1

    return adjusted


def _force_slice_prbs(
    allocations: Dict[str, int],
    target_slice: str,
    desired_prbs: int,
    total_prbs: int,
) -> Dict[str, int]:
    adjusted = dict(allocations)
    max_target = total_prbs - MIN_PRBS_PER_SLICE * (len(SLICES) - 1)
    desired = max(MIN_PRBS_PER_SLICE, min(desired_prbs, max_target))

    delta = desired - adjusted[target_slice]
    if delta > 0:
        donors = sorted(
            [s for s in SLICES if s != target_slice],
            key=lambda s: adjusted[s],
            reverse=True,
        )
        for donor in donors:
            movable = max(0, adjusted[donor] - MIN_PRBS_PER_SLICE)
            transfer = min(movable, delta)
            adjusted[donor] -= transfer
            adjusted[target_slice] += transfer
            delta -= transfer
            if delta == 0:
                break
    elif delta < 0:
        receivers = sorted(
            [s for s in SLICES if s != target_slice],
            key=lambda s: adjusted[s],
        )
        remaining = -delta
        for receiver in receivers:
            increment = remaining // len(receivers)
            if increment == 0:
                increment = 1
            adjusted[receiver] += increment
            adjusted[target_slice] -= increment
            remaining -= increment
            if remaining <= 0:
                break

    return _enforce_total_and_min_prbs(adjusted, total_prbs, MIN_PRBS_PER_SLICE)


def _scenario_profile(scenario_type: str) -> Dict[str, Any]:
    if scenario_type not in ALL_SCENARIOS:
        raise ValueError(f"Unknown scenario_type: {scenario_type}")

    profile: Dict[str, Any] = {
        "total_prbs_range": (90, 150),
        "prb_weights": {"URLLC": 1.0, "eMBB": 1.2, "IoT": 0.8},
        "load": {"URLLC": 0.55, "eMBB": 0.60, "IoT": 0.50},
        "queue_bias": {"URLLC": 0.0, "eMBB": 0.0, "IoT": 0.0},
        "latency_bias": {"URLLC": 0.0, "eMBB": 0.0, "IoT": 0.0},
        "throughput_bias": {"URLLC": 0.0, "eMBB": 0.0, "IoT": 0.0},
        "noise_scale": 1.0,
        "focus_slice": None,
    }

    focus_slice: Optional[str] = None

    if scenario_type == "NORMAL":
        profile.update(
            {
                "total_prbs_range": (110, 170),
                "load": {"URLLC": 0.35, "eMBB": 0.42, "IoT": 0.30},
                "queue_bias": {"URLLC": -10.0, "eMBB": -5.0, "IoT": -12.0},
                "latency_bias": {"URLLC": -3.0, "eMBB": -2.0, "IoT": -3.0},
                "throughput_bias": {"URLLC": 6.0, "eMBB": 10.0, "IoT": 4.0},
                "noise_scale": 0.8,
            }
        )

    elif scenario_type == "full_congestion":
        profile.update(
            {
                "total_prbs_range": (55, 95),
                "prb_weights": {"URLLC": 1.0, "eMBB": 1.0, "IoT": 1.0},
                "load": {"URLLC": 1.10, "eMBB": 1.20, "IoT": 1.00},
                "queue_bias": {"URLLC": 45.0, "eMBB": 50.0, "IoT": 40.0},
                "latency_bias": {"URLLC": 15.0, "eMBB": 12.0, "IoT": 10.0},
                "throughput_bias": {"URLLC": -15.0, "eMBB": -20.0, "IoT": -8.0},
                "noise_scale": 1.3,
            }
        )

    elif scenario_type == "single_slice_overload":
        focus_slice = random.choice(SLICES)
        profile["total_prbs_range"] = (80, 145)
        profile["load"] = {"URLLC": 0.50, "eMBB": 0.58, "IoT": 0.42}
        profile["queue_bias"] = {"URLLC": 4.0, "eMBB": 4.0, "IoT": 2.0}
        profile["latency_bias"] = {"URLLC": 2.0, "eMBB": 1.0, "IoT": 1.0}
        profile["throughput_bias"] = {"URLLC": 0.0, "eMBB": 0.0, "IoT": 0.0}
        profile["noise_scale"] = 1.1
        profile["load"][focus_slice] += 0.75
        profile["queue_bias"][focus_slice] += 55.0
        profile["latency_bias"][focus_slice] += 16.0
        profile["throughput_bias"][focus_slice] -= 18.0

    elif scenario_type == "burst_spike":
        focus_slice = random.choice(SLICES)
        profile["total_prbs_range"] = (85, 150)
        profile["load"] = {"URLLC": 0.55, "eMBB": 0.60, "IoT": 0.50}
        profile["queue_bias"] = {"URLLC": 10.0, "eMBB": 12.0, "IoT": 8.0}
        profile["latency_bias"] = {"URLLC": 4.0, "eMBB": 2.0, "IoT": 2.0}
        profile["throughput_bias"] = {"URLLC": -2.0, "eMBB": -2.0, "IoT": -1.0}
        profile["noise_scale"] = 1.7
        profile["load"][focus_slice] += 0.35
        profile["queue_bias"][focus_slice] += 35.0

    elif scenario_type == "idle_network":
        profile.update(
            {
                "total_prbs_range": (120, 200),
                "load": {"URLLC": 0.12, "eMBB": 0.18, "IoT": 0.10},
                "queue_bias": {"URLLC": -28.0, "eMBB": -24.0, "IoT": -30.0},
                "latency_bias": {"URLLC": -5.0, "eMBB": -4.0, "IoT": -4.0},
                "throughput_bias": {"URLLC": 2.0, "eMBB": 6.0, "IoT": 1.0},
                "noise_scale": 0.6,
            }
        )

    elif scenario_type == "prb_scarcity":
        profile.update(
            {
                "total_prbs_range": (50, 72),
                "prb_weights": {"URLLC": 1.2, "eMBB": 1.0, "IoT": 0.8},
                "load": {"URLLC": 0.85, "eMBB": 0.95, "IoT": 0.75},
                "queue_bias": {"URLLC": 20.0, "eMBB": 26.0, "IoT": 18.0},
                "latency_bias": {"URLLC": 8.0, "eMBB": 6.0, "IoT": 5.0},
                "throughput_bias": {"URLLC": -10.0, "eMBB": -16.0, "IoT": -6.0},
                "noise_scale": 1.2,
            }
        )

    elif scenario_type == "excess_resources":
        profile.update(
            {
                "total_prbs_range": (170, 200),
                "prb_weights": {"URLLC": 1.0, "eMBB": 1.1, "IoT": 0.9},
                "load": {"URLLC": 0.30, "eMBB": 0.35, "IoT": 0.25},
                "queue_bias": {"URLLC": -12.0, "eMBB": -8.0, "IoT": -10.0},
                "latency_bias": {"URLLC": -6.0, "eMBB": -4.0, "IoT": -4.0},
                "throughput_bias": {"URLLC": 10.0, "eMBB": 14.0, "IoT": 6.0},
                "noise_scale": 0.7,
            }
        )

    elif scenario_type == "near_threshold":
        profile.update(
            {
                "total_prbs_range": (85, 120),
                "load": {"URLLC": 0.62, "eMBB": 0.70, "IoT": 0.50},
                "queue_bias": {"URLLC": 8.0, "eMBB": 12.0, "IoT": 4.0},
                "latency_bias": {"URLLC": 4.0, "eMBB": 2.0, "IoT": 2.0},
                "throughput_bias": {"URLLC": -4.0, "eMBB": -8.0, "IoT": -2.0},
                "noise_scale": 0.9,
            }
        )

    elif scenario_type == "slight_violation":
        profile.update(
            {
                "total_prbs_range": (80, 120),
                "load": {"URLLC": 0.68, "eMBB": 0.75, "IoT": 0.55},
                "queue_bias": {"URLLC": 12.0, "eMBB": 15.0, "IoT": 6.0},
                "latency_bias": {"URLLC": 5.0, "eMBB": 2.0, "IoT": 3.0},
                "throughput_bias": {"URLLC": -6.0, "eMBB": -10.0, "IoT": -3.0},
                "noise_scale": 1.0,
            }
        )

    elif scenario_type == "severe_violation":
        profile.update(
            {
                "total_prbs_range": (50, 90),
                "prb_weights": {"URLLC": 1.1, "eMBB": 1.2, "IoT": 0.7},
                "load": {"URLLC": 1.00, "eMBB": 1.10, "IoT": 0.90},
                "queue_bias": {"URLLC": 42.0, "eMBB": 48.0, "IoT": 35.0},
                "latency_bias": {"URLLC": 18.0, "eMBB": 12.0, "IoT": 10.0},
                "throughput_bias": {"URLLC": -16.0, "eMBB": -24.0, "IoT": -10.0},
                "noise_scale": 1.4,
            }
        )

    elif scenario_type == "conflicting_intents":
        profile.update(
            {
                "total_prbs_range": (70, 130),
                "prb_weights": {"URLLC": 1.4, "eMBB": 1.4, "IoT": 0.5},
                "load": {"URLLC": 0.90, "eMBB": 0.95, "IoT": 0.45},
                "queue_bias": {"URLLC": 20.0, "eMBB": 24.0, "IoT": 2.0},
                "latency_bias": {"URLLC": 8.0, "eMBB": 6.0, "IoT": 1.0},
                "throughput_bias": {"URLLC": -6.0, "eMBB": -8.0, "IoT": 1.0},
                "noise_scale": 1.1,
            }
        )

    elif scenario_type == "multi_priority":
        profile.update(
            {
                "total_prbs_range": (75, 130),
                "prb_weights": {"URLLC": 1.5, "eMBB": 1.3, "IoT": 0.6},
                "load": {"URLLC": 0.95, "eMBB": 0.90, "IoT": 0.50},
                "queue_bias": {"URLLC": 24.0, "eMBB": 20.0, "IoT": 4.0},
                "latency_bias": {"URLLC": 9.0, "eMBB": 5.0, "IoT": 2.0},
                "throughput_bias": {"URLLC": -8.0, "eMBB": -7.0, "IoT": -1.0},
                "noise_scale": 1.1,
            }
        )

    elif scenario_type == "starvation":
        focus_slice = random.choice(SLICES)
        profile["total_prbs_range"] = (70, 135)
        profile["prb_weights"] = {"URLLC": 1.0, "eMBB": 1.0, "IoT": 1.0}
        profile["load"] = {"URLLC": 0.70, "eMBB": 0.70, "IoT": 0.70}
        profile["queue_bias"] = {"URLLC": 5.0, "eMBB": 5.0, "IoT": 5.0}
        profile["latency_bias"] = {"URLLC": 2.0, "eMBB": 2.0, "IoT": 2.0}
        profile["throughput_bias"] = {"URLLC": 2.0, "eMBB": 2.0, "IoT": 2.0}
        profile["noise_scale"] = 1.2

        for slice_name in SLICES:
            if slice_name == focus_slice:
                profile["prb_weights"][slice_name] = 0.12
                profile["load"][slice_name] = 1.20
                profile["queue_bias"][slice_name] = 60.0
                profile["latency_bias"][slice_name] = 16.0
                profile["throughput_bias"][slice_name] = -20.0
            else:
                profile["prb_weights"][slice_name] = 1.45

    elif scenario_type == "equal_load":
        profile.update(
            {
                "total_prbs_range": (90, 170),
                "prb_weights": {"URLLC": 1.0, "eMBB": 1.0, "IoT": 1.0},
                "load": {"URLLC": 0.60, "eMBB": 0.60, "IoT": 0.60},
                "queue_bias": {"URLLC": 8.0, "eMBB": 8.0, "IoT": 8.0},
                "latency_bias": {"URLLC": 2.0, "eMBB": 2.0, "IoT": 2.0},
                "throughput_bias": {"URLLC": 0.0, "eMBB": 0.0, "IoT": 0.0},
                "noise_scale": 0.9,
            }
        )

    elif scenario_type == "noisy_metrics":
        profile.update(
            {
                "total_prbs_range": (80, 170),
                "load": {"URLLC": 0.55, "eMBB": 0.60, "IoT": 0.55},
                "queue_bias": {"URLLC": 10.0, "eMBB": 10.0, "IoT": 10.0},
                "latency_bias": {"URLLC": 3.0, "eMBB": 3.0, "IoT": 3.0},
                "throughput_bias": {"URLLC": -2.0, "eMBB": -3.0, "IoT": -2.0},
                "noise_scale": 2.4,
            }
        )

    elif scenario_type == "inconsistent_state":
        profile.update(
            {
                "total_prbs_range": (70, 130),
                "load": {"URLLC": 0.65, "eMBB": 0.65, "IoT": 0.60},
                "queue_bias": {"URLLC": 18.0, "eMBB": 18.0, "IoT": 16.0},
                "latency_bias": {"URLLC": 5.0, "eMBB": 4.0, "IoT": 4.0},
                "throughput_bias": {"URLLC": -4.0, "eMBB": -5.0, "IoT": -3.0},
                "noise_scale": 1.4,
            }
        )

    elif scenario_type == "retransmission_failure":
        profile.update(
            {
                "total_prbs_range": (75, 130),
                "prb_weights": {"URLLC": 1.1, "eMBB": 1.2, "IoT": 0.7},
                "load": {"URLLC": 0.95, "eMBB": 1.00, "IoT": 0.50},
                "queue_bias": {"URLLC": 22.0, "eMBB": 28.0, "IoT": 8.0},
                "latency_bias": {"URLLC": 14.0, "eMBB": 11.0, "IoT": 2.0},
                "throughput_bias": {"URLLC": -12.0, "eMBB": -18.0, "IoT": -2.0},
                "noise_scale": 1.3,
            }
        )

    elif scenario_type == "slice_failure":
        focus_slice = random.choice(SLICES)
        profile["total_prbs_range"] = (60, 120)
        profile["prb_weights"] = {"URLLC": 1.0, "eMBB": 1.0, "IoT": 1.0}
        profile["load"] = {"URLLC": 0.60, "eMBB": 0.60, "IoT": 0.60}
        profile["queue_bias"] = {"URLLC": 5.0, "eMBB": 5.0, "IoT": 5.0}
        profile["latency_bias"] = {"URLLC": 2.0, "eMBB": 2.0, "IoT": 2.0}
        profile["throughput_bias"] = {"URLLC": 3.0, "eMBB": 3.0, "IoT": 3.0}
        profile["noise_scale"] = 1.3

        for slice_name in SLICES:
            if slice_name == focus_slice:
                profile["prb_weights"][slice_name] = 0.05
                profile["load"][slice_name] = 1.30
                profile["queue_bias"][slice_name] = 70.0
                profile["latency_bias"][slice_name] = 24.0
                profile["throughput_bias"][slice_name] = -28.0
            else:
                profile["prb_weights"][slice_name] = 1.55

    profile["focus_slice"] = focus_slice
    return profile


def _generate_slice_metrics(
    slice_name: str,
    allocated_prbs: int,
    load_factor: float,
    queue_bias: float,
    latency_bias: float,
    throughput_bias: float,
    noise_scale: float,
) -> Dict[str, Any]:
    queue_mean = 25.0 + load_factor * 70.0 + queue_bias
    queue_size = int(np.clip(np.random.normal(queue_mean, 8.0 * noise_scale), 0, 180))

    latency_noise = float(np.random.normal(0, 1.2 * noise_scale))
    throughput_noise = float(np.random.normal(0, 2.8 * noise_scale))

    if slice_name == "URLLC":
        latency_ms = 4.5 + 0.20 * queue_size - 0.10 * allocated_prbs + latency_bias + latency_noise
        throughput = 7.0 + 0.95 * allocated_prbs - 0.05 * queue_size + throughput_bias + throughput_noise
    elif slice_name == "eMBB":
        latency_ms = 10.0 + 0.13 * queue_size - 0.05 * allocated_prbs + latency_bias + latency_noise
        throughput = 14.0 + 1.75 * allocated_prbs - 0.05 * queue_size + throughput_bias + throughput_noise
    else:
        latency_ms = 16.0 + 0.09 * queue_size - 0.03 * allocated_prbs + latency_bias + latency_noise
        throughput = 4.0 + 0.70 * allocated_prbs - 0.025 * queue_size + throughput_bias + throughput_noise

    return {
        "queue_size": int(queue_size),
        "latency_ms": round(_clip_float(latency_ms, 1.0, 350.0), 2),
        "throughput": round(_clip_float(throughput, 0.1, 250.0), 2),
        "allocated_prbs": int(allocated_prbs),
    }


def _apply_post_scenario_adjustments(
    slices: Dict[str, Dict[str, Any]],
    scenario_type: str,
    focus_slice: Optional[str],
) -> None:
    if scenario_type == "near_threshold":
        slices["URLLC"]["latency_ms"] = round(random.uniform(18.6, 21.0), 2)
        slices["eMBB"]["throughput"] = round(random.uniform(29.0, 32.5), 2)

    elif scenario_type == "slight_violation":
        if random.random() < 0.5:
            slices["URLLC"]["latency_ms"] = round(random.uniform(20.2, 24.8), 2)
        else:
            slices["eMBB"]["throughput"] = round(random.uniform(24.0, 29.9), 2)

    elif scenario_type == "severe_violation":
        slices["URLLC"]["latency_ms"] = round(random.uniform(35.0, 90.0), 2)
        slices["eMBB"]["throughput"] = round(random.uniform(6.0, 22.0), 2)
        for slice_name in ("URLLC", "eMBB"):
            slices[slice_name]["queue_size"] = int(
                min(180, slices[slice_name]["queue_size"] + random.randint(20, 50))
            )

    elif scenario_type == "full_congestion":
        for slice_name in SLICES:
            slices[slice_name]["queue_size"] = int(max(80, slices[slice_name]["queue_size"]))
            lower_bound = 25.0 if slice_name == "URLLC" else 35.0
            slices[slice_name]["latency_ms"] = round(max(lower_bound, slices[slice_name]["latency_ms"]), 2)
            slices[slice_name]["throughput"] = round(
                max(0.1, slices[slice_name]["throughput"] - random.uniform(2.0, 8.0)),
                2,
            )

    elif scenario_type == "burst_spike":
        target = focus_slice or random.choice(SLICES)
        slices[target]["queue_size"] = int(min(180, slices[target]["queue_size"] + random.randint(25, 70)))
        slices[target]["latency_ms"] = round(
            _clip_float(slices[target]["latency_ms"] + random.uniform(8.0, 25.0), 1.0, 350.0),
            2,
        )
        slices[target]["throughput"] = round(
            _clip_float(slices[target]["throughput"] - random.uniform(4.0, 16.0), 0.1, 250.0),
            2,
        )

    elif scenario_type == "idle_network":
        for slice_name in SLICES:
            slices[slice_name]["queue_size"] = int(min(slices[slice_name]["queue_size"], random.randint(0, 20)))
            slices[slice_name]["latency_ms"] = round(_clip_float(slices[slice_name]["latency_ms"], 2.0, 45.0), 2)

    elif scenario_type == "noisy_metrics":
        for slice_name in SLICES:
            slices[slice_name]["queue_size"] = int(
                np.clip(slices[slice_name]["queue_size"] + int(np.random.normal(0, 15)), 0, 180)
            )
            slices[slice_name]["latency_ms"] = round(
                _clip_float(slices[slice_name]["latency_ms"] + float(np.random.normal(0, 8)), 1.0, 350.0),
                2,
            )
            slices[slice_name]["throughput"] = round(
                _clip_float(
                    slices[slice_name]["throughput"] + float(np.random.normal(0, 12)),
                    0.1,
                    250.0,
                ),
                2,
            )

    elif scenario_type == "inconsistent_state":
        target = random.choice(SLICES)
        slices[target]["queue_size"] = random.randint(100, 170)
        slices[target]["latency_ms"] = round(
            _clip_float(slices[target]["latency_ms"] + random.uniform(10.0, 35.0), 1.0, 350.0),
            2,
        )
        slices[target]["throughput"] = round(max(slices[target]["throughput"], random.uniform(45.0, 120.0)), 2)

        other = random.choice([s for s in SLICES if s != target])
        slices[other]["queue_size"] = random.randint(0, 20)
        slices[other]["throughput"] = round(min(slices[other]["throughput"], random.uniform(2.0, 20.0)), 2)

    elif scenario_type == "retransmission_failure":
        for slice_name in ("URLLC", "eMBB"):
            slices[slice_name]["queue_size"] = int(
                min(180, slices[slice_name]["queue_size"] + random.randint(15, 45))
            )
            slices[slice_name]["latency_ms"] = round(
                _clip_float(slices[slice_name]["latency_ms"] + random.uniform(12.0, 32.0), 1.0, 350.0),
                2,
            )
            slices[slice_name]["throughput"] = round(
                _clip_float(slices[slice_name]["throughput"] - random.uniform(8.0, 20.0), 0.1, 250.0),
                2,
            )

    elif scenario_type == "slice_failure":
        target = focus_slice or random.choice(SLICES)
        slices[target]["queue_size"] = random.randint(130, 180)
        slices[target]["latency_ms"] = round(random.uniform(80.0, 220.0), 2)
        slices[target]["throughput"] = round(random.uniform(0.1, 6.0), 2)


def generate_network_state(scenario_type: str) -> Dict[str, Any]:
    """Generate network state metrics according to scenario-specific dynamics."""
    profile = _scenario_profile(scenario_type)
    total_prbs = random.randint(*profile["total_prbs_range"])

    jittered_weights = {
        slice_name: profile["prb_weights"][slice_name] * max(0.2, float(np.random.normal(1.0, 0.08)))
        for slice_name in SLICES
    }
    prb_allocations = _allocate_prbs(total_prbs, jittered_weights, MIN_PRBS_PER_SLICE)

    focus_slice = profile.get("focus_slice")
    if scenario_type == "starvation" and focus_slice:
        prb_allocations = _force_slice_prbs(
            prb_allocations,
            focus_slice,
            random.randint(MIN_PRBS_PER_SLICE, MIN_PRBS_PER_SLICE + 4),
            total_prbs,
        )
    elif scenario_type == "slice_failure" and focus_slice:
        prb_allocations = _force_slice_prbs(
            prb_allocations,
            focus_slice,
            random.randint(MIN_PRBS_PER_SLICE, MIN_PRBS_PER_SLICE + 2),
            total_prbs,
        )

    slices: Dict[str, Dict[str, Any]] = {}
    for slice_name in SLICES:
        slices[slice_name] = _generate_slice_metrics(
            slice_name=slice_name,
            allocated_prbs=prb_allocations[slice_name],
            load_factor=float(profile["load"][slice_name]),
            queue_bias=float(profile["queue_bias"][slice_name]),
            latency_bias=float(profile["latency_bias"][slice_name]),
            throughput_bias=float(profile["throughput_bias"][slice_name]),
            noise_scale=float(profile["noise_scale"]),
        )

    _apply_post_scenario_adjustments(slices, scenario_type, focus_slice)

    return {
        "total_prbs": int(total_prbs),
        "slices": slices,
    }


def compute_sla_status(network_state: Dict[str, Any]) -> Dict[str, str]:
    """Compute SLA status per slice from network state metrics."""
    urllc_latency = float(network_state["slices"]["URLLC"]["latency_ms"])
    embb_throughput = float(network_state["slices"]["eMBB"]["throughput"])
    iot_latency = float(network_state["slices"]["IoT"]["latency_ms"])
    iot_throughput = float(network_state["slices"]["IoT"]["throughput"])

    return {
        "URLLC": "satisfied" if urllc_latency < 20.0 else "violated",
        "eMBB": "satisfied" if embb_throughput > 30.0 else "violated",
        "IoT": "satisfied" if (iot_latency < 180.0 and iot_throughput > 3.0) else "violated",
    }


def _slice_deficit_score(slice_name: str, slice_state: Dict[str, Any]) -> float:
    queue_pressure = float(slice_state["queue_size"]) / 100.0

    if slice_name == "URLLC":
        sla_gap = max(0.0, (float(slice_state["latency_ms"]) - 20.0) / 20.0)
    elif slice_name == "eMBB":
        sla_gap = max(0.0, (30.0 - float(slice_state["throughput"])) / 30.0)
    else:
        iot_latency_gap = max(0.0, (float(slice_state["latency_ms"]) - 180.0) / 180.0)
        iot_tp_gap = max(0.0, (3.0 - float(slice_state["throughput"])) / 3.0)
        sla_gap = iot_latency_gap + iot_tp_gap

    return queue_pressure + 2.0 * sla_gap


def _choose_target_slice(
    network_state: Dict[str, Any],
    sla_status: Dict[str, str],
    scenario_type: str,
) -> str:
    violated = [slice_name for slice_name in SLICES if sla_status[slice_name] == "violated"]
    if violated:
        return max(
            violated,
            key=lambda s: _slice_deficit_score(s, network_state["slices"][s]),
        )

    if scenario_type in {"multi_priority", "conflicting_intents"}:
        return random.choice(("URLLC", "eMBB"))

    return random.choices(list(SLICES), weights=[0.4, 0.4, 0.2], k=1)[0]


def _intent_priority(
    scenario_type: str,
    target_slice: str,
    sla_status: Dict[str, str],
) -> str:
    high_urgency_scenarios = {
        "full_congestion",
        "severe_violation",
        "slice_failure",
        "retransmission_failure",
        "starvation",
    }
    medium_urgency_scenarios = {
        "slight_violation",
        "near_threshold",
        "conflicting_intents",
        "multi_priority",
        "burst_spike",
    }

    if sla_status[target_slice] == "violated" or scenario_type in high_urgency_scenarios:
        return "high"
    if scenario_type in medium_urgency_scenarios:
        return "medium"
    if scenario_type in {"idle_network", "excess_resources"}:
        return "low"
    return "medium"


def _intent_type_for_scenario(scenario_type: str) -> str:
    mapping = {
        "NORMAL": "balanced_optimization",
        "full_congestion": "congestion_mitigation",
        "single_slice_overload": "targeted_overload_relief",
        "burst_spike": "spike_absorption",
        "idle_network": "energy_efficiency",
        "prb_scarcity": "resource_efficiency",
        "excess_resources": "capacity_maximization",
        "near_threshold": "preventive_control",
        "slight_violation": "sla_correction",
        "severe_violation": "incident_recovery",
        "conflicting_intents": "tradeoff_resolution",
        "multi_priority": "priority_coordination",
        "starvation": "fairness_enforcement",
        "equal_load": "balanced_fairness",
        "noisy_metrics": "robust_optimization",
        "inconsistent_state": "state_sanitization",
        "retransmission_failure": "reliability_recovery",
        "slice_failure": "slice_restoration",
    }
    return mapping[scenario_type]


def _build_constraints(target_slice: str, priority: str, scenario_type: str) -> Dict[str, float]:
    if target_slice == "URLLC":
        latency_range = {
            "high": (6.0, 15.0),
            "medium": (10.0, 18.0),
            "low": (12.0, 22.0),
        }[priority]
        throughput_range = {
            "high": (18.0, 40.0),
            "medium": (14.0, 30.0),
            "low": (10.0, 22.0),
        }[priority]
    elif target_slice == "eMBB":
        latency_range = {
            "high": (15.0, 30.0),
            "medium": (20.0, 40.0),
            "low": (25.0, 50.0),
        }[priority]
        throughput_range = {
            "high": (45.0, 95.0),
            "medium": (35.0, 75.0),
            "low": (30.0, 55.0),
        }[priority]
    else:
        latency_range = {
            "high": (40.0, 90.0),
            "medium": (60.0, 120.0),
            "low": (80.0, 180.0),
        }[priority]
        throughput_range = {
            "high": (8.0, 22.0),
            "medium": (5.0, 15.0),
            "low": (3.0, 10.0),
        }[priority]

    constraints: Dict[str, float] = {
        "latency_ms": round(random.uniform(*latency_range), 2),
    }

    include_throughput = (
        target_slice == "eMBB"
        or scenario_type in {"conflicting_intents", "multi_priority", "excess_resources", "NORMAL"}
        or random.random() < 0.35
    )
    if include_throughput:
        constraints["throughput"] = round(random.uniform(*throughput_range), 2)

    return constraints


def generate_intent(
    scenario_type: str,
    network_state: Dict[str, Any],
    sla_status: Dict[str, str],
) -> Dict[str, Any]:
    """Generate intent text and constraints conditioned on scenario and network state."""
    target_slice = _choose_target_slice(network_state, sla_status, scenario_type)
    priority = _intent_priority(scenario_type, target_slice, sla_status)
    intent_type = _intent_type_for_scenario(scenario_type)
    constraints = _build_constraints(target_slice, priority, scenario_type)

    raw_text = (
        f"Apply {intent_type} for {target_slice} with {priority} priority: "
        f"keep latency below {constraints['latency_ms']:.2f} ms"
    )
    if "throughput" in constraints:
        raw_text += f" and throughput above {constraints['throughput']:.2f} Mbps"
    raw_text += "; preserve stability across all slices."

    return {
        "raw_text": raw_text,
        "intent_type": intent_type,
        "target_slice": target_slice,
        "constraints": constraints,
        "priority": priority,
    }


def _compute_slice_scores(
    intent: Dict[str, Any],
    network_state: Dict[str, Any],
    sla_status: Dict[str, str],
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for slice_name in SLICES:
        state = network_state["slices"][slice_name]
        score = 1.0
        score += float(state["queue_size"]) / 120.0

        if slice_name == "URLLC":
            score += max(0.0, (float(state["latency_ms"]) - 20.0) / 12.0)
        elif slice_name == "eMBB":
            score += max(0.0, (30.0 - float(state["throughput"])) / 18.0)
        else:
            score += max(0.0, (float(state["latency_ms"]) - 180.0) / 80.0)
            score += max(0.0, (3.0 - float(state["throughput"])) / 4.0)

        if sla_status[slice_name] == "violated":
            score += 2.5

        if intent["target_slice"] == slice_name:
            score += {"high": 1.8, "medium": 1.0, "low": 0.4}[intent["priority"]]

        scores[slice_name] = max(0.1, score)

    return scores


def _promote_violated_slices(
    current_allocations: Dict[str, int],
    proposed_allocations: Dict[str, int],
    sla_status: Dict[str, str],
    total_prbs: int,
) -> Dict[str, int]:
    adjusted = dict(proposed_allocations)
    violated = [s for s in SLICES if sla_status[s] == "violated"]

    for target in violated:
        if adjusted[target] > current_allocations[target]:
            continue

        needed = current_allocations[target] - adjusted[target] + 1
        donors = sorted(
            [s for s in SLICES if s != target],
            key=lambda s: adjusted[s] - MIN_PRBS_PER_SLICE,
            reverse=True,
        )

        for donor in donors:
            movable = max(0, adjusted[donor] - MIN_PRBS_PER_SLICE)
            shift = min(movable, needed)
            adjusted[donor] -= shift
            adjusted[target] += shift
            needed -= shift
            if needed <= 0:
                break

    return _enforce_total_and_min_prbs(adjusted, total_prbs, MIN_PRBS_PER_SLICE)


def _slice_priority_map(
    scores: Dict[str, float],
    intent: Dict[str, Any],
    sla_status: Dict[str, str],
) -> Dict[str, str]:
    ordered = sorted(SLICES, key=lambda s: scores[s], reverse=True)
    ranking_labels = {
        ordered[0]: "high",
        ordered[1]: "medium",
        ordered[2]: "low",
    }

    priorities: Dict[str, str] = {}
    for slice_name in SLICES:
        if sla_status[slice_name] == "violated":
            priorities[slice_name] = "critical"
        elif slice_name == intent["target_slice"]:
            priorities[slice_name] = {
                "high": "high",
                "medium": "elevated",
                "low": "guarded",
            }[intent["priority"]]
        else:
            priorities[slice_name] = ranking_labels[slice_name]

    return priorities


def _ric_strategy(
    scenario_type: str,
    sla_status: Dict[str, str],
    intent: Dict[str, Any],
) -> str:
    violated = [slice_name for slice_name in SLICES if sla_status[slice_name] == "violated"]
    if violated:
        if len(violated) > 1:
            return "multi_slice_sla_recovery"
        if violated[0] == "URLLC":
            return "latency_first_reallocation"
        if violated[0] == "eMBB":
            return "throughput_reinforcement"
        return "iot_stability_recovery"

    if scenario_type == "prb_scarcity":
        return "scarcity_aware_reallocation"
    if scenario_type == "excess_resources":
        return "capacity_maximization"
    if intent["priority"] == "high" and intent["target_slice"] == "URLLC":
        return "proactive_latency_protection"
    return "balanced_proportional_control"


def _normalized_queue_weights(raw_weights: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(raw_weights.values()))
    normalized = {slice_name: raw_weights[slice_name] / total for slice_name in SLICES}

    rounded = {slice_name: round(normalized[slice_name], 3) for slice_name in SLICES}
    correction = round(1.0 - sum(rounded.values()), 3)
    rounded["IoT"] = round(rounded["IoT"] + correction, 3)

    if rounded["IoT"] < 0.0:
        rounded["IoT"] = 0.0
        residual = round(1.0 - rounded["IoT"], 3)
        two_slice_total = rounded["URLLC"] + rounded["eMBB"]
        rounded["URLLC"] = round(residual * (rounded["URLLC"] / two_slice_total), 3)
        rounded["eMBB"] = round(1.0 - rounded["URLLC"], 3)

    return rounded


def generate_policy_output(
    intent: Dict[str, Any],
    network_state: Dict[str, Any],
    sla_status: Dict[str, str],
    scenario_type: str,
) -> Dict[str, Any]:
    """Generate multi-layer O-RAN policy outputs with SLA-aware PRB reallocation."""
    total_prbs = int(network_state["total_prbs"])
    current_allocations = {
        slice_name: int(network_state["slices"][slice_name]["allocated_prbs"])
        for slice_name in SLICES
    }

    scores = _compute_slice_scores(intent, network_state, sla_status)
    proposed_allocations = _allocate_prbs(total_prbs, scores, MIN_PRBS_PER_SLICE)

    ric_allocations = _promote_violated_slices(
        current_allocations=current_allocations,
        proposed_allocations=proposed_allocations,
        sla_status=sla_status,
        total_prbs=total_prbs,
    )

    urllc_is_critical = sla_status["URLLC"] == "violated" or (
        intent["target_slice"] == "URLLC" and intent["priority"] == "high"
    )
    scheduler = "priority" if urllc_is_critical else "proportional_fair"

    queue_weights = dict(scores)
    if scheduler == "priority":
        queue_weights["URLLC"] *= 1.35
    normalized_weights = _normalized_queue_weights(queue_weights)

    if scenario_type in {"slice_failure", "retransmission_failure", "severe_violation"}:
        handover_mode = "resilient_make_before_break"
    elif any(value == "violated" for value in sla_status.values()):
        handover_mode = "load_aware_fast_handover"
    elif intent["priority"] == "high":
        handover_mode = "predictive_handover"
    else:
        handover_mode = "standard_adaptive"

    if intent["priority"] == "high":
        bearer_priority = "gbr_critical"
    elif intent["priority"] == "medium":
        bearer_priority = "gbr_preferred"
    else:
        bearer_priority = "best_effort"

    if sla_status["URLLC"] == "violated":
        power_bias = "URLLC:+2dB"
    elif sla_status["eMBB"] == "violated":
        power_bias = "eMBB:+1dB"
    elif sla_status["IoT"] == "violated":
        power_bias = "IoT:+1dB"
    elif scenario_type == "idle_network":
        power_bias = "energy_saving:-1dB"
    else:
        power_bias = "balanced:0dB"

    return {
        "SMO": {
            "slice_priority": _slice_priority_map(scores, intent, sla_status),
        },
        "RIC": {
            "prb_allocation": ric_allocations,
            "strategy": _ric_strategy(scenario_type, sla_status, intent),
        },
        "CU": {
            "handover_mode": handover_mode,
            "bearer_priority": bearer_priority,
        },
        "DU": {
            "scheduler": scheduler,
            "queue_weights": normalized_weights,
        },
        "RU": {
            "power_bias": power_bias,
        },
    }


def generate_sample(sample_index: int, scenario_type: str) -> Dict[str, Any]:
    """Generate one complete sample matching the strict dataset schema."""
    _ = sample_index
    network_state = generate_network_state(scenario_type)
    sla_status = compute_sla_status(network_state)
    intent = generate_intent(scenario_type, network_state, sla_status)
    policy_output = generate_policy_output(intent, network_state, sla_status, scenario_type)

    return {
        "sample_id": str(uuid.uuid4()),
        "scenario_type": scenario_type,
        "intent": intent,
        "network_state": network_state,
        "sla_status": sla_status,
        "policy_output": policy_output,
    }


def expected_scenario_distribution(total_samples: int) -> Dict[str, int]:
    normal_count = int(round(total_samples * 0.40))
    edge_count = total_samples - normal_count

    distribution: Dict[str, int] = {"NORMAL": normal_count}
    base_per_edge = edge_count // len(EDGE_CASE_SCENARIOS)
    remainder = edge_count % len(EDGE_CASE_SCENARIOS)

    for idx, scenario in enumerate(EDGE_CASE_SCENARIOS):
        distribution[scenario] = base_per_edge + (1 if idx < remainder else 0)

    return distribution


def build_scenario_plan(total_samples: int) -> List[str]:
    """Build a shuffled scenario plan with 40/60 normal-edge split."""
    if total_samples <= 0:
        raise ValueError("total_samples must be > 0")

    distribution = expected_scenario_distribution(total_samples)
    plan: List[str] = []
    for scenario, count in distribution.items():
        plan.extend([scenario] * count)

    random.shuffle(plan)
    return plan


def _validate_slice_state(slice_state: Dict[str, Any], path: str) -> None:
    required = {"queue_size", "latency_ms", "throughput", "allocated_prbs"}
    if set(slice_state.keys()) != required:
        raise ValueError(f"{path} has invalid keys: {slice_state.keys()}")

    if not isinstance(slice_state["queue_size"], int):
        raise ValueError(f"{path}.queue_size must be int")
    if not isinstance(slice_state["allocated_prbs"], int):
        raise ValueError(f"{path}.allocated_prbs must be int")

    if not isinstance(slice_state["latency_ms"], (int, float)):
        raise ValueError(f"{path}.latency_ms must be float")
    if not isinstance(slice_state["throughput"], (int, float)):
        raise ValueError(f"{path}.throughput must be float")


def validate_sample_schema(sample: Dict[str, Any]) -> None:
    """Validate one sample against the strict schema and constraints."""
    required_top_keys = {
        "sample_id",
        "scenario_type",
        "intent",
        "network_state",
        "sla_status",
        "policy_output",
    }
    if set(sample.keys()) != required_top_keys:
        raise ValueError(f"Top-level keys mismatch: {sample.keys()}")

    try:
        uuid.UUID(sample["sample_id"])
    except Exception as exc:  # pragma: no cover - defensive guard
        raise ValueError("sample_id is not a valid UUID") from exc

    if sample["scenario_type"] not in ALL_SCENARIOS:
        raise ValueError(f"Invalid scenario_type: {sample['scenario_type']}")

    intent = sample["intent"]
    intent_keys = {"raw_text", "intent_type", "target_slice", "constraints", "priority"}
    if set(intent.keys()) != intent_keys:
        raise ValueError("intent keys mismatch")

    if intent["target_slice"] not in SLICES:
        raise ValueError("intent.target_slice must be one of URLLC/eMBB/IoT")
    if intent["priority"] not in {"high", "medium", "low"}:
        raise ValueError("intent.priority must be high/medium/low")

    constraints = intent["constraints"]
    if "latency_ms" not in constraints:
        raise ValueError("intent.constraints.latency_ms is required")
    if set(constraints.keys()) not in ({"latency_ms"}, {"latency_ms", "throughput"}):
        raise ValueError("intent.constraints allows only latency_ms and optional throughput")
    if not isinstance(constraints["latency_ms"], (int, float)):
        raise ValueError("intent.constraints.latency_ms must be float")
    if "throughput" in constraints and not isinstance(constraints["throughput"], (int, float)):
        raise ValueError("intent.constraints.throughput must be float")

    network_state = sample["network_state"]
    if set(network_state.keys()) != {"total_prbs", "slices"}:
        raise ValueError("network_state keys mismatch")
    if not isinstance(network_state["total_prbs"], int):
        raise ValueError("network_state.total_prbs must be int")

    slices_state = network_state["slices"]
    if set(slices_state.keys()) != set(SLICES):
        raise ValueError("network_state.slices keys mismatch")

    for slice_name in SLICES:
        _validate_slice_state(slices_state[slice_name], f"network_state.slices.{slice_name}")

    allocation_sum = sum(int(slices_state[s]["allocated_prbs"]) for s in SLICES)
    if allocation_sum != network_state["total_prbs"]:
        raise ValueError("network_state allocated_prbs must sum to total_prbs")

    sla_status = sample["sla_status"]
    if set(sla_status.keys()) != set(SLICES):
        raise ValueError("sla_status keys mismatch")
    for slice_name in SLICES:
        if sla_status[slice_name] not in {"satisfied", "violated"}:
            raise ValueError("sla_status values must be satisfied/violated")

    policy_output = sample["policy_output"]
    if set(policy_output.keys()) != {"SMO", "RIC", "CU", "DU", "RU"}:
        raise ValueError("policy_output keys mismatch")

    slice_priority = policy_output["SMO"]["slice_priority"]
    if set(slice_priority.keys()) != set(SLICES):
        raise ValueError("SMO.slice_priority keys mismatch")

    ric_prbs = policy_output["RIC"]["prb_allocation"]
    if set(ric_prbs.keys()) != set(SLICES):
        raise ValueError("RIC.prb_allocation keys mismatch")
    if any(not isinstance(ric_prbs[s], int) for s in SLICES):
        raise ValueError("RIC.prb_allocation values must be ints")
    if any(ric_prbs[s] < MIN_PRBS_PER_SLICE for s in SLICES):
        raise ValueError("RIC.prb_allocation violates minimum PRBs per slice")
    if sum(ric_prbs[s] for s in SLICES) != network_state["total_prbs"]:
        raise ValueError("RIC.prb_allocation must sum to network_state.total_prbs")

    if not isinstance(policy_output["RIC"]["strategy"], str):
        raise ValueError("RIC.strategy must be string")

    if not isinstance(policy_output["CU"]["handover_mode"], str):
        raise ValueError("CU.handover_mode must be string")
    if not isinstance(policy_output["CU"]["bearer_priority"], str):
        raise ValueError("CU.bearer_priority must be string")

    if policy_output["DU"]["scheduler"] not in {"priority", "proportional_fair"}:
        raise ValueError("DU.scheduler must be priority or proportional_fair")

    queue_weights = policy_output["DU"]["queue_weights"]
    if set(queue_weights.keys()) != set(SLICES):
        raise ValueError("DU.queue_weights keys mismatch")
    if any(not isinstance(queue_weights[s], (int, float)) for s in SLICES):
        raise ValueError("DU.queue_weights values must be float")

    total_queue_weight = float(sum(float(queue_weights[s]) for s in SLICES))
    if abs(total_queue_weight - 1.0) > 0.02:
        raise ValueError("DU.queue_weights must approximately sum to 1.0")

    if not isinstance(policy_output["RU"]["power_bias"], str):
        raise ValueError("RU.power_bias must be string")


def validate_dataset(dataset: List[Dict[str, Any]]) -> None:
    """Validate full dataset schema, scenario coverage, and distribution."""
    if not isinstance(dataset, list) or not dataset:
        raise ValueError("dataset must be a non-empty list")

    for sample in dataset:
        validate_sample_schema(sample)

    scenario_counter = Counter(sample["scenario_type"] for sample in dataset)
    expected = expected_scenario_distribution(len(dataset))

    if scenario_counter != Counter(expected):
        raise ValueError(
            "Scenario distribution mismatch. "
            f"Expected {expected}, found {dict(scenario_counter)}"
        )

    if len(dataset) >= len(ALL_SCENARIOS):
        missing = [scenario for scenario in ALL_SCENARIOS if scenario_counter[scenario] == 0]
        if missing:
            raise ValueError(f"Missing scenarios in dataset: {missing}")


def generate_dataset(total_samples: int = 2000, progress_every: int = 100) -> List[Dict[str, Any]]:
    """Generate a complete dataset and validate schema/distribution constraints."""
    if total_samples <= 0:
        raise ValueError("total_samples must be > 0")

    scenario_plan = build_scenario_plan(total_samples)
    dataset: List[Dict[str, Any]] = []

    for idx, scenario_type in enumerate(scenario_plan, start=1):
        dataset.append(generate_sample(idx, scenario_type))

        if progress_every > 0 and idx % progress_every == 0:
            print(f"[progress] generated {idx}/{total_samples} samples")

    validate_dataset(dataset)
    return dataset
