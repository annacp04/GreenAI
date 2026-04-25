"""
exposure_score.py
-----------------
Compute the node-level traffic exposure score (0–100) and category.

Formula (heuristic, hackathon MVP):
  flow_score    = min(FLOW_MAX,    FLOW_MAX    * weighted_unique_moving / FLOW_REF)
  density_score = min(DENSITY_MAX, DENSITY_MAX * (weighted_moving_seconds / window_s) / DENSITY_REF)
  peak_score    = min(PEAK_MAX,    PEAK_MAX    * max_weighted_moving_visible / PEAK_REF)
  total         = flow_score + density_score + peak_score   ∈ [0, 100]

Categories:
  0  ≤ score < 30  → LOW
  30 ≤ score < 65  → MEDIUM
  65 ≤ score ≤ 100 → HIGH

All constants are in config.py so they can be tuned without touching this file.
"""

from __future__ import annotations

from typing import Tuple

from config import (
    FLOW_MAX, FLOW_REF,
    DENSITY_MAX, DENSITY_REF,
    PEAK_MAX, PEAK_REF,
    EXPOSURE_LOW_MAX, EXPOSURE_MEDIUM_MAX,
    VEHICLE_WEIGHTS,
)


def compute_weighted_moving_visible(
    current_light:  int,
    current_medium: int,
    current_heavy:  int,
) -> float:
    """
    Instantaneous weighted count of moving vehicles visible in a frame.
    Used for real-time HUD display and to update max_weighted_moving_visible.
    """
    return (
        VEHICLE_WEIGHTS["light_vehicle"]  * current_light
        + VEHICLE_WEIGHTS["medium_vehicle"] * current_medium
        + VEHICLE_WEIGHTS["heavy_vehicle"]  * current_heavy
    )


def compute_weighted_unique_moving(
    unique_light:  int,
    unique_medium: int,
    unique_heavy:  int,
) -> float:
    """
    Weighted count of unique vehicles that were moving during a window.
    """
    return (
        VEHICLE_WEIGHTS["light_vehicle"]  * unique_light
        + VEHICLE_WEIGHTS["medium_vehicle"] * unique_medium
        + VEHICLE_WEIGHTS["heavy_vehicle"]  * unique_heavy
    )


def compute_weighted_moving_seconds(
    moving_secs_light:  float,
    moving_secs_medium: float,
    moving_secs_heavy:  float,
) -> float:
    """
    Total weighted vehicle-seconds of movement during the window.
    e.g. 1 bus moving for 10 s = 55 weighted vehicle-seconds
    """
    return (
        VEHICLE_WEIGHTS["light_vehicle"]  * moving_secs_light
        + VEHICLE_WEIGHTS["medium_vehicle"] * moving_secs_medium
        + VEHICLE_WEIGHTS["heavy_vehicle"]  * moving_secs_heavy
    )


def compute_flow_score(weighted_unique_moving: float) -> float:
    return min(FLOW_MAX, FLOW_MAX * weighted_unique_moving / FLOW_REF)


def compute_density_score(weighted_moving_seconds: float, window_seconds: float) -> float:
    if window_seconds <= 0:
        return 0.0
    avg = weighted_moving_seconds / window_seconds
    return min(DENSITY_MAX, DENSITY_MAX * avg / DENSITY_REF)


def compute_peak_score(max_weighted_moving_visible: float) -> float:
    return min(PEAK_MAX, PEAK_MAX * max_weighted_moving_visible / PEAK_REF)


def score_to_category(score: float) -> str:
    if score < EXPOSURE_LOW_MAX:
        return "LOW"
    elif score < EXPOSURE_MEDIUM_MAX:
        return "MEDIUM"
    else:
        return "HIGH"


def compute_exposure_score(
    unique_light:  int,
    unique_medium: int,
    unique_heavy:  int,
    weighted_moving_seconds: float,
    max_weighted_moving_visible: float,
    window_seconds: float,
) -> Tuple[float, str]:
    """
    Main entry point.  Returns (score: float, category: str).
    score ∈ [0.0, 100.0]
    category ∈ {"LOW", "MEDIUM", "HIGH"}
    """
    weighted_unique = compute_weighted_unique_moving(unique_light, unique_medium, unique_heavy)

    flow    = compute_flow_score(weighted_unique)
    density = compute_density_score(weighted_moving_seconds, window_seconds)
    peak    = compute_peak_score(max_weighted_moving_visible)

    score    = flow + density + peak
    score    = min(100.0, max(0.0, score))
    category = score_to_category(score)

    return score, category