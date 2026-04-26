"""
data_structures.py
------------------
Pure data containers shared across all modules.
No logic here — just typed fields.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


# ---------------------------------------------------------------------------
# Detection  (single frame output from YOLO)
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    label: str            # COCO label string, e.g. "car"
    project_class: str    # Internal class: light/medium/heavy_vehicle
    confidence: float
    bbox: Tuple[float, float, float, float]   # (x1, y1, x2, y2) pixels
    centroid: Tuple[float, float]             # (cx, cy) pixels
    class_id: int         # COCO numeric class id


# ---------------------------------------------------------------------------
# Track  (persistent across frames while a vehicle is visible)
# ---------------------------------------------------------------------------
@dataclass
class Track:
    track_id: int
    project_class: str    # light / medium / heavy _vehicle

    # Voting on label (to smooth noisy class flips between frames)
    label_votes: Dict[str, int] = field(default_factory=dict)

    # Centroid history: list of (timestamp_seconds, cx, cy)
    positions: List[Tuple[float, float, float]] = field(default_factory=list)

    first_seen: float = 0.0   # timestamp (seconds since epoch)
    last_seen:  float = 0.0   # timestamp of last matched detection

    # One of: UNKNOWN / MOVING / STATIONARY / LOST
    state: str = "UNKNOWN"

    # Whether this track has already been counted in the current window
    is_counted_in_current_window: bool = False

    # Accumulated durations (carry across windows for the same track)
    missing_seconds:    float = 0.0

    # Internal: timestamp when the track last transitioned into STATIONARY
    _stationary_since: float = 0.0

    # Internal: timestamp of the last update (used to compute dt)
    _last_update_time: float = 0.0


# ---------------------------------------------------------------------------
# WindowFeatures  (one row written to CSV / JSONL per window)
# ---------------------------------------------------------------------------
@dataclass
class WindowFeatures:
    window_start: float   # epoch seconds
    window_end:   float
    window_seconds: float # actual duration of this window

    # --- Total unique vehicles seen during the window (counted once each) --
    count_motorcycle: int = 0   # light_vehicle
    count_car:        int = 0   # medium_vehicle
    count_heavy:      int = 0   # heavy_vehicle (bus + truck)

    # --- Mean vehicles visible per minute for each type --------------------
    # Computed as: (sum of per-frame counts) / (window_minutes)
    mean_per_min_motorcycle: float = 0.0
    mean_per_min_car:        float = 0.0
    mean_per_min_heavy:      float = 0.0

    # --- FPS info ----------------------------------------------------------
    fps_mean: float = 0.0

    def to_dict(self) -> dict:
        """Serialise to a plain dict (for CSV / JSONL output)."""
        return {
            "window_start":          self.window_start,
            "window_end":            self.window_end,
            "window_seconds":        self.window_seconds,
            "count_motorcycle":      self.count_motorcycle,
            "count_car":             self.count_car,
            "count_heavy":           self.count_heavy,
            "mean_per_min_motorcycle": round(self.mean_per_min_motorcycle, 2),
            "mean_per_min_car":        round(self.mean_per_min_car,        2),
            "mean_per_min_heavy":      round(self.mean_per_min_heavy,      2),
            "fps_mean":              round(self.fps_mean, 2),
        }