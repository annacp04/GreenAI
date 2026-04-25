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

    # Aggregation helpers (reset each window)
    is_counted_in_current_window: bool = False

    # Accumulated durations (carry across windows for the same track)
    moving_seconds:     float = 0.0
    stationary_seconds: float = 0.0
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
    window_start: float    # epoch seconds
    window_end:   float

    # Unique vehicles that were MOVING at least once during the window
    unique_light_moving:  int = 0
    unique_medium_moving: int = 0
    unique_heavy_moving:  int = 0

    # Vehicles moving RIGHT NOW at window close (snapshot)
    current_light_moving:  int = 0
    current_medium_moving: int = 0
    current_heavy_moving:  int = 0

    # Vehicles stationary RIGHT NOW at window close
    stationary_light_count:  int = 0
    stationary_medium_count: int = 0
    stationary_heavy_count:  int = 0

    # Total seconds each class spent MOVING during the window
    moving_vehicle_seconds_light:  float = 0.0
    moving_vehicle_seconds_medium: float = 0.0
    moving_vehicle_seconds_heavy:  float = 0.0

    # Total seconds all vehicles spent STATIONARY during the window
    stationary_vehicle_seconds: float = 0.0

    # Weighted aggregates (used for exposure score)
    total_moving_weighted_count:  float = 0.0   # sum of weights of unique movers
    weighted_moving_seconds:       float = 0.0   # sum(weight * moving_seconds)

    # Statistics over frames within the window
    mean_moving_vehicles_visible: float = 0.0
    max_moving_vehicles_visible:  int   = 0
    max_weighted_moving_visible:  float = 0.0

    # Exposure
    traffic_exposure_score: float = 0.0
    exposure_category: str = "LOW"

    # Performance
    fps_mean: float = 0.0

    def to_dict(self) -> dict:
        """Serialise to a plain dict (for CSV / JSONL output)."""
        return {
            "window_start": self.window_start,
            "window_end":   self.window_end,
            "unique_light_moving":  self.unique_light_moving,
            "unique_medium_moving": self.unique_medium_moving,
            "unique_heavy_moving":  self.unique_heavy_moving,
            "current_light_moving":  self.current_light_moving,
            "current_medium_moving": self.current_medium_moving,
            "current_heavy_moving":  self.current_heavy_moving,
            "stationary_light_count":  self.stationary_light_count,
            "stationary_medium_count": self.stationary_medium_count,
            "stationary_heavy_count":  self.stationary_heavy_count,
            "moving_vehicle_seconds_light":  self.moving_vehicle_seconds_light,
            "moving_vehicle_seconds_medium": self.moving_vehicle_seconds_medium,
            "moving_vehicle_seconds_heavy":  self.moving_vehicle_seconds_heavy,
            "stationary_vehicle_seconds":    self.stationary_vehicle_seconds,
            "total_moving_weighted_count":   self.total_moving_weighted_count,
            "weighted_moving_seconds":        self.weighted_moving_seconds,
            "mean_moving_vehicles_visible":  self.mean_moving_vehicles_visible,
            "max_moving_vehicles_visible":   self.max_moving_vehicles_visible,
            "max_weighted_moving_visible":   self.max_weighted_moving_visible,
            "traffic_exposure_score": self.traffic_exposure_score,
            "exposure_category":      self.exposure_category,
            "fps_mean":               self.fps_mean,
        }