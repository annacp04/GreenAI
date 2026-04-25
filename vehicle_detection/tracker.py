"""
tracker.py
----------
Vehicle tracker with two modes:

1. YOLO built-in tracker (ByteTrack/BoT-SORT via Ultralytics)
   Activated with --use_yolo_tracker.  Provides stable IDs from the model.

2. Fallback centroid tracker (default)
   Simple nearest-neighbour matching on centroids. Robust and lightweight.

For each track the module computes:
  - Recent speed (px/s) from the last few centroid positions
  - Total displacement from first seen position
  - Movement state: UNKNOWN → MOVING / STATIONARY → LOST

Privacy: track IDs are temporary local integers.
         They reset when a vehicle leaves the frame.
         No real-world identity is stored.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

from config import (
    MAX_MATCH_DISTANCE_PX,
    MOVE_THRESHOLD_PX,
    SPEED_THRESHOLD_PX_PER_SEC,
    STATIONARY_SECONDS,
    MAX_MISSING_SECONDS,
    SPEED_WINDOW_POSITIONS,
    VEHICLE_WEIGHTS,
)
from data_structures import Detection, Track


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _compute_speed(positions: list) -> float:
    """
    Compute speed (px/s) from the last SPEED_WINDOW_POSITIONS entries.
    positions: list of (timestamp, cx, cy)
    Returns 0.0 if not enough data.
    """
    if len(positions) < 2:
        return 0.0

    recent = positions[-SPEED_WINDOW_POSITIONS:]
    t0, x0, y0 = recent[0]
    t1, x1, y1 = recent[-1]
    dt = t1 - t0
    if dt <= 0:
        return 0.0
    dist = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    return dist / dt


def _compute_total_displacement(positions: list) -> float:
    """
    Straight-line displacement from first to last position.
    positions: list of (timestamp, cx, cy)
    """
    if len(positions) < 2:
        return 0.0
    _, x0, y0 = positions[0]
    _, x1, y1 = positions[-1]
    return math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)


# ---------------------------------------------------------------------------
# Centroid Tracker (fallback / default)
# ---------------------------------------------------------------------------

class CentroidTracker:
    """
    Frame-by-frame nearest-neighbour centroid tracker.
    Keeps track states and movement classification.
    """

    def __init__(self) -> None:
        self._tracks: Dict[int, Track] = {}      # track_id → Track
        self._next_id: int = 1
        self._yolo_id_map: Dict[int, int] = {}   # yolo_track_id → our track_id

    # ------------------------------------------------------------------
    def update(
        self,
        detections: List[Detection],
        timestamp: float,
        yolo_track_ids: Optional[List[Optional[int]]] = None,
    ) -> Dict[int, Track]:
        """
        Match detections to existing tracks.
        Update movement states.
        Returns the current dict of active (non-LOST) tracks.

        Parameters
        ----------
        detections      : list of Detection objects for this frame
        timestamp       : current time in seconds (time.time())
        yolo_track_ids  : optional list of YOLO-assigned IDs, parallel to detections
        """
        dt = 0.0  # will be set per track from _last_update_time

        # --- 1. Mark all active tracks as "not matched yet" ---------------
        unmatched_track_ids = set(
            tid for tid, tr in self._tracks.items() if tr.state != "LOST"
        )

        matched_track_ids:   set = set()
        new_detections:      List[Tuple[Detection, Optional[int]]] = []

        # --- 2. Match detections to tracks ---------------------------------
        if yolo_track_ids is not None:
            # Use YOLO IDs when available
            for det, yid in zip(detections, yolo_track_ids):
                if yid is not None and yid in self._yolo_id_map:
                    our_id = self._yolo_id_map[yid]
                    if our_id in self._tracks:
                        self._update_track(self._tracks[our_id], det, timestamp)
                        matched_track_ids.add(our_id)
                        continue
                # No YOLO ID or unknown — fall through to centroid matching
                new_detections.append((det, yid))
        else:
            new_detections = [(det, None) for det in detections]

        # Centroid matching for unmatched detections
        remaining_track_ids = unmatched_track_ids - matched_track_ids
        for det, yid in new_detections:
            best_id   = None
            best_dist = float("inf")
            det_cx, det_cy = det.centroid

            for tid in remaining_track_ids:
                tr = self._tracks[tid]
                if not tr.positions:
                    continue
                _, tcx, tcy = tr.positions[-1]
                dist = _euclidean((det_cx, det_cy), (tcx, tcy))
                if dist < best_dist and dist < MAX_MATCH_DISTANCE_PX:
                    best_dist = dist
                    best_id   = tid

            if best_id is not None:
                self._update_track(self._tracks[best_id], det, timestamp)
                matched_track_ids.add(best_id)
                remaining_track_ids.discard(best_id)
                # Register YOLO ID mapping if available
                if yid is not None:
                    self._yolo_id_map[yid] = best_id
            else:
                # New track
                new_id = self._next_id
                self._next_id += 1
                tr = Track(
                    track_id=new_id,
                    project_class=det.project_class,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    _last_update_time=timestamp,
                )
                tr.label_votes[det.label] = 1
                tr.positions.append((timestamp, det.centroid[0], det.centroid[1]))
                self._tracks[new_id] = tr
                matched_track_ids.add(new_id)
                if yid is not None:
                    self._yolo_id_map[yid] = new_id

        # --- 3. Handle unmatched (missing) tracks --------------------------
        for tid in unmatched_track_ids - matched_track_ids:
            tr = self._tracks[tid]
            dt = timestamp - tr._last_update_time
            tr.missing_seconds += dt
            tr._last_update_time = timestamp

            if tr.missing_seconds > MAX_MISSING_SECONDS:
                tr.state = "LOST"

        # --- 4. Update movement states for all matched tracks --------------
        for tid in matched_track_ids:
            tr = self._tracks[tid]
            self._classify_movement(tr, timestamp)

        # --- 5. Clean up YOLO ID map for LOST tracks ----------------------
        lost_ids = {tid for tid, tr in self._tracks.items() if tr.state == "LOST"}
        stale_yids = [yid for yid, oid in self._yolo_id_map.items() if oid in lost_ids]
        for yid in stale_yids:
            del self._yolo_id_map[yid]

        # Return only non-LOST tracks
        return {tid: tr for tid, tr in self._tracks.items() if tr.state != "LOST"}

    # ------------------------------------------------------------------
    def _update_track(self, tr: Track, det: Detection, timestamp: float) -> None:
        """Update an existing track with a new detection."""
        dt = timestamp - tr._last_update_time

        # Update label votes (majority vote smoothing)
        tr.label_votes[det.label] = tr.label_votes.get(det.label, 0) + 1
        # Update project_class to majority vote label
        dominant_label = max(tr.label_votes, key=tr.label_votes.get)
        from config import LABEL_TO_PROJECT_CLASS
        tr.project_class = LABEL_TO_PROJECT_CLASS.get(dominant_label, tr.project_class)

        tr.positions.append((timestamp, det.centroid[0], det.centroid[1]))
        # Keep position history bounded (last 60 entries ≈ ~2 min at 30fps is fine
        # but on UNO Q at ~5fps 300 entries = 60 s, which is our window)
        if len(tr.positions) > 300:
            tr.positions = tr.positions[-300:]

        tr.last_seen = timestamp
        tr.missing_seconds = 0.0
        tr._last_update_time = timestamp

    # ------------------------------------------------------------------
    def _classify_movement(self, tr: Track, timestamp: float) -> None:
        """
        Decide MOVING / STATIONARY based on recent speed and total displacement.
        Transition rules:
          UNKNOWN   → MOVING    if speed > threshold OR displacement > threshold
          UNKNOWN   → STATIONARY if speed < threshold for STATIONARY_SECONDS
          MOVING    → STATIONARY if speed < threshold for STATIONARY_SECONDS
          STATIONARY→ MOVING    if speed > threshold again
        """
        speed = _compute_speed(tr.positions)
        disp  = _compute_total_displacement(tr.positions)

        is_fast = (
            speed > SPEED_THRESHOLD_PX_PER_SEC
            or disp > MOVE_THRESHOLD_PX
        )

        if is_fast:
            # Vehicle is (or became) moving
            if tr.state != "MOVING":
                tr.state = "MOVING"
                tr._stationary_since = 0.0
        else:
            # Vehicle appears stationary
            if tr._stationary_since == 0.0:
                tr._stationary_since = timestamp
            time_stationary = timestamp - tr._stationary_since

            if time_stationary >= STATIONARY_SECONDS:
                tr.state = "STATIONARY"
            elif tr.state == "UNKNOWN":
                # Still warming up — keep UNKNOWN (weight 0)
                pass
            # If previously MOVING and now slowing, keep MOVING until threshold
            # exceeded — this prevents brief stops counting as parked
            elif tr.state == "MOVING" and time_stationary < STATIONARY_SECONDS:
                pass   # still MOVING until STATIONARY_SECONDS elapses

    # ------------------------------------------------------------------
    def get_all_tracks(self) -> Dict[int, Track]:
        return self._tracks

    def reset_window_flags(self) -> None:
        """Reset per-window aggregation flags on all active tracks."""
        for tr in self._tracks.values():
            tr.is_counted_in_current_window = False