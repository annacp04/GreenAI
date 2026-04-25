"""
traffic_aggregator.py
---------------------
Accumulates per-frame track information and summarises it into
WindowFeatures at the end of each WINDOW_SECONDS interval.

Key rules:
  - A track is counted as a "unique mover" only ONCE per window,
    the first time it transitions to MOVING.
  - Stationary vehicles contribute 0 to the exposure score.
  - If a vehicle starts moving mid-window, it is counted from that moment.
  - vehicle-seconds are accumulated every frame using the elapsed dt.
  - Window counters reset at the start of each new window;
    live Track objects are NOT reset (they persist across windows).
"""

from __future__ import annotations

import time
from typing import Dict, List

from config import (
    VEHICLE_WEIGHTS,
    WINDOW_SECONDS,
)
from data_structures import Track, WindowFeatures
from exposure_score import compute_exposure_score


class TrafficAggregator:

    def __init__(self, window_seconds: float = WINDOW_SECONDS) -> None:
        self.window_seconds = window_seconds
        self._window_start: float = time.time()

        # --- unique movers seen this window ---------------------------------
        self._unique_light_moving:  int = 0
        self._unique_medium_moving: int = 0
        self._unique_heavy_moving:  int = 0

        # --- accumulated moving seconds this window -------------------------
        self._moving_secs_light:  float = 0.0
        self._moving_secs_medium: float = 0.0
        self._moving_secs_heavy:  float = 0.0

        # --- accumulated stationary seconds this window --------------------
        self._stationary_secs: float = 0.0

        # --- frame-level stats for mean/max calculation --------------------
        self._frame_moving_counts:   List[int]   = []  # raw moving vehicle count per frame
        self._frame_weighted_moving: List[float] = []  # weighted moving count per frame

        # --- FPS tracking --------------------------------------------------
        self._fps_samples: List[float] = []

        self._last_update_time: float = time.time()

    # ------------------------------------------------------------------
    def update(
        self,
        active_tracks: Dict[int, Track],
        timestamp: float,
        fps: float,
    ) -> None:
        """
        Called every frame.
        active_tracks: dict of non-LOST tracks from the tracker.
        timestamp:     current time (seconds, from time.time()).
        fps:           measured FPS this frame.
        """
        dt = timestamp - self._last_update_time
        # Guard against negative or huge dt (e.g. first frame)
        if dt < 0 or dt > 5.0:
            dt = 0.0
        self._last_update_time = timestamp

        self._fps_samples.append(fps)

        # --- Counters for this frame's snapshot ----------------------------
        current_light  = 0
        current_medium = 0
        current_heavy  = 0

        for tr in active_tracks.values():
            weight = VEHICLE_WEIGHTS.get(tr.project_class, 1.0)

            if tr.state == "MOVING":
                # Accumulate moving vehicle-seconds
                if tr.project_class == "light_vehicle":
                    self._moving_secs_light  += dt
                    current_light += 1
                elif tr.project_class == "medium_vehicle":
                    self._moving_secs_medium += dt
                    current_medium += 1
                elif tr.project_class == "heavy_vehicle":
                    self._moving_secs_heavy  += dt
                    current_heavy += 1

                # Count as unique mover if first time this window
                if not tr.is_counted_in_current_window:
                    tr.is_counted_in_current_window = True
                    if tr.project_class == "light_vehicle":
                        self._unique_light_moving += 1
                    elif tr.project_class == "medium_vehicle":
                        self._unique_medium_moving += 1
                    elif tr.project_class == "heavy_vehicle":
                        self._unique_heavy_moving += 1

            elif tr.state == "STATIONARY":
                self._stationary_secs += dt

        # Frame snapshot
        total_moving = current_light + current_medium + current_heavy
        weighted_moving = (
            0.65 * current_light
            + 1.0  * current_medium
            + 5.5  * current_heavy
        )
        self._frame_moving_counts.append(total_moving)
        self._frame_weighted_moving.append(weighted_moving)

    # ------------------------------------------------------------------
    def get_live_snapshot(self, active_tracks: Dict[int, Track]) -> dict:
        """
        Returns a lightweight dict with current-frame moving/stationary counts
        and the live weighted_moving_visible for real-time display.
        """
        counts = {"light": 0, "medium": 0, "heavy": 0,
                  "stat_light": 0, "stat_medium": 0, "stat_heavy": 0}
        for tr in active_tracks.values():
            if tr.state == "MOVING":
                if tr.project_class == "light_vehicle":   counts["light"]  += 1
                elif tr.project_class == "medium_vehicle": counts["medium"] += 1
                elif tr.project_class == "heavy_vehicle":  counts["heavy"]  += 1
            elif tr.state == "STATIONARY":
                if tr.project_class == "light_vehicle":   counts["stat_light"]  += 1
                elif tr.project_class == "medium_vehicle": counts["stat_medium"] += 1
                elif tr.project_class == "heavy_vehicle":  counts["stat_heavy"]  += 1

        weighted_moving_visible = (
            0.65 * counts["light"]
            + 1.0  * counts["medium"]
            + 5.5  * counts["heavy"]
        )
        return {**counts, "weighted_moving_visible": weighted_moving_visible}

    # ------------------------------------------------------------------
    def close_window(
        self,
        active_tracks: Dict[int, Track],
        window_end: float,
    ) -> WindowFeatures:
        """
        Finalise and return a WindowFeatures for the elapsed window.
        Resets internal accumulators. Does NOT reset Track objects.
        """
        # --- Snapshot: current moving / stationary counts -----------------
        current_light  = sum(1 for tr in active_tracks.values()
                             if tr.state == "MOVING" and tr.project_class == "light_vehicle")
        current_medium = sum(1 for tr in active_tracks.values()
                             if tr.state == "MOVING" and tr.project_class == "medium_vehicle")
        current_heavy  = sum(1 for tr in active_tracks.values()
                             if tr.state == "MOVING" and tr.project_class == "heavy_vehicle")

        stat_light  = sum(1 for tr in active_tracks.values()
                          if tr.state == "STATIONARY" and tr.project_class == "light_vehicle")
        stat_medium = sum(1 for tr in active_tracks.values()
                          if tr.state == "STATIONARY" and tr.project_class == "medium_vehicle")
        stat_heavy  = sum(1 for tr in active_tracks.values()
                          if tr.state == "STATIONARY" and tr.project_class == "heavy_vehicle")

        # --- Weighted moving seconds --------------------------------------
        weighted_moving_secs = (
            0.65 * self._moving_secs_light
            + 1.0  * self._moving_secs_medium
            + 5.5  * self._moving_secs_heavy
        )

        # --- Weighted unique movers ---------------------------------------
        total_moving_weighted = (
            0.65 * self._unique_light_moving
            + 1.0  * self._unique_medium_moving
            + 5.5  * self._unique_heavy_moving
        )

        # --- Frame-level stats --------------------------------------------
        mean_moving = (
            sum(self._frame_moving_counts) / len(self._frame_moving_counts)
            if self._frame_moving_counts else 0.0
        )
        max_moving = max(self._frame_moving_counts) if self._frame_moving_counts else 0
        max_weighted = max(self._frame_weighted_moving) if self._frame_weighted_moving else 0.0

        # --- FPS ----------------------------------------------------------
        fps_mean = (
            sum(self._fps_samples) / len(self._fps_samples)
            if self._fps_samples else 0.0
        )

        # --- Exposure score -----------------------------------------------
        score, category = compute_exposure_score(
            unique_light=self._unique_light_moving,
            unique_medium=self._unique_medium_moving,
            unique_heavy=self._unique_heavy_moving,
            weighted_moving_seconds=weighted_moving_secs,
            max_weighted_moving_visible=max_weighted,
            window_seconds=self.window_seconds,
        )

        wf = WindowFeatures(
            window_start=self._window_start,
            window_end=window_end,
            unique_light_moving=self._unique_light_moving,
            unique_medium_moving=self._unique_medium_moving,
            unique_heavy_moving=self._unique_heavy_moving,
            current_light_moving=current_light,
            current_medium_moving=current_medium,
            current_heavy_moving=current_heavy,
            stationary_light_count=stat_light,
            stationary_medium_count=stat_medium,
            stationary_heavy_count=stat_heavy,
            moving_vehicle_seconds_light=self._moving_secs_light,
            moving_vehicle_seconds_medium=self._moving_secs_medium,
            moving_vehicle_seconds_heavy=self._moving_secs_heavy,
            stationary_vehicle_seconds=self._stationary_secs,
            total_moving_weighted_count=total_moving_weighted,
            weighted_moving_seconds=weighted_moving_secs,
            mean_moving_vehicles_visible=round(mean_moving, 2),
            max_moving_vehicles_visible=max_moving,
            max_weighted_moving_visible=round(max_weighted, 2),
            traffic_exposure_score=round(score, 2),
            exposure_category=category,
            fps_mean=round(fps_mean, 2),
        )

        # --- Reset window accumulators (NOT the Track objects) ------------
        self._reset_window(window_end, active_tracks)

        return wf

    # ------------------------------------------------------------------
    def _reset_window(self, new_start: float, active_tracks: Dict[int, Track]) -> None:
        self._window_start = new_start
        self._unique_light_moving  = 0
        self._unique_medium_moving = 0
        self._unique_heavy_moving  = 0
        self._moving_secs_light  = 0.0
        self._moving_secs_medium = 0.0
        self._moving_secs_heavy  = 0.0
        self._stationary_secs    = 0.0
        self._frame_moving_counts   = []
        self._frame_weighted_moving = []
        self._fps_samples = []

        # Reset "counted this window" flag on all live tracks
        for tr in active_tracks.values():
            tr.is_counted_in_current_window = False

    # ------------------------------------------------------------------
    @property
    def window_start(self) -> float:
        return self._window_start

    def seconds_since_window_start(self, now: float) -> float:
        return now - self._window_start