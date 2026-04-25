"""
visualization.py
----------------
Optional OpenCV drawing utilities.
The entire pipeline runs correctly without calling anything in this module.
Call draw_frame() to add an informational overlay for debugging / demo.

Privacy note: We only label vehicle type and movement state.
              No face, person, or license-plate boxes are ever drawn.
"""

from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np

from data_structures import Detection, Track


# Colour palette per project class
CLASS_COLORS = {
    "light_vehicle":  (0,  200, 255),   # yellow-orange
    "medium_vehicle": (0,  255,  80),   # green
    "heavy_vehicle":  (0,   80, 255),   # red-orange
}

STATE_COLORS = {
    "MOVING":     (50,  255, 50),    # bright green
    "STATIONARY": (50,  50,  200),   # blue
    "UNKNOWN":    (180, 180, 180),   # grey
    "LOST":       (80,  80,   80),   # dark grey
}

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
THICKNESS  = 1


def draw_frame(
    frame: np.ndarray,
    active_tracks: Dict[int, Track],
    detections_by_track: Optional[Dict[int, Detection]],
    exposure_score: float,
    exposure_category: str,
    fps: float,
    window_elapsed_s: float,
) -> np.ndarray:
    """
    Draw bounding boxes, track info, and the exposure HUD onto `frame`.
    Returns the annotated frame (in-place modification + return).

    Parameters
    ----------
    frame                  : BGR numpy array (modified in place)
    active_tracks          : non-LOST track dict from CentroidTracker
    detections_by_track    : optional map track_id → Detection (for bbox)
    exposure_score         : current window exposure score (0-100)
    exposure_category      : LOW / MEDIUM / HIGH
    fps                    : measured FPS this frame
    window_elapsed_s       : seconds since window start
    """

    # --- Draw bounding boxes + labels for each tracked vehicle ------------
    if detections_by_track:
        for tid, det in detections_by_track.items():
            tr = active_tracks.get(tid)
            if tr is None:
                continue

            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            state  = tr.state
            color  = STATE_COLORS.get(state, (200, 200, 200))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, THICKNESS + 1)

            label_text = (
                f"[{tid}] {tr.project_class.split('_')[0].upper()} "
                f"{'▶' if state == 'MOVING' else '■'}"
            )
            (tw, th), _ = cv2.getTextSize(label_text, FONT, FONT_SCALE, THICKNESS)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                frame, label_text,
                (x1 + 2, y1 - 3),
                FONT, FONT_SCALE, (0, 0, 0), THICKNESS, cv2.LINE_AA
            )

    # --- Exposure HUD (top-left panel) ------------------------------------
    hud_lines = [
        f"Exposure: {exposure_score:.1f}  [{exposure_category}]",
        f"FPS: {fps:.1f}",
        f"Window: {window_elapsed_s:.0f}s",
    ]
    _draw_hud(frame, hud_lines, exposure_category)

    return frame


def _draw_hud(frame: np.ndarray, lines: list, category: str) -> None:
    """Draw a small overlay panel in the top-left corner."""
    category_bg = {
        "LOW":    (0,  160, 0),
        "MEDIUM": (0,  140, 220),
        "HIGH":   (0,  0,   200),
    }
    bg_color = category_bg.get(category, (60, 60, 60))

    pad_x, pad_y = 8, 6
    line_h = 18
    panel_h = pad_y * 2 + line_h * len(lines)
    panel_w = 260

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), bg_color, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, line in enumerate(lines):
        y = pad_y + (i + 1) * line_h - 2
        cv2.putText(
            frame, line,
            (pad_x, y),
            FONT, FONT_SCALE + 0.05, (255, 255, 255), THICKNESS + 1, cv2.LINE_AA
        )


def draw_legend(frame: np.ndarray) -> np.ndarray:
    """Draw a small colour legend in the bottom-left corner."""
    items = [
        ("MOVING",     STATE_COLORS["MOVING"]),
        ("STATIONARY", STATE_COLORS["STATIONARY"]),
        ("UNKNOWN",    STATE_COLORS["UNKNOWN"]),
    ]
    h, w = frame.shape[:2]
    bx, by = 8, h - 10 - 18 * len(items)
    for i, (text, color) in enumerate(items):
        y = by + i * 18
        cv2.putText(frame, text, (bx, y), FONT, FONT_SCALE, color, THICKNESS + 1, cv2.LINE_AA)
    return frame