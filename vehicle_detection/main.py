"""
main.py
-------
EdgeAI Green Routes – vehicle detector node.
Runs on the Linux/Python side of the Arduino UNO Q.

Usage examples
--------------
# Webcam, display on screen:
  python main.py --source 0 --display

# Video file, headless, CSV + JSONL output:
  python main.py --video sample_traffic.mp4 --headless \
                 --output_csv traffic_features.csv \
                 --output_json traffic_features.jsonl

# Use YOLO11n instead of YOLOv8n:
  python main.py --model yolo11n.pt --video sample_traffic.mp4 --display

# Save annotated demo video:
  python main.py --video sample_traffic.mp4 --save_annotated_video output.mp4

# Smaller image size for speed on UNO Q:
  python main.py --source 0 --imgsz 320 --headless
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Optional

import cv2

from config import (
    DEFAULT_MODEL,
    IMAGE_SIZE,
    CONFIDENCE_THRESHOLD,
    WINDOW_SECONDS,
    TERMINAL_PRINT_INTERVAL_SEC,
)
from data_structures import Detection, Track, WindowFeatures
from exposure_score import compute_exposure_score, compute_weighted_moving_visible
from traffic_aggregator import TrafficAggregator
from tracker import CentroidTracker
from visualization import draw_frame, draw_legend
from yolo_detector import YOLODetector


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EdgeAI Green Routes – vehicle detector node"
    )
    p.add_argument("--model",   default=DEFAULT_MODEL,
                   help="YOLO model weights (yolov8n.pt or yolo11n.pt)")
    p.add_argument("--source",  default=None, type=int,
                   help="Webcam device index (e.g. 0)")
    p.add_argument("--video",   default=None,
                   help="Path to input video file")
    p.add_argument("--output_csv",  default=None,
                   help="Path for CSV window summaries")
    p.add_argument("--output_json", default=None,
                   help="Path for JSONL window summaries")
    p.add_argument("--display",  action="store_true",
                   help="Show live OpenCV window")
    p.add_argument("--headless", action="store_true",
                   help="Run without any display (overrides --display)")
    p.add_argument("--imgsz",   default=IMAGE_SIZE, type=int,
                   help="YOLO inference image size (640, 416, 320)")
    p.add_argument("--conf",    default=CONFIDENCE_THRESHOLD, type=float,
                   help="YOLO confidence threshold")
    p.add_argument("--window_seconds", default=WINDOW_SECONDS, type=float,
                   help="Aggregation window duration in seconds")
    p.add_argument("--use_yolo_tracker", action="store_true",
                   help="Use Ultralytics built-in tracker (ByteTrack/BoT-SORT)")
    p.add_argument("--save_annotated_video", default=None,
                   help="Save annotated frames to this video file")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def init_csv(path: str) -> tuple:
    """Open a CSV file and write the header row. Returns (file, writer)."""
    fh = open(path, "w", newline="")
    fieldnames = list(WindowFeatures(0, 0).to_dict().keys())
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    return fh, writer


def write_csv_row(writer, wf: WindowFeatures) -> None:
    writer.writerow(wf.to_dict())


def write_jsonl_row(fh, wf: WindowFeatures) -> None:
    fh.write(json.dumps(wf.to_dict()) + "\n")
    fh.flush()


def print_window_summary(wf: WindowFeatures) -> None:
    print(
        f"\n{'='*60}\n"
        f"  WINDOW SUMMARY  {time.strftime('%H:%M:%S', time.localtime(wf.window_start))}"
        f" → {time.strftime('%H:%M:%S', time.localtime(wf.window_end))}\n"
        f"  Unique movers  : light={wf.unique_light_moving}  "
        f"medium={wf.unique_medium_moving}  heavy={wf.unique_heavy_moving}\n"
        f"  Moving now     : light={wf.current_light_moving}  "
        f"medium={wf.current_medium_moving}  heavy={wf.current_heavy_moving}\n"
        f"  Stationary now : light={wf.stationary_light_count}  "
        f"medium={wf.stationary_medium_count}  heavy={wf.stationary_heavy_count}\n"
        f"  Weighted mov.s : {wf.weighted_moving_seconds:.1f} vehicle-seconds\n"
        f"  Peak weighted  : {wf.max_weighted_moving_visible:.2f}\n"
        f"  ► EXPOSURE     : {wf.traffic_exposure_score:.1f} / 100  [{wf.exposure_category}]\n"
        f"  FPS mean       : {wf.fps_mean:.1f}\n"
        f"{'='*60}"
    )


def print_live_status(
    timestamp: float,
    snapshot: dict,
    score: float,
    category: str,
    fps: float,
    window_elapsed: float,
) -> None:
    print(
        f"[{time.strftime('%H:%M:%S')}] "
        f"Moving: L={snapshot['light']} M={snapshot['medium']} H={snapshot['heavy']} | "
        f"Stationary: L={snapshot['stat_light']} M={snapshot['stat_medium']} H={snapshot['stat_heavy']} | "
        f"Score: {score:.1f} [{category}] | "
        f"FPS: {fps:.1f} | "
        f"Win: {window_elapsed:.0f}s"
    )


# ---------------------------------------------------------------------------
# Map active_tracks → detections (needed for bbox drawing)
# We rebuild this from YOLO output each frame.
# ---------------------------------------------------------------------------

def build_detection_by_track(
    active_tracks: Dict[int, Track],
    detections: List[Detection],
    tracker: CentroidTracker,
) -> Dict[int, Detection]:
    """
    Best-effort map of track_id → most-recent Detection.
    Used only for bounding-box drawing; not critical for pipeline logic.
    """
    result: Dict[int, Detection] = {}
    # Simple: pair by centroid proximity to last known position
    for det in detections:
        best_id   = None
        best_dist = 9999.0
        for tid, tr in active_tracks.items():
            if not tr.positions:
                continue
            _, tcx, tcy = tr.positions[-1]
            dist = ((det.centroid[0] - tcx) ** 2 + (det.centroid[1] - tcy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_id   = tid
        if best_id is not None and best_dist < 60:
            result[best_id] = det
    return result


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    show_display = args.display and not args.headless

    # --- Video source ------------------------------------------------------
    source = args.source if args.source is not None else args.video
    if source is None:
        print("[ERROR] Specify --source 0 for webcam or --video <path> for a file.")
        sys.exit(1)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        sys.exit(1)

    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[main] Source: {source}  Resolution: {frame_width}×{frame_height}  FPS: {source_fps:.1f}")

    # --- Output video writer -----------------------------------------------
    video_writer = None
    if args.save_annotated_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            args.save_annotated_video, fourcc, source_fps, (frame_width, frame_height)
        )
        print(f"[main] Saving annotated video to: {args.save_annotated_video}")

    # --- YOLO detector -----------------------------------------------------
    detector = YOLODetector(
        model_path=args.model,
        imgsz=args.imgsz,
        conf=args.conf,
    )

    # --- Tracker -----------------------------------------------------------
    tracker = CentroidTracker()
    print(f"[main] Tracker: {'YOLO built-in' if args.use_yolo_tracker else 'Centroid (default)'}")

    # --- Aggregator --------------------------------------------------------
    aggregator = TrafficAggregator(window_seconds=args.window_seconds)

    # --- Output files ------------------------------------------------------
    csv_fh = csv_writer = json_fh = None
    if args.output_csv:
        csv_fh, csv_writer = init_csv(args.output_csv)
        print(f"[main] CSV output: {args.output_csv}")
    if args.output_json:
        json_fh = open(args.output_json, "a")
        print(f"[main] JSONL output: {args.output_json}")

    # --- Loop state --------------------------------------------------------
    frame_count      = 0
    fps              = 0.0
    last_fps_time    = time.time()
    last_print_time  = time.time()
    last_score       = 0.0
    last_category    = "LOW"

    print("[main] Starting detection loop. Press 'q' to quit.")

    try:
        while True:
            t_frame_start = time.time()
            ret, frame = cap.read()
            if not ret:
                print("[main] End of stream or read error.")
                break

            frame_count += 1
            timestamp = time.time()

            # --- Detect ----------------------------------------------------
            if args.use_yolo_tracker:
                detections, yolo_ids = detector.detect_with_tracker(frame)
            else:
                detections = detector.detect(frame)
                yolo_ids   = None

            # --- Track -----------------------------------------------------
            active_tracks = tracker.update(detections, timestamp, yolo_ids)

            # --- Aggregate -------------------------------------------------
            aggregator.update(active_tracks, timestamp, fps)

            # --- Live snapshot for display / printing ----------------------
            snapshot = aggregator.get_live_snapshot(active_tracks)

            # Live score estimate (using partial window data)
            live_weighted = snapshot["weighted_moving_visible"]

            # --- Check if window is complete -------------------------------
            elapsed = aggregator.seconds_since_window_start(timestamp)
            if elapsed >= args.window_seconds:
                wf = aggregator.close_window(active_tracks, timestamp)
                last_score    = wf.traffic_exposure_score
                last_category = wf.exposure_category
                print_window_summary(wf)

                if csv_writer:
                    write_csv_row(csv_writer, wf)
                    if csv_fh:
                        csv_fh.flush()
                if json_fh:
                    write_jsonl_row(json_fh, wf)

            # --- FPS calculation -------------------------------------------
            t_frame_end = time.time()
            frame_time = t_frame_end - t_frame_start
            fps = 1.0 / frame_time if frame_time > 0 else 0.0

            # --- Periodic terminal print -----------------------------------
            if timestamp - last_print_time >= TERMINAL_PRINT_INTERVAL_SEC:
                print_live_status(
                    timestamp, snapshot,
                    last_score, last_category,
                    fps,
                    aggregator.seconds_since_window_start(timestamp),
                )
                last_print_time = timestamp

            # --- Visualisation --------------------------------------------
            if show_display or video_writer:
                det_by_track = build_detection_by_track(active_tracks, detections, tracker)
                annotated = draw_frame(
                    frame,
                    active_tracks,
                    det_by_track,
                    last_score,
                    last_category,
                    fps,
                    aggregator.seconds_since_window_start(timestamp),
                )
                draw_legend(annotated)

                if show_display:
                    cv2.imshow("EdgeAI Green Routes – vehicle detector", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print("[main] User pressed 'q'. Exiting.")
                        break

                if video_writer:
                    video_writer.write(annotated)

    finally:
        # --- Clean up ------------------------------------------------------
        cap.release()
        if video_writer:
            video_writer.release()
        if show_display:
            cv2.destroyAllWindows()
        if csv_fh:
            csv_fh.close()
        if json_fh:
            json_fh.close()

        print(f"\n[main] Done. Processed {frame_count} frames.")


if __name__ == "__main__":
    main()