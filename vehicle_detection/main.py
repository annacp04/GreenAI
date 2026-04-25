"""
main.py
-------
EdgeAI Green Routes – vehicle detector node.
Corre en el lado Linux/Python del Arduino UNO Q.

Salida por ventana de tiempo:
  - count_motorcycle   : motos únicas vistas
  - count_car          : coches únicos vistos
  - count_heavy        : buses/camiones únicos vistos
  - mean_per_min_*     : media de vehículos visibles por minuto de cada tipo

Uso
---
  python main.py --source 0 --display                        # webcam
  python main.py --video clip.mp4 --headless --output_csv out.csv
  python main.py --model yolo11n.pt --video clip.mp4 --display
  python main.py --source 0 --imgsz 320 --headless           # más rápido en UNO Q
  python main.py --video clip.mp4 --save_annotated_video demo.mp4
"""

from __future__ import annotations

import argparse
import csv
import json
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
from traffic_aggregator import TrafficAggregator
from tracker import CentroidTracker
from visualization import draw_frame, draw_legend
from yolo_detector import YOLODetector


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EdgeAI Green Routes – vehicle detector node")
    p.add_argument("--model",   default=DEFAULT_MODEL,
                   help="Pesos YOLO (yolov8n.pt o yolo11n.pt)")
    p.add_argument("--source",  default=None, type=int,
                   help="Índice de webcam (ej. 0)")
    p.add_argument("--video",   default=None,
                   help="Ruta al fichero de vídeo")
    p.add_argument("--output_csv",  default=None,
                   help="Ruta del CSV de salida")
    p.add_argument("--output_json", default=None,
                   help="Ruta del JSONL de salida")
    p.add_argument("--display",  action="store_true",
                   help="Mostrar ventana OpenCV")
    p.add_argument("--headless", action="store_true",
                   help="Sin visualización (anula --display)")
    p.add_argument("--imgsz",   default=IMAGE_SIZE, type=int,
                   help="Resolución de inferencia YOLO (640, 416, 320)")
    p.add_argument("--conf",    default=CONFIDENCE_THRESHOLD, type=float,
                   help="Umbral de confianza YOLO")
    p.add_argument("--window_seconds", default=WINDOW_SECONDS, type=float,
                   help="Duración de la ventana de agregación (segundos)")
    p.add_argument("--use_yolo_tracker", action="store_true",
                   help="Usar tracker interno de Ultralytics (ByteTrack/BoT-SORT)")
    p.add_argument("--save_annotated_video", default=None,
                   help="Guardar vídeo anotado en esta ruta")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers de salida
# ---------------------------------------------------------------------------

def init_csv(path: str):
    """Abre el CSV y escribe la cabecera. Devuelve (file, writer)."""
    fh = open(path, "w", newline="", encoding="utf-8")
    fieldnames = list(WindowFeatures(0, 0, 0).to_dict().keys())
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    return fh, writer


def print_window_summary(wf: WindowFeatures) -> None:
    t_start = time.strftime("%H:%M:%S", time.localtime(wf.window_start))
    t_end   = time.strftime("%H:%M:%S", time.localtime(wf.window_end))
    print(
        f"\n{'='*55}\n"
        f"  VENTANA  {t_start} → {t_end}  ({wf.window_seconds:.0f}s)\n"
        f"  Vehículos únicos vistos:\n"
        f"    Motos        : {wf.count_motorcycle}\n"
        f"    Coches       : {wf.count_car}\n"
        f"    Buses/Camion : {wf.count_heavy}\n"
        f"  Media visibles/minuto:\n"
        f"    Motos        : {wf.mean_per_min_motorcycle:.2f}\n"
        f"    Coches       : {wf.mean_per_min_car:.2f}\n"
        f"    Buses/Camion : {wf.mean_per_min_heavy:.2f}\n"
        f"  FPS medio      : {wf.fps_mean:.1f}\n"
        f"{'='*55}"
    )


def print_live_status(snapshot: dict, fps: float, window_elapsed: float) -> None:
    print(
        f"[{time.strftime('%H:%M:%S')}]  "
        f"Visibles → motos:{snapshot['motorcycle']}  "
        f"coches:{snapshot['car']}  "
        f"pesados:{snapshot['heavy']}  | "
        f"FPS:{fps:.1f}  Win:{window_elapsed:.0f}s"
    )


def build_detection_by_track(
    active_tracks: Dict[int, Track],
    detections: List[Detection],
) -> Dict[int, Detection]:
    """Asocia track_id → Detection más reciente (para dibujar los bboxes)."""
    result: Dict[int, Detection] = {}
    for det in detections:
        best_id, best_dist = None, 9999.0
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
# Bucle principal
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    show_display = args.display and not args.headless

    # --- Fuente de vídeo ---------------------------------------------------
    source = args.source if args.source is not None else args.video
    if source is None:
        print("[ERROR] Especifica --source 0 (webcam) o --video <ruta>.")
        sys.exit(1)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] No se puede abrir la fuente: {source}")
        sys.exit(1)

    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[main] Fuente: {source}  {frame_width}×{frame_height}  {source_fps:.1f} fps")

    # --- Writer de vídeo anotado ------------------------------------------
    video_writer = None
    if args.save_annotated_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            args.save_annotated_video, fourcc, source_fps, (frame_width, frame_height)
        )
        print(f"[main] Guardando vídeo anotado en: {args.save_annotated_video}")

    # --- Detector YOLO -----------------------------------------------------
    detector = YOLODetector(model_path=args.model, imgsz=args.imgsz, conf=args.conf)

    # --- Tracker -----------------------------------------------------------
    tracker    = CentroidTracker()
    aggregator = TrafficAggregator(window_seconds=args.window_seconds)
    print(f"[main] Tracker: {'YOLO built-in' if args.use_yolo_tracker else 'Centroide (default)'}")
    print(f"[main] Ventana: {args.window_seconds}s")

    # --- Ficheros de salida ------------------------------------------------
    csv_fh = csv_writer = json_fh = None
    if args.output_csv:
        csv_fh, csv_writer = init_csv(args.output_csv)
        print(f"[main] CSV → {args.output_csv}")
    if args.output_json:
        json_fh = open(args.output_json, "a", encoding="utf-8")
        print(f"[main] JSONL → {args.output_json}")

    # --- Estado del bucle --------------------------------------------------
    frame_count     = 0
    fps             = 0.0
    last_print_time = time.time()

    print("[main] Iniciando detección. Pulsa 'q' para salir.\n")

    try:
        while True:
            t0 = time.time()

            ret, frame = cap.read()
            if not ret:
                print("[main] Fin del stream.")
                break

            frame_count += 1
            timestamp = time.time()

            # --- Detección -------------------------------------------------
            if args.use_yolo_tracker:
                detections, yolo_ids = detector.detect_with_tracker(frame)
            else:
                detections = detector.detect(frame)
                yolo_ids   = None

            # --- Tracking --------------------------------------------------
            active_tracks = tracker.update(detections, timestamp, yolo_ids)

            # --- Agregación ------------------------------------------------
            aggregator.update(active_tracks, timestamp, fps)

            # --- Cerrar ventana si toca ------------------------------------
            elapsed = aggregator.seconds_since_window_start(timestamp)
            if elapsed >= args.window_seconds:
                wf = aggregator.close_window(active_tracks, timestamp)
                print_window_summary(wf)

                if csv_writer:
                    csv_writer.writerow(wf.to_dict())
                    csv_fh.flush()
                if json_fh:
                    json_fh.write(json.dumps(wf.to_dict()) + "\n")
                    json_fh.flush()

            # --- FPS -------------------------------------------------------
            frame_time = time.time() - t0
            fps = 1.0 / frame_time if frame_time > 0 else 0.0

            # --- Print periódico -------------------------------------------
            if timestamp - last_print_time >= TERMINAL_PRINT_INTERVAL_SEC:
                snapshot = aggregator.get_live_snapshot(active_tracks)
                print_live_status(
                    snapshot, fps,
                    aggregator.seconds_since_window_start(timestamp),
                )
                last_print_time = timestamp

            # --- Visualización --------------------------------------------
            if show_display or video_writer:
                snapshot = aggregator.get_live_snapshot(active_tracks)
                det_by_track = build_detection_by_track(active_tracks, detections)
                annotated = draw_frame(
                    frame,
                    active_tracks,
                    det_by_track,
                    exposure_score=0.0,           # ya no usamos score
                    exposure_category="",
                    fps=fps,
                    window_elapsed_s=aggregator.seconds_since_window_start(timestamp),
                )
                draw_legend(annotated)

                if show_display:
                    cv2.imshow("EdgeAI Green Routes", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("[main] Usuario pulsó 'q'.")
                        break

                if video_writer:
                    video_writer.write(annotated)

    finally:
        cap.release()
        if video_writer:
            video_writer.release()
        if show_display:
            cv2.destroyAllWindows()
        if csv_fh:
            csv_fh.close()
        if json_fh:
            json_fh.close()

        print(f"\n[main] Fin. Frames procesados: {frame_count}")


if __name__ == "__main__":
    main()