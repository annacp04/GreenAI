# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
# SPDX-License-Identifier: MPL-2.0

import datetime
import math
import os
import csv
import time
import cv2
import json
import threading
from flask import Flask, Response, send_file, jsonify, render_template_string

from arduino.app_bricks.dbstorage_tsstore import TimeSeriesStore
from arduino.app_bricks.web_ui import WebUI
from arduino.app_utils import App, Bridge

from yolo_detector import YOLODetector
from tracker import CentroidTracker
from traffic_aggregator import TrafficAggregator
from visualization import draw_frame, draw_legend
from data_structures import WindowFeatures


# ================= CONFIG =================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "yolo26n.pt")
CSV_PATH = os.path.join(BASE_DIR, "traffic_climate_log.csv")

CAMERA_SOURCE = 0
IMAGE_SIZE = 320
CONFIDENCE = 0.35
WINDOW_SECONDS = 60

# Shared live state
state_lock = threading.Lock()
latest_jpeg = None

latest_live = {
    "motorcycle": 0,
    "car": 0,
    "heavy": 0,
    "total": 0,
    "fps": 0.0,
    "window_elapsed": 0.0,
    "temperature": None,
    "humidity": None,
}

latest_sensor = {
    "temperature": None,
    "humidity": None,
    "dew_point": None,
    "heat_index": None,
    "absolute_humidity": None,
    "timestamp": None,
}

last_window = {}

stop_event = threading.Event()


# ================= CSV SETUP =================

CSV_FIELDS = [
    "timestamp",
    "window_start",
    "window_end",
    "window_seconds",

    "temperature_c",
    "humidity_percent",
    "dew_point_c",
    "heat_index_c",
    "absolute_humidity",

    "count_motorcycle",
    "count_car",
    "count_heavy",

    "mean_per_min_motorcycle",
    "mean_per_min_car",
    "mean_per_min_heavy",

    "fps_mean",
]

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()


def append_window_to_csv(wf: WindowFeatures):
    with state_lock:
        sensor = dict(latest_sensor)

    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "window_start": wf.window_start,
        "window_end": wf.window_end,
        "window_seconds": wf.window_seconds,

        "temperature_c": sensor.get("temperature"),
        "humidity_percent": sensor.get("humidity"),
        "dew_point_c": sensor.get("dew_point"),
        "heat_index_c": sensor.get("heat_index"),
        "absolute_humidity": sensor.get("absolute_humidity"),

        "count_motorcycle": wf.count_motorcycle,
        "count_car": wf.count_car,
        "count_heavy": wf.count_heavy,

        "mean_per_min_motorcycle": round(wf.mean_per_min_motorcycle, 2),
        "mean_per_min_car": round(wf.mean_per_min_car, 2),
        "mean_per_min_heavy": round(wf.mean_per_min_heavy, 2),

        "fps_mean": round(wf.fps_mean, 2),
    }

    with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writerow(row)


# ================= FLASK DASHBOARD =================

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>AirPace BCN - Live Dashboard</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      background: #111;
      color: #f5f5f5;
      margin: 24px;
    }
    .grid {
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 24px;
    }
    img {
      width: 100%;
      border: 2px solid #333;
      border-radius: 12px;
      background: #000;
    }
    .card {
      background: #1c1c1c;
      padding: 18px;
      border-radius: 12px;
      margin-bottom: 16px;
    }
    .row {
      display: flex;
      justify-content: space-between;
      border-bottom: 1px solid #333;
      padding: 8px 0;
    }
    .num {
      font-weight: bold;
      font-size: 26px;
    }
    a.button {
      display: inline-block;
      margin-top: 14px;
      padding: 12px 16px;
      background: #2ecc71;
      color: #111;
      border-radius: 8px;
      text-decoration: none;
      font-weight: bold;
    }
  </style>
</head>
<body>
  <h1>AirPace BCN - Live Vehicle Counter</h1>

  <div class="grid">
    <div>
      <img src="/video_feed" />
    </div>

    <div>
      <div class="card">
        <h2>Current visible vehicles</h2>
        <div class="row"><span>Motorcycles</span><span id="motorcycle" class="num">0</span></div>
        <div class="row"><span>Cars</span><span id="car" class="num">0</span></div>
        <div class="row"><span>Heavy</span><span id="heavy" class="num">0</span></div>
        <div class="row"><span>Total</span><span id="total" class="num">0</span></div>
        <div class="row"><span>FPS</span><span id="fps">0</span></div>
        <div class="row"><span>Window elapsed</span><span id="window_elapsed">0s</span></div>
      </div>

      <div class="card">
        <h2>Climate</h2>
        <div class="row"><span>Temperature</span><span id="temperature">-</span></div>
        <div class="row"><span>Humidity</span><span id="humidity">-</span></div>
      </div>

      <div class="card">
        <h2>Last minute summary</h2>
        <div class="row"><span>Motorcycles</span><span id="last_motorcycle">-</span></div>
        <div class="row"><span>Cars</span><span id="last_car">-</span></div>
        <div class="row"><span>Heavy</span><span id="last_heavy">-</span></div>
        <div class="row"><span>FPS mean</span><span id="last_fps">-</span></div>

        <a class="button" href="/download/csv">Download CSV</a>
      </div>
    </div>
  </div>

  <script>
    async function refresh() {
      const live = await fetch('/api/live').then(r => r.json());

      document.getElementById('motorcycle').textContent = live.motorcycle;
      document.getElementById('car').textContent = live.car;
      document.getElementById('heavy').textContent = live.heavy;
      document.getElementById('total').textContent = live.total;
      document.getElementById('fps').textContent = live.fps.toFixed(1);
      document.getElementById('window_elapsed').textContent = live.window_elapsed.toFixed(0) + "s";

      document.getElementById('temperature').textContent =
        live.temperature === null ? "-" : live.temperature.toFixed(1) + " °C";
      document.getElementById('humidity').textContent =
        live.humidity === null ? "-" : live.humidity.toFixed(1) + " %";

      const win = await fetch('/api/last_window').then(r => r.json());
      if (win && Object.keys(win).length > 0) {
        document.getElementById('last_motorcycle').textContent = win.count_motorcycle;
        document.getElementById('last_car').textContent = win.count_car;
        document.getElementById('last_heavy').textContent = win.count_heavy;
        document.getElementById('last_fps').textContent = win.fps_mean;
      }
    }

    setInterval(refresh, 1000);
    refresh();
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            with state_lock:
                frame = latest_jpeg

            if frame is None:
                time.sleep(0.05)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
            time.sleep(0.03)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/live")
def api_live():
    with state_lock:
        return jsonify(latest_live)


@app.route("/api/last_window")
def api_last_window():
    with state_lock:
        return jsonify(last_window)


@app.route("/download/csv")
def download_csv():
    return send_file(CSV_PATH, as_attachment=True, download_name="traffic_climate_log.csv")


def start_flask():
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)


# ================= CAMERA + YOLO LOOP =================

def build_detection_by_track(active_tracks, detections):
    result = {}

    for det in detections:
        best_id = None
        best_dist = 9999.0

        for tid, tr in active_tracks.items():
            if not tr.positions:
                continue

            _, tcx, tcy = tr.positions[-1]
            dist = ((det.centroid[0] - tcx) ** 2 + (det.centroid[1] - tcy) ** 2) ** 0.5

            if dist < best_dist:
                best_dist = dist
                best_id = tid

        if best_id is not None and best_dist < 80:
            result[best_id] = det

    return result


def camera_yolo_loop():
    global latest_jpeg, latest_live, last_window

    print("[YOLO] Opening camera...")
    cap = cv2.VideoCapture(CAMERA_SOURCE)

    if not cap.isOpened():
        print("[YOLO ERROR] Could not open camera.")
        return

    print("[YOLO] Loading model:", MODEL_PATH)
    detector = YOLODetector(model_path=MODEL_PATH, imgsz=IMAGE_SIZE, conf=CONFIDENCE)

    tracker = CentroidTracker()
    aggregator = TrafficAggregator(window_seconds=WINDOW_SECONDS)

    fps = 0.0

    print("[YOLO] Detection loop started.")

    while not stop_event.is_set():
        t0 = time.time()

        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        timestamp = time.time()

        detections = detector.detect(frame)
        active_tracks = tracker.update(detections, timestamp, yolo_track_ids=None)

        aggregator.update(active_tracks, timestamp, fps)

        snapshot = aggregator.get_live_snapshot(active_tracks)
        window_elapsed = aggregator.seconds_since_window_start(timestamp)

        # Close one-minute window
        if window_elapsed >= WINDOW_SECONDS:
            wf = aggregator.close_window(active_tracks, timestamp)
            append_window_to_csv(wf)

            with state_lock:
                last_window = wf.to_dict()

            print(
                f"[WINDOW] moto={wf.count_motorcycle} "
                f"car={wf.count_car} heavy={wf.count_heavy} "
                f"fps={wf.fps_mean:.1f}"
            )

            # Send to App Lab WebUI too
            ts = int(datetime.datetime.now().timestamp() * 1000)
            ui.send_message("count_motorcycle", {"value": wf.count_motorcycle, "ts": ts})
            ui.send_message("count_car", {"value": wf.count_car, "ts": ts})
            ui.send_message("count_heavy", {"value": wf.count_heavy, "ts": ts})

        # Annotate frame
        det_by_track = build_detection_by_track(active_tracks, detections)
        annotated = draw_frame(
            frame,
            active_tracks,
            det_by_track,
            exposure_score=0.0,
            exposure_category="",
            fps=fps,
            window_elapsed_s=window_elapsed,
        )
        draw_legend(annotated)

        ok_jpg, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok_jpg:
            with state_lock:
                latest_jpeg = buffer.tobytes()

                latest_live = {
                    "motorcycle": snapshot["motorcycle"],
                    "car": snapshot["car"],
                    "heavy": snapshot["heavy"],
                    "total": snapshot["total"],
                    "fps": fps,
                    "window_elapsed": window_elapsed,
                    "temperature": latest_sensor["temperature"],
                    "humidity": latest_sensor["humidity"],
                }

        dt = time.time() - t0
        fps = 1.0 / dt if dt > 0 else 0.0

    cap.release()


# ================= SENSOR STORAGE =================

db = TimeSeriesStore()

def on_get_samples(resource: str, start: str, aggr_window: str):
    samples = db.read_samples(
        measure=resource,
        start_from=start,
        aggr_window=aggr_window,
        aggr_func="mean",
        limit=100,
    )
    return [{"ts": s[1], "value": s[2]} for s in samples]


ui = WebUI()
ui.expose_api("GET", "/get_samples/{resource}/{start}/{aggr_window}", on_get_samples)


def record_sensor_samples(celsius: float, humidity: float):
    if celsius is None or humidity is None:
        print("Received invalid sensor samples: celsius=%s, humidity=%s" % (celsius, humidity))
        return

    ts = int(datetime.datetime.now().timestamp() * 1000)

    T = float(celsius)
    RH = float(humidity)

    db.write_sample("temperature", T, ts)
    db.write_sample("humidity", RH, ts)

    ui.send_message("temperature", {"value": T, "ts": ts})
    ui.send_message("humidity", {"value": RH, "ts": ts})

    a = 17.27
    b = 237.7
    dew_point = None
    if RH > 0.0:
        rh_frac = max(min(RH, 100.0), 1e-6)
        gamma = (a * T) / (b + T) + math.log(rh_frac / 100.0)
        dew_point = (b * gamma) / (a - gamma)

    T_f = T * 9.0 / 5.0 + 32.0
    R = max(min(RH, 100.0), 0.0)
    HI_f = (
        -42.379
        + 2.04901523 * T_f
        + 10.14333127 * R
        - 0.22475541 * T_f * R
        - 0.00683783 * T_f * T_f
        - 0.05481717 * R * R
        + 0.00122874 * T_f * T_f * R
        + 0.00085282 * T_f * R * R
        - 0.00000199 * T_f * T_f * R * R
    )
    heat_index = (HI_f - 32.0) * 5.0 / 9.0

    absolute_humidity = None
    if RH >= 0.0:
        es = 6.112 * math.exp((17.67 * T) / (T + 243.5))
        absolute_humidity = es * (R / 100.0) * 2.1674 / (273.15 + T)

    if dew_point is not None:
        db.write_sample("dew_point", float(dew_point), ts)
        ui.send_message("dew_point", {"value": float(dew_point), "ts": ts})

    if heat_index is not None:
        db.write_sample("heat_index", float(heat_index), ts)
        ui.send_message("heat_index", {"value": float(heat_index), "ts": ts})

    if absolute_humidity is not None:
        db.write_sample("absolute_humidity", float(absolute_humidity), ts)
        ui.send_message("absolute_humidity", {"value": float(absolute_humidity), "ts": ts})

    with state_lock:
        latest_sensor["temperature"] = T
        latest_sensor["humidity"] = RH
        latest_sensor["dew_point"] = dew_point
        latest_sensor["heat_index"] = heat_index
        latest_sensor["absolute_humidity"] = absolute_humidity
        latest_sensor["timestamp"] = ts


# ================= START APP =================

print("Starting Flask dashboard...")
threading.Thread(target=start_flask, daemon=True).start()

print("Starting YOLO camera loop...")
threading.Thread(target=camera_yolo_loop, daemon=True).start()

print("Registering 'record_sensor_samples' callback.")
Bridge.provide("record_sensor_samples", record_sensor_samples)

print("Starting App...")
App.run()