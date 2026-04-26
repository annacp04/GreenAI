# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
# SPDX-License-Identifier: MPL-2.0

"""
main.py — EdgeAI Green Routes · Arduino App Lab
================================================
Integrates:
  · Arduino App Lab / Bridge      (temperature + humidity from MCU sketch)
  · ObjectDetection Brick         (vehicle detection via Edge Impulse YOLOv11)
  · Flask MJPEG stream            (/video_feed)
  · Flask dashboard               (/)
  · Centroid tracker              (CentroidTracker)
  · Traffic aggregator            (TrafficAggregator)
  · CSV logger                    (traffic_climate_log.csv)

Thread layout
-------------
  main thread    → App Lab bridge + App.run()
  camera_thread  → OpenCV capture + ObjectDetection brick + tracker + aggregator (daemon)
  flask_thread   → Flask HTTP server on 0.0.0.0:5000                             (daemon)

Shared state (all protected by locks)
--------------------------------------
  jpeg_lock   → latest_jpeg    (bytes: last annotated JPEG for /video_feed)
  live_lock   → latest_live    (dict:  current frame vehicle counts + fps)
  window_lock → last_window    (dict:  last closed window features)
  sensor_lock → latest_sensor  (dict:  last temperature/humidity + derived)
"""

import csv
import datetime
import math
import os
import threading
import time

import cv2
from flask import Flask, Response, jsonify, render_template_string, send_file

# Arduino App Lab imports
from arduino.app_bricks.dbstorage_tsstore import TimeSeriesStore
from arduino.app_bricks.object_detection import ObjectDetection
from arduino.app_utils import App, Bridge
from arduino.app_utils.image import draw_bounding_boxes
from arduino.app_bricks.web_ui import WebUI

# Our existing pipeline modules (unchanged)
from tracker import CentroidTracker
from traffic_aggregator import TrafficAggregator
from visualization import draw_frame, draw_legend

# ---------------------------------------------------------------------------
# Configuration — change these without touching anything else
# ---------------------------------------------------------------------------
CAMERA_SOURCE  = 0
CONFIDENCE     = 0.35
WINDOW_SECONDS = 60
CSV_PATH       = "traffic_climate_log.csv"
FLASK_HOST     = "0.0.0.0"
FLASK_PORT     = 5000

# Map Edge Impulse / COCO class names → internal categories
# Adjust keys if your EI project uses different label names
EI_LABEL_MAP = {
    "motorcycle": "motorcycle",
    "motorbike":  "motorcycle",
    "car":        "car",
    "vehicle":    "car",
    "truck":      "heavy",
    "bus":        "heavy",
}

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
latest_jpeg   = None
latest_live   = {"motorcycle": 0, "car": 0, "heavy": 0, "total": 0, "fps": 0.0}
last_window   = {}
latest_sensor = {
    "temperature_c": None, "humidity_percent": None,
    "dew_point_c": None, "heat_index_c": None, "absolute_humidity": None,
}

jpeg_lock   = threading.Lock()
live_lock   = threading.Lock()
window_lock = threading.Lock()
sensor_lock = threading.Lock()

# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
CSV_COLUMNS = [
    "timestamp", "window_start", "window_end", "window_seconds",
    "temperature_c", "humidity_percent", "dew_point_c",
    "heat_index_c", "absolute_humidity",
    "count_motorcycle", "count_car", "count_heavy",
    "mean_per_min_motorcycle", "mean_per_min_car", "mean_per_min_heavy",
    "fps_mean",
]

def _init_csv():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=CSV_COLUMNS).writeheader()
        print(f"[CSV] Created {CSV_PATH}")

def _write_csv_row(wf_dict, sensor):
    row = {
        "timestamp":               datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "window_start":            wf_dict.get("window_start", ""),
        "window_end":              wf_dict.get("window_end", ""),
        "window_seconds":          wf_dict.get("window_seconds", ""),
        "temperature_c":           sensor.get("temperature_c", ""),
        "humidity_percent":        sensor.get("humidity_percent", ""),
        "dew_point_c":             sensor.get("dew_point_c", ""),
        "heat_index_c":            sensor.get("heat_index_c", ""),
        "absolute_humidity":       sensor.get("absolute_humidity", ""),
        "count_motorcycle":        wf_dict.get("count_motorcycle", 0),
        "count_car":               wf_dict.get("count_car", 0),
        "count_heavy":             wf_dict.get("count_heavy", 0),
        "mean_per_min_motorcycle": wf_dict.get("mean_per_min_motorcycle", 0.0),
        "mean_per_min_car":        wf_dict.get("mean_per_min_car", 0.0),
        "mean_per_min_heavy":      wf_dict.get("mean_per_min_heavy", 0.0),
        "fps_mean":                wf_dict.get("fps_mean", 0.0),
    }
    try:
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=CSV_COLUMNS).writerow(row)
            fh.flush()
    except Exception as e:
        print(f"[CSV] Write error: {e}")

# ---------------------------------------------------------------------------
# Helper: match active tracks → detections (for bbox drawing)
# ---------------------------------------------------------------------------
def _build_detection_by_track(active_tracks, detections):
    result = {}
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
# Camera + ObjectDetection brick loop
# ---------------------------------------------------------------------------
def camera_loop():
    global latest_jpeg, latest_live

    print(f"[camera] Opening camera {CAMERA_SOURCE} ...")
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print(f"[camera] ERROR: cannot open camera {CAMERA_SOURCE}. Check USB connection.")
        return

    # ── ObjectDetection brick (replaces YOLODetector) ────────────────────
    print("[camera] Initialising ObjectDetection brick ...")
    try:
        detector = ObjectDetection(confidence=CONFIDENCE)
    except Exception as e:
        print(f"[camera] ERROR initialising ObjectDetection brick: {e}")
        cap.release()
        return

    tracker    = CentroidTracker()
    aggregator = TrafficAggregator(window_seconds=WINDOW_SECONDS)
    fps        = 0.0

    print("[camera] Detection loop started.")

    while True:
        t0 = time.time()

        ret, frame = cap.read()
        if not ret:
            print("[camera] Frame read failed — retrying in 0.1 s ...")
            time.sleep(0.1)
            continue

        timestamp = time.time()
        h, w = frame.shape[:2]

        # ── Encode frame to JPEG bytes for the brick ──────────────────────
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            continue
        frame_bytes = buf.tobytes()

        # ── Detection via ObjectDetection brick ───────────────────────────
        try:
            out = detector.detect(frame_bytes, image_type="jpg")
        except Exception as e:
            print(f"[camera] Detection error: {e}")
            out = {}

        # ── Convert brick output → Detection objects for the tracker ──────
        from tracker import Detection   # namedtuple/dataclass in your tracker module

        detections = []
        raw_list = (out or {}).get("detection", [])

        for obj in raw_list:
            label = EI_LABEL_MAP.get((obj.get("class_name") or "").lower())
            if label is None:
                continue                             # ignore irrelevant classes

            conf             = obj.get("confidence", 0.0)
            x1, y1, x2, y2  = obj.get("bounding_box_xyxy", [0, 0, 0, 0])

            # ── Coord safety: if brick returns 0..1 floats, rescale to px ──
            # The brick SHOULD return absolute pixel coords, but if the values
            # are all ≤ 1.0 we assume normalised and rescale defensively.
            if all(v <= 1.0 for v in (x1, y1, x2, y2)):
                x1, x2 = int(x1 * w), int(x2 * w)
                y1, y2 = int(y1 * h), int(y2 * h)
            else:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            detections.append(Detection(
                centroid=(cx, cy),
                bbox=(x1, y1, x2, y2),
                label=label,
                confidence=conf,
            ))

        # ── Tracking ──────────────────────────────────────────────────────
        try:
            active_tracks = tracker.update(detections, timestamp)
        except Exception as e:
            print(f"[camera] Tracker error: {e}")
            active_tracks = {}

        # ── Aggregation ───────────────────────────────────────────────────
        aggregator.update(active_tracks, timestamp, fps)

        if aggregator.seconds_since_window_start(timestamp) >= WINDOW_SECONDS:
            wf      = aggregator.close_window(active_tracks, timestamp)
            wf_dict = wf.to_dict()

            with sensor_lock:
                sensor_snap = dict(latest_sensor)
            with window_lock:
                last_window.clear()
                last_window.update(wf_dict)

            _write_csv_row(wf_dict, sensor_snap)

            print(
                f"[window] motos={wf.count_motorcycle}  coches={wf.count_car}  "
                f"pesados={wf.count_heavy}  "
                f"media/min → m:{wf.mean_per_min_motorcycle:.2f} "
                f"c:{wf.mean_per_min_car:.2f} p:{wf.mean_per_min_heavy:.2f}  "
                f"fps={wf.fps_mean:.1f}"
            )

        # ── Live snapshot ─────────────────────────────────────────────────
        snapshot = aggregator.get_live_snapshot(active_tracks)
        with live_lock:
            latest_live.update({**snapshot, "fps": round(fps, 1)})

        # ── Annotate frame ────────────────────────────────────────────────
        try:
            det_by_track = _build_detection_by_track(active_tracks, detections)
            annotated    = draw_frame(
                frame, active_tracks, det_by_track,
                exposure_score=0.0, exposure_category="",
                fps=fps,
                window_elapsed_s=aggregator.seconds_since_window_start(timestamp),
            )
            draw_legend(annotated)
        except Exception as e:
            print(f"[camera] Visualization error: {e}")
            annotated = frame

        # ── Encode final JPEG for /video_feed ─────────────────────────────
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with jpeg_lock:
                latest_jpeg = buf.tobytes()

        elapsed = time.time() - t0
        fps = 1.0 / elapsed if elapsed > 0 else 0.0

    cap.release()
    print("[camera] Loop ended.")

# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------
flask_app = Flask(__name__)

def _mjpeg_generator():
    while True:
        with jpeg_lock:
            frame_bytes = latest_jpeg
        if frame_bytes is None:
            time.sleep(0.05)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes + b"\r\n"
        )
        time.sleep(0.03)   # ~30 fps cap

@flask_app.route("/video_feed")
def video_feed():
    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

# --- Dashboard HTML (self-contained, no external CDN) ----------------------
_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EdgeAI Green Routes — Dashboard</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',sans-serif;background:#0f1117;color:#e0e0e0;min-height:100vh}
header{background:#1a1d27;padding:14px 24px;display:flex;align-items:center;gap:12px;border-bottom:1px solid #2a2d3a}
header h1{font-size:1.2rem;font-weight:600;color:#fff}
header span{font-size:.8rem;color:#7c8db5}
.layout{display:grid;grid-template-columns:1fr 340px;gap:16px;padding:16px;max-width:1400px;margin:0 auto}
.video-panel img{width:100%;border-radius:8px;border:1px solid #2a2d3a;display:block}
.sidebar{display:flex;flex-direction:column;gap:14px}
.card{background:#1a1d27;border:1px solid #2a2d3a;border-radius:8px;padding:14px 16px}
.card h2{font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;color:#7c8db5;margin-bottom:10px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.stat{background:#12141c;border-radius:6px;padding:10px 12px}
.stat .label{font-size:.7rem;color:#7c8db5;margin-bottom:2px}
.stat .value{font-size:1.4rem;font-weight:700;color:#fff}
.moto .value{color:#f0a500}.car .value{color:#3ecf8e}.heavy .value{color:#e05252}
.total .value{color:#60a5fa}.fps .value{font-size:1rem;color:#a78bfa}
.temp .value{color:#fb923c}.hum .value{color:#38bdf8}
.dew .value,.ah .value{font-size:1rem;color:#34d399}
.hi .value{font-size:1rem;color:#f87171}
.span2{grid-column:span 2}
.wtable{width:100%;border-collapse:collapse;font-size:.82rem}
.wtable th{text-align:left;color:#7c8db5;padding:3px 6px;font-weight:500}
.wtable td{padding:3px 6px}
.wtable tr:nth-child(even) td{background:#12141c;border-radius:4px}
.dl-btn{display:block;width:100%;padding:10px;background:#2563eb;color:#fff;border:none;
        border-radius:6px;font-size:.9rem;font-weight:600;cursor:pointer;text-align:center;
        text-decoration:none;transition:background .2s}
.dl-btn:hover{background:#1d4ed8}
.null{color:#4b5563;font-style:italic}
</style>
</head>
<body>
<header>
  <h1>🌿 EdgeAI Green Routes</h1>
  <span>Live traffic + climate · Arduino UNO Q · Edge Impulse YOLOv11</span>
</header>
<div class="layout">
  <div class="video-panel">
    <img src="/video_feed" alt="Live camera">
  </div>
  <div class="sidebar">

    <div class="card">
      <h2>Live vehicles (current frame)</h2>
      <div class="grid2">
        <div class="stat moto"><div class="label">Motorcycles</div><div class="value" id="lm">—</div></div>
        <div class="stat car"><div class="label">Cars</div><div class="value" id="lc">—</div></div>
        <div class="stat heavy"><div class="label">Bus / Truck</div><div class="value" id="lh">—</div></div>
        <div class="stat total"><div class="label">Total</div><div class="value" id="lt">—</div></div>
        <div class="stat fps span2"><div class="label">FPS</div><div class="value" id="lf">—</div></div>
      </div>
    </div>

    <div class="card">
      <h2>Climate (last sensor reading)</h2>
      <div class="grid2">
        <div class="stat temp"><div class="label">Temperature</div><div class="value" id="st">—</div></div>
        <div class="stat hum"><div class="label">Humidity</div><div class="value" id="sh">—</div></div>
        <div class="stat dew"><div class="label">Dew Point</div><div class="value" id="sd">—</div></div>
        <div class="stat hi"><div class="label">Heat Index</div><div class="value" id="si">—</div></div>
        <div class="stat ah span2"><div class="label">Absolute Humidity</div><div class="value" id="sa">—</div></div>
      </div>
    </div>

    <div class="card">
      <h2>Last window summary</h2>
      <table class="wtable">
        <tr><th>Metric</th><th>Motos</th><th>Cars</th><th>Heavy</th></tr>
        <tr><td>Count (unique)</td><td id="wcm">—</td><td id="wcc">—</td><td id="wch">—</td></tr>
        <tr><td>Mean / min</td><td id="wmm">—</td><td id="wmc">—</td><td id="wmh">—</td></tr>
        <tr><td colspan="2" style="color:#7c8db5">Window FPS</td><td colspan="2" id="wfps">—</td></tr>
      </table>
    </div>

    <div class="card">
      <h2>Data export</h2>
      <a class="dl-btn" href="/download/csv">⬇ Download CSV</a>
    </div>

  </div>
</div>
<script>
function fmt(v,d,u){
  if(v===null||v===undefined)return '<span class="null">—</span>';
  return(typeof v==="number"?v.toFixed(d):v)+(u||"");
}
async function refreshLive(){
  try{
    const d=await fetch("/api/live").then(r=>r.json());
    document.getElementById("lm").innerHTML=fmt(d.motorcycle,0);
    document.getElementById("lc").innerHTML=fmt(d.car,0);
    document.getElementById("lh").innerHTML=fmt(d.heavy,0);
    document.getElementById("lt").innerHTML=fmt(d.total,0);
    document.getElementById("lf").innerHTML=fmt(d.fps,1," fps");
    const s=d.sensor||{};
    document.getElementById("st").innerHTML=fmt(s.temperature_c,1," °C");
    document.getElementById("sh").innerHTML=fmt(s.humidity_percent,1," %");
    document.getElementById("sd").innerHTML=fmt(s.dew_point_c,1," °C");
    document.getElementById("si").innerHTML=fmt(s.heat_index_c,1," °C");
    document.getElementById("sa").innerHTML=fmt(s.absolute_humidity,2," g/m³");
  }catch(e){}
}
async function refreshWindow(){
  try{
    const w=await fetch("/api/last_window").then(r=>r.json());
    if(!w||!Object.keys(w).length)return;
    document.getElementById("wcm").innerHTML=fmt(w.count_motorcycle,0);
    document.getElementById("wcc").innerHTML=fmt(w.count_car,0);
    document.getElementById("wch").innerHTML=fmt(w.count_heavy,0);
    document.getElementById("wmm").innerHTML=fmt(w.mean_per_min_motorcycle,2);
    document.getElementById("wmc").innerHTML=fmt(w.mean_per_min_car,2);
    document.getElementById("wmh").innerHTML=fmt(w.mean_per_min_heavy,2);
    document.getElementById("wfps").innerHTML=fmt(w.fps_mean,1," fps");
  }catch(e){}
}
setInterval(refreshLive,1000);
setInterval(refreshWindow,5000);
refreshLive();refreshWindow();
</script>
</body>
</html>"""

@flask_app.route("/")
def dashboard():
    return render_template_string(_DASHBOARD_HTML)

@flask_app.route("/api/live")
def api_live():
    with live_lock:
        live = dict(latest_live)
    with sensor_lock:
        live["sensor"] = dict(latest_sensor)
    return jsonify(live)

@flask_app.route("/api/last_window")
def api_last_window():
    with window_lock:
        return jsonify(dict(last_window))

@flask_app.route("/download/csv")
def download_csv():
    if not os.path.exists(CSV_PATH):
        return "CSV not found yet — wait for the first window to close.", 404
    return send_file(
        os.path.abspath(CSV_PATH),
        as_attachment=True,
        download_name="traffic_climate_log.csv",
        mimetype="text/csv",
    )

def start_flask():
    print(f"[flask] Starting on http://{FLASK_HOST}:{FLASK_PORT}")
    flask_app.run(host=FLASK_HOST, port=FLASK_PORT,
                  threaded=True, use_reloader=False, debug=False)

# ---------------------------------------------------------------------------
# Arduino App Lab — TimeSeriesStore + WebUI
# ---------------------------------------------------------------------------
db = TimeSeriesStore()

def _on_get_samples(resource: str, start: str, aggr_window: str):
    samples = db.read_samples(
        measure=resource, start_from=start,
        aggr_window=aggr_window, aggr_func="mean", limit=100,
    )
    return [{"ts": s[1], "value": s[2]} for s in samples]

ui = WebUI()
ui.expose_api("GET", "/get_samples/{resource}/{start}/{aggr_window}", _on_get_samples)

# ---------------------------------------------------------------------------
# Bridge callback — temperature + humidity from MCU sketch (unchanged)
# ---------------------------------------------------------------------------
def record_sensor_samples(celsius: float, humidity: float) -> None:
    """
    Called by the MCU sketch via Bridge.notify.
    Computes dew point, heat index, absolute humidity.
    Stores everything in TimeSeriesStore, pushes to WebUI,
    and updates latest_sensor for the camera loop + dashboard.
    """
    if celsius is None or humidity is None:
        print(f"[sensor] Invalid sample: celsius={celsius}, humidity={humidity}")
        return

    T  = float(celsius)
    RH = float(humidity)
    ts = int(datetime.datetime.now().timestamp() * 1000)

    db.write_sample("temperature", T,  ts)
    db.write_sample("humidity",    RH, ts)
    ui.send_message("temperature", {"value": T,  "ts": ts})
    ui.send_message("humidity",    {"value": RH, "ts": ts})

    # Dew point (Magnus formula)
    a, b      = 17.27, 237.7
    dew_point = None
    if RH > 0.0:
        rh_frac   = max(min(RH, 100.0), 1e-6)
        gamma     = (a * T) / (b + T) + math.log(rh_frac / 100.0)
        dew_point = (b * gamma) / (a - gamma)

    # Heat Index (Rothfusz regression)
    T_f  = T * 9.0 / 5.0 + 32.0
    R    = max(min(RH, 100.0), 0.0)
    HI_f = (
        -42.379
        + 2.04901523  * T_f
        + 10.14333127 * R
        - 0.22475541  * T_f * R
        - 0.00683783  * T_f * T_f
        - 0.05481717  * R   * R
        + 0.00122874  * T_f * T_f * R
        + 0.00085282  * T_f * R   * R
        - 0.00000199  * T_f * T_f * R * R
    )
    heat_index = (HI_f - 32.0) * 5.0 / 9.0

    # Absolute humidity (g/m³)
    absolute_humidity = None
    if RH >= 0.0:
        es                = 6.112 * math.exp((17.67 * T) / (T + 243.5))
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

    with sensor_lock:
        latest_sensor["temperature_c"]     = round(T, 2)
        latest_sensor["humidity_percent"]  = round(RH, 2)
        latest_sensor["dew_point_c"]       = round(dew_point,        2) if dew_point        is not None else None
        latest_sensor["heat_index_c"]      = round(heat_index,       2) if heat_index       is not None else None
        latest_sensor["absolute_humidity"] = round(absolute_humidity, 3) if absolute_humidity is not None else None

    print(
        f"[sensor] T={T:.1f}°C  RH={RH:.1f}%  "
        f"dew={dew_point:.1f}°C  HI={heat_index:.1f}°C  "
        f"AH={absolute_humidity:.2f} g/m³"
    )

# ---------------------------------------------------------------------------
# Start-up
# ---------------------------------------------------------------------------
_init_csv()

threading.Thread(target=camera_loop, name="camera", daemon=True).start()
threading.Thread(target=start_flask,  name="flask",  daemon=True).start()

print("[app] Registering Bridge callback ...")
Bridge.provide("record_sensor_samples", record_sensor_samples)

print("[app] Starting App Lab runtime ...")
App.run()