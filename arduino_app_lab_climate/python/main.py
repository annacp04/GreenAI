# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import datetime
import math
import os
import csv
import time
import io
import base64
from PIL.Image import Image
from arduino.app_utils import App, Bridge
from arduino.app_bricks.web_ui import WebUI
from arduino.app_bricks.dbstorage_tsstore import TimeSeriesStore
from arduino.app_bricks.camera_code_detection import CameraCodeDetection

# Initialize core services first
db = TimeSeriesStore()

# ================= DATA LOGGING SETUP =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_file_path = os.path.join(BASE_DIR, "data_log.csv")

if not os.path.exists(csv_file_path):
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Temperature (C)", "Humidity (%)", "Light Level"])

def log_to_csv(celsius, humidity, light):
    try:
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), celsius, humidity, light])
    except Exception as e:
        print("CSV Error:", e)

# Throttle frames to 5 FPS to avoid WebSocket flooding
last_frame_time = 0

def on_frame(frame: Image):
    global last_frame_time
    now = time.time()
    if now - last_frame_time < 0.2:
        return
    last_frame_time = now

    """Callback function that processes each frame from the camera."""
    try:
        # Resize for bandwidth optimization to prevent WebSocket crash
        frame.thumbnail((640, 480))
        buffer = io.BytesIO()
        frame.save(buffer, format="JPEG", quality=60)
        b64_frame = base64.b64encode(buffer.getvalue()).decode("utf-8")

        ui.send_message('camera_frame', {
            "image": b64_frame,
            "image_type": "image/jpeg"
        })
    except Exception as e:
        pass # Silently handle frame errors to avoid log spam

def on_code_detected(frame: Image, detection):
    pass

def on_error(e: Exception):
    print(f"Camera error: {e}")

detector = CameraCodeDetection()
detector.on_detect(on_code_detected)
detector.on_frame(on_frame)
detector.on_error(on_error)

# Initialize WebUI after Camera detector
ui = WebUI()

# ================= API ENDPOINTS =================
def on_get_samples(resource: str, start: str, aggr_window: str):
    try:
        samples = db.read_samples(measure=resource, start_from=start, aggr_window=aggr_window, aggr_func="mean", limit=100)
        return [{"ts": s[1], "value": s[2]} for s in samples]
    except Exception:
        return []

def on_download_csv():
    if os.path.exists(csv_file_path):
        with open(csv_file_path, 'r') as f:
            return f.read()
    return "No data yet"

ui.expose_api("GET", "/get_samples/{resource}/{start}/{aggr_window}", on_get_samples)
ui.expose_api("GET", "/download_csv", on_download_csv)

# ================= SENSOR LOGIC =================
def record_sensor_samples(celsius: float, humidity: float, light: float = 0.0):
    if celsius is None or humidity is None:
        return

    log_to_csv(celsius, humidity, light)
    ts = int(datetime.datetime.now().timestamp() * 1000)

    # Helper to write to DB with safety check for early notifications
    def safe_write(measure, val, timestamp):
        try:
            db.write_sample(measure, float(val), timestamp)
        except Exception as e:
            # If InfluxDB isn't ready yet (write_api missing), we just skip this sample
            # and log to CSV as backup (which we already did above)
            if "write_api" in str(e):
                pass 
            else:
                print(f"DB Write Error ({measure}): {e}")

    safe_write("temperature", celsius, ts)
    safe_write("humidity", humidity, ts)
    safe_write("light", light, ts)

    ui.send_message('temperature', {"value": float(celsius), "ts": ts})
    ui.send_message('humidity', {"value": float(humidity), "ts": ts})
    ui.send_message('light', {"value": float(light), "ts": ts})

    T, RH = float(celsius), float(humidity)
    a, b = 17.27, 237.7
    
    # Derived calculations
    if RH > 0.0:
        rh_frac = max(min(RH, 100.0), 1e-6)
        gamma = (a * T) / (b + T) + math.log(rh_frac / 100.0)
        dew_point = (b * gamma) / (a - gamma)
        safe_write("dew_point", dew_point, ts)
        ui.send_message('dew_point', {"value": float(dew_point), "ts": ts})

    T_f = T * 9.0 / 5.0 + 32.0
    R = max(min(RH, 100.0), 0.0)
    HI_f = (-42.379 + 2.04901523 * T_f + 10.14333127 * R - 0.22475541 * T_f * R
            - 0.00683783 * T_f * T_f - 0.05481717 * R * R
            + 0.00122874 * T_f * T_f * R + 0.00085282 * T_f * R * R
            - 0.00000199 * T_f * T_f * R * R)
    heat_index = (HI_f - 32.0) * 5.0 / 9.0
    safe_write("heat_index", heat_index, ts)
    ui.send_message('heat_index', {"value": float(heat_index), "ts": ts})

    if RH >= 0.0:
        es = 6.112 * math.exp((17.67 * T) / (T + 243.5))
        abs_hum = es * (R / 100.0) * 2.1674 / (273.15 + T)
        safe_write("absolute_humidity", abs_hum, ts)
        ui.send_message('absolute_humidity', {"value": float(abs_hum), "ts": ts})

Bridge.provide("record_sensor_samples", record_sensor_samples)

print("Starting GreenAI Monitor...")
App.run()
