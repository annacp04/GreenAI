"""
config.py
---------
Central configuration for the EdgeAI Green Routes detector.
All thresholds, weights, and class mappings live here so they can
be tuned without touching the logic modules.
"""

# ---------------------------------------------------------------------------
# COCO class IDs we care about (all others are ignored)
# ---------------------------------------------------------------------------
COCO_CLASS_IDS = {
    "car":        2,
    "motorcycle": 3,
    "bus":        5,
    "truck":      7,
}

# Reverse map: id -> coco label string
COCO_ID_TO_LABEL = {v: k for k, v in COCO_CLASS_IDS.items()}

# ---------------------------------------------------------------------------
# Project class mapping: COCO label -> internal project class
# ---------------------------------------------------------------------------
LABEL_TO_PROJECT_CLASS = {
    "motorcycle": "light_vehicle",
    "car":        "medium_vehicle",
    "bus":        "heavy_vehicle",
    "truck":      "heavy_vehicle",
}

# ---------------------------------------------------------------------------
# Exposure weights per project class
# Normalised relative to a medium car (174 g CO2e/km ≈ 1.0)
#   motorcycle ~114 g  → 0.65
#   car        ~174 g  → 1.0
#   van        ~213 g  → 1.2  (optional heuristic, disabled by default)
#   bus/truck  ~977 g  → 5.5
# ---------------------------------------------------------------------------
VEHICLE_WEIGHTS = {
    "light_vehicle":  0.65,
    "medium_vehicle": 1.0,
    "heavy_vehicle":  5.5,
}

# Optional van-like weight (large bounding box / high aspect ratio cars).
# Disabled by default — unreliable without depth info.
VAN_LIKE_VEHICLE_WEIGHT  = 1.2
ENABLE_VAN_LIKE_HEURISTIC = False   # set True to activate

# ---------------------------------------------------------------------------
# Movement state weights
# Only MOVING vehicles contribute to the exposure score.
# ---------------------------------------------------------------------------
STATE_WEIGHTS = {
    "MOVING":     1.0,
    "STATIONARY": 0.0,
    "UNKNOWN":    0.0,
    "LOST":       0.0,
}

# ---------------------------------------------------------------------------
# Detection / inference thresholds
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD  = 0.35   # YOLO confidence filter
IOU_THRESHOLD         = 0.50   # NMS IoU threshold
IMAGE_SIZE            = 640    # Inference resolution (try 416 or 320 for speed)

# ---------------------------------------------------------------------------
# Tracking thresholds
# ---------------------------------------------------------------------------
MAX_MATCH_DISTANCE_PX  = 50    # Max centroid distance (px) to match a detection to a track
MOVE_THRESHOLD_PX      = 15    # Total displacement (px) to confirm movement
SPEED_THRESHOLD_PX_PER_SEC = 10.0
STATIONARY_SECONDS = 2.0     # Seconds below speed threshold → STATIONARY
MAX_MISSING_SECONDS    = 2.0   # Seconds without detection → LOST

# Number of recent positions used to compute instantaneous speed
SPEED_WINDOW_POSITIONS = 5

# ---------------------------------------------------------------------------
# Aggregation window
# ---------------------------------------------------------------------------
WINDOW_SECONDS = 60

# ---------------------------------------------------------------------------
# Exposure score formula constants (heuristic, hackathon MVP)
# ---------------------------------------------------------------------------
# flow_score    = min(FLOW_MAX,    FLOW_MAX    * weighted_unique_moving / FLOW_REF)
# density_score = min(DENSITY_MAX, DENSITY_MAX * (weighted_moving_seconds / WINDOW_SECONDS) / DENSITY_REF)
# peak_score    = min(PEAK_MAX,    PEAK_MAX    * max_weighted_moving_visible / PEAK_REF)

FLOW_MAX    = 35.0
FLOW_REF    = 35.0    # weighted unique vehicles that saturates flow_score

DENSITY_MAX = 45.0
DENSITY_REF = 10.0    # avg weighted vehicles/s that saturates density_score

PEAK_MAX    = 20.0
PEAK_REF    = 15.0    # peak weighted vehicles visible that saturates peak_score

# Exposure category thresholds
EXPOSURE_LOW_MAX    = 30.0
EXPOSURE_MEDIUM_MAX = 65.0
# score >= EXPOSURE_MEDIUM_MAX → HIGH

# ---------------------------------------------------------------------------
# Default model
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "yolov8n.pt"   # switch to "yolo11n.pt" via --model CLI arg

# ---------------------------------------------------------------------------
# Logging / output
# ---------------------------------------------------------------------------
TERMINAL_PRINT_INTERVAL_SEC = 5   # How often to print a live summary