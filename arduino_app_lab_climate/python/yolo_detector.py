"""
yolo_detector.py
----------------
Loads a YOLO nano model (YOLOv8n or YOLO11n) and runs per-frame inference.
Returns a list of Detection objects — NO tracking here, just detection.

Privacy note: We filter ONLY vehicle classes (car, motorcycle, bus, truck).
People, faces, and license plates are NOT extracted.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from config import (
    COCO_CLASS_IDS,
    COCO_ID_TO_LABEL,
    LABEL_TO_PROJECT_CLASS,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    IMAGE_SIZE,
    DEFAULT_MODEL,
    ENABLE_VAN_LIKE_HEURISTIC,
    VAN_LIKE_VEHICLE_WEIGHT,
    VEHICLE_WEIGHTS,
)
from data_structures import Detection


# Set of COCO class IDs we want — used as a fast filter set
_ALLOWED_CLASS_IDS = set(COCO_CLASS_IDS.values())


class YOLODetector:
    """
    Wraps Ultralytics YOLO for single-frame vehicle detection.

    Usage
    -----
    detector = YOLODetector(model_path="yolov8n.pt", imgsz=640)
    detections = detector.detect(frame)   # frame is an OpenCV BGR numpy array
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        imgsz: int = IMAGE_SIZE,
        conf: float = CONFIDENCE_THRESHOLD,
        iou: float = IOU_THRESHOLD,
        device: Optional[str] = None,   # None → auto (CPU on UNO Q)
    ) -> None:
        from ultralytics import YOLO  # import here so the module loads even without GPU

        print(f"[YOLODetector] Loading model: {model_path}  imgsz={imgsz}")
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf  = conf
        self.iou   = iou
        self.device = device or "cpu"

        print(f"[YOLODetector] Running on device: {self.device}")

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run YOLO inference on a single BGR frame.
        Returns only vehicle detections (car, motorcycle, bus, truck).
        NO person or face detection.
        """
        results = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            classes=list(_ALLOWED_CLASS_IDS),   # pre-filter at model level
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []

        if not results:
            return detections

        result = results[0]   # single image → single result

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes      = result.boxes.xyxy.cpu().numpy()    # (N, 4) x1 y1 x2 y2
        confs      = result.boxes.conf.cpu().numpy()    # (N,)
        class_ids  = result.boxes.cls.cpu().numpy().astype(int)  # (N,)

        for box, conf_val, cid in zip(boxes, confs, class_ids):
            if cid not in _ALLOWED_CLASS_IDS:
                continue   # safety double-filter

            label = COCO_ID_TO_LABEL.get(cid)
            if label is None:
                continue

            project_class = LABEL_TO_PROJECT_CLASS.get(label)
            if project_class is None:
                continue

            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # Optional van-like heuristic: if a "car" has van-like proportions,
            # we can later use a heavier weight. Detection still uses project_class
            # = medium_vehicle; the aggregator can check the bbox if needed.
            det = Detection(
                label=label,
                project_class=project_class,
                confidence=float(conf_val),
                bbox=(x1, y1, x2, y2),
                centroid=(cx, cy),
                class_id=cid,
            )
            detections.append(det)

        return detections

    # ------------------------------------------------------------------
    def detect_with_tracker(self, frame: np.ndarray):
        """
        Use Ultralytics' built-in ByteTrack/BoT-SORT tracking.
        Returns (raw_result, track_id_map) where track_id_map maps
        detection index → yolo_track_id (int or None).

        Only used when --use_yolo_tracker is active.
        """
        results = self.model.track(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            classes=list(_ALLOWED_CLASS_IDS),
            device=self.device,
            persist=True,        # keep track state between calls
            verbose=False,
        )

        detections: List[Detection] = []
        yolo_track_ids: List[Optional[int]] = []

        if not results:
            return detections, yolo_track_ids

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return detections, yolo_track_ids

        boxes     = result.boxes.xyxy.cpu().numpy()
        confs     = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        # YOLO track IDs may be None if tracking hasn't locked on yet
        if result.boxes.id is not None:
            track_ids = result.boxes.id.cpu().numpy().astype(int)
        else:
            track_ids = [None] * len(boxes)

        for box, conf_val, cid, tid in zip(boxes, confs, class_ids, track_ids):
            if cid not in _ALLOWED_CLASS_IDS:
                continue

            label = COCO_ID_TO_LABEL.get(cid)
            if label is None:
                continue

            project_class = LABEL_TO_PROJECT_CLASS.get(label)
            if project_class is None:
                continue

            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            det = Detection(
                label=label,
                project_class=project_class,
                confidence=float(conf_val),
                bbox=(x1, y1, x2, y2),
                centroid=(cx, cy),
                class_id=cid,
            )
            detections.append(det)
            yolo_track_ids.append(int(tid) if tid is not None else None)

        return detections, yolo_track_ids