# src/ml/inference.py
from ultralytics import YOLO
import numpy as np
import cv2
import io
from typing import Tuple, List, Dict
from PIL import Image, ImageDraw, ImageFont
import time
from src.config.project_config import settings
import os

WEIGHTS_PATH = os.path.join("src", "ml", "weights", "best.pt")
model = YOLO(WEIGHTS_PATH)

def run_inference_on_image_bytes(image_bytes: bytes) -> Tuple[List[Dict], Dict]:
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    start = time.time()
    results = model(img)
    latency_ms = int((time.time() - start) * 1000)

    detected = []
    for r in results:
        # r.boxes may be empty
        if not hasattr(r, "boxes") or r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = model.names.get(cls_id, str(cls_id))
            detected.append({
                "class_id": f"P{cls_id:02d}",
                "class_name": cls_name,
                "frame_first_seen": 0,
                "frame_last_seen": 0,
                "frames_seen": 1,
                "bbox_last": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                "confidences": [round(conf, 4)],
                "aggregated_confidence": round(conf, 4),
                "label": None,
                "evidence_url": None,
                "processed_image_url": None
            })

    metrics = {
        "processing_latency_ms": latency_ms,
        "frames_processed": 1,
        "model_version": "yolov8",
        "aggregator_window_s": None
    }
    return detected, metrics

def draw_detections_and_return_bytes(image_bytes: bytes, detections: List[Dict]) -> bytes:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for d in detections:
        if not d.get("bbox_last"):
            continue
        x, y, w, h = d["bbox_last"]
        x2, y2 = x + w, y + h
        draw.rectangle([x, y, x2, y2], outline="red", width=3)
        txt = f"{d['class_name']} {d['aggregated_confidence']:.2f}"
        if font:
            draw.text((x + 5, y + 5), txt, fill="red", font=font)
        else:
            draw.text((x + 5, y + 5), txt, fill="red")

    out = io.BytesIO()
    img.save(out, format="JPEG")
    return out.getvalue()
