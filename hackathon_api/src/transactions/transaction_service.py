# src/transactions/transaction_service.py
import uuid
import datetime
import json
import zipfile
import io
from typing import Optional, List, Dict
from fastapi import UploadFile, HTTPException

from src.ml.inference import run_inference_on_image_bytes, draw_detections_and_return_bytes
from src.evidence.storage_service import store_evidence
from src.transactions.transaction_repository import save_transaction_to_file, save_transaction_to_db
from src.config.project_config import settings

def build_summary(expected_list: List[Dict], detected_items: List[Dict]) -> Dict:
    detected_total = len(detected_items)
    expected_total = sum(int(it.get("expected_qty", 1)) for it in expected_list) if expected_list else detected_total
    match_percent = round(100.0 * detected_total / max(1, expected_total), 2)
    missing = []
    if expected_list:
        for exp in expected_list:
            cid = exp.get("class_id")
            name = exp.get("name")
            exp_qty = int(exp.get("expected_qty", 1))
            det_qty = sum(1 for d in detected_items if d.get("class_id") == cid)
            if det_qty < exp_qty:
                missing.append({"class_id": cid, "name": name, "missing_qty": exp_qty - det_qty})
    alerts = []
    if match_percent < float(settings.MATCH_THRESHOLD):
        alerts.append("manual_count_required")
    return {
        "expected_total": expected_total,
        "detected_total": detected_total,
        "match_percent": match_percent,
        "missing": missing,
        "alerts": alerts
    }

async def process_image_transaction(image: UploadFile, event_type: str, camera_id: str, operator_id: Optional[str], expected_list_json: Optional[str]) -> Dict:
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="empty image")

    tx_id = str(uuid.uuid4())
    ts = datetime.datetime.utcnow().isoformat() + "Z"

    detected_items, metrics = run_inference_on_image_bytes(image_bytes)
    processed_bytes = draw_detections_and_return_bytes(image_bytes, detected_items) if detected_items else None

    original_url = store_evidence(tx_id, image.filename or "image.jpg", image_bytes)
    processed_url = None
    if processed_bytes:
        processed_url = store_evidence(tx_id, f"processed_{image.filename or 'image.jpg'}", processed_bytes)

    for d in detected_items:
        d["evidence_url"] = original_url
        d["processed_image_url"] = processed_url

    expected_list = []
    if expected_list_json:
        try:
            expected_list = json.loads(expected_list_json)
        except Exception:
            expected_list = []

    summary = build_summary(expected_list, detected_items)

    transaction = {
        "transaction_id": tx_id,
        "event_type": event_type,
        "timestamp_utc": ts,
        "camera_id": camera_id,
        "operator_id": operator_id,
        "expected_list": expected_list,
        "detected_items": detected_items,
        "summary": summary,
        "raw_metrics": metrics
    }

    save_transaction_to_file(transaction)
    save_transaction_to_db(transaction)

    return transaction

async def process_batch_archive(archive: UploadFile, event_type: str, camera_id: str, operator_id: Optional[str], expected_list_json: Optional[str]) -> Dict:
    content = await archive.read()
    if not content:
        raise HTTPException(status_code=400, detail="empty archive")
    expected_list = []
    if expected_list_json:
        try:
            expected_list = json.loads(expected_list_json)
        except Exception:
            expected_list = []

    batch_id = str(uuid.uuid4())
    results = []
    z = zipfile.ZipFile(io.BytesIO(content))
    members = [m for m in z.namelist() if m.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not members:
        raise HTTPException(status_code=400, detail="zip contains no images")
    for filename in members:
        img_bytes = z.read(filename)
        tx_id = str(uuid.uuid4())
        ts = datetime.datetime.utcnow().isoformat() + "Z"

        detected_items, metrics = run_inference_on_image_bytes(img_bytes)
        processed_bytes = draw_detections_and_return_bytes(img_bytes, detected_items) if detected_items else None

        original_url = store_evidence(tx_id, filename, img_bytes)
        processed_url = None
        if processed_bytes:
            processed_url = store_evidence(tx_id, f"processed_{filename}", processed_bytes)

        for d in detected_items:
            d["evidence_url"] = original_url
            d["processed_image_url"] = processed_url

        summary = build_summary(expected_list, detected_items)

        transaction = {
            "transaction_id": tx_id,
            "event_type": event_type,
            "timestamp_utc": ts,
            "camera_id": camera_id,
            "operator_id": operator_id,
            "expected_list": expected_list,
            "detected_items": detected_items,
            "summary": summary,
            "raw_metrics": metrics
        }

        save_transaction_to_file(transaction)
        save_transaction_to_db(transaction)
        results.append(transaction)

    batch_report = {
        "batch_id": batch_id,
        "total_files": len(results),
        "results": results
    }
    # Save batch report as file for convenience
    save_transaction_to_file({"transaction_id": batch_id, "batch_report": batch_report})
    return batch_report
