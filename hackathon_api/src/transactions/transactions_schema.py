# src/transactions/transaction_schema.py
from typing import List, Optional, Any, Dict
from pydantic import BaseModel

class ExpectedItem(BaseModel):
    class_id: str
    name: str
    expected_qty: int = 1

class Label(BaseModel):
    type: str
    text: str
    confidence: float

class DetectedItem(BaseModel):
    class_id: str
    class_name: str
    frame_first_seen: Optional[int] = None
    frame_last_seen: Optional[int] = None
    frames_seen: int = 1
    bbox_last: Optional[List[int]] = None
    confidences: Optional[List[float]] = None
    aggregated_confidence: Optional[float] = None
    label: Optional[Label] = None
    evidence_url: Optional[str] = None
    processed_image_url: Optional[str] = None

class Summary(BaseModel):
    expected_total: int
    detected_total: int
    match_percent: float
    missing: Optional[List[Dict[str, Any]]] = []
    alerts: Optional[List[str]] = []

class RawMetrics(BaseModel):
    processing_latency_ms: int
    frames_processed: int = 1
    model_version: str
    aggregator_window_s: Optional[float] = None

class Transaction(BaseModel):
    transaction_id: str
    event_type: str
    timestamp_utc: str
    camera_id: Optional[str] = None
    operator_id: Optional[str] = None
    expected_list: Optional[List[ExpectedItem]] = []
    detected_items: Optional[List[DetectedItem]] = []
    summary: Summary
    raw_metrics: RawMetrics
