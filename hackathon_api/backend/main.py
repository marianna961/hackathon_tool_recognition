import os
import uuid
import datetime
import json
import zipfile
import tempfile
from typing import List, Optional, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from minio import Minio
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
import asyncio
from sqlalchemy import create_engine, Column, String, DateTime, Integer, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Настройка базы данных
DB_DSN = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@db:5432/postgres")
engine = create_engine(DB_DSN)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Transaction(Base):
    __tablename__ = "transactions"
    id = Column(String, primary_key=True, index=True)
    sequence_number = Column(Integer, unique=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    data = Column(JSON)

Base.metadata.create_all(bind=engine)

def save_to_database(transaction_id: str, sequence_number: int, data: dict):
    db = SessionLocal()
    try:
        tx = Transaction(id=transaction_id, sequence_number=sequence_number, data=data)
        db.add(tx)
        db.commit()
        print(f"✅ Transaction {transaction_id} saved to database")
    except Exception as e:
        db.rollback()
        print(f"❌ Database error: {e}")
    finally:
        db.close()

# Глобальный счетчик для transaction_id
transaction_counter = 0

# Environment variables
MINIO_URL = os.getenv("MINIO_URL", "http://minio:9000")
MINIO_KEY = os.getenv("MINIO_KEY", "minioadmin")
MINIO_SECRET = os.getenv("MINIO_SECRET", "minioadmin")
MINIO_BUCKET_RAW = os.getenv("MINIO_BUCKET_RAW", "raw-images")
MINIO_BUCKET_VIZ = os.getenv("MINIO_BUCKET_VIZ", "viz-images")

# FastAPI app
app = FastAPI(title="Tool Recognition API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MinIO client
minio_client = Minio(
    MINIO_URL.replace("http://", "").replace("https://", ""),
    access_key=MINIO_KEY,
    secret_key=MINIO_SECRET,
    secure=False
)

# Create buckets if not exist
for bucket in [MINIO_BUCKET_RAW, MINIO_BUCKET_VIZ]:
    try:
        if not minio_client.bucket_exists(bucket):
            minio_client.make_bucket(bucket)
        print(f"✅ Bucket {bucket} ready")
    except Exception as e:
        print(f"⚠️ Bucket {bucket} error: {e}")

# Load YOLO segmentation model
try:
    model = YOLO("best.pt")
    print("✅ YOLO model loaded successfully")
    print(f"📊 Model names: {model.names}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

# Tool classes mapping based on your new class labels
TOOL_CLASSES = [
    {"class_id": "P10", "name": "Combination Wrench 3/4", "model_class": "10_Combination_Wrench_3_4"},
    {"class_id": "P11", "name": "Diagonal Cutters", "model_class": "11_Diagonal_Cutters"},
    {"class_id": "P01", "name": "Flathead Screwdriver", "model_class": "1_Flathead_Screwdriver"},
    {"class_id": "P02", "name": "Phillips Screwdriver", "model_class": "2_Phillips_Screwdriver"},
    {"class_id": "P03", "name": "Pozidriv Screwdriver", "model_class": "3_Pozidriv_Screwdriver"},
    {"class_id": "P04", "name": "Hand Drill", "model_class": "4_Hand_Drill"},
    {"class_id": "P05", "name": "Safety Wire Pliers", "model_class": "5_Safety_Wire_Pliers"},
    {"class_id": "P06", "name": "Slip Joint Pliers", "model_class": "6_Slip_Joint_Pliers"},
    {"class_id": "P07", "name": "Circlip Pliers", "model_class": "7_Circlip_Pliers"},
    {"class_id": "P08", "name": "Adjustable Wrench", "model_class": "8_Adjustable_Wrench"},
    {"class_id": "P09", "name": "Oil Can Opener", "model_class": "9_Oil_Can_Opener"}
]

# Utility functions
def get_next_transaction_id(photo_name: str) -> str:
    global transaction_counter
    transaction_counter += 1
    return f"TX_{transaction_counter:05d}_{photo_name}"

def save_to_minio(bucket: str, key: str, data: bytes, content_type: str = "image/jpeg") -> str:
    try:
        minio_client.put_object(bucket, key, BytesIO(data), len(data), content_type)
        return f"http://localhost:9001/browser/{bucket}/{key}"
    except Exception as e:
        print(f"MinIO error: {e}")
        return ""

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Вычисляет IoU между двумя bounding box'ами в формате [x1, y1, x2, y2]"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0

# Health check
@app.get("/")
def root():
    return {"message": "Tool Recognition API", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

# Single image prediction
@app.post("/predict/single")
async def predict_single(
    file: UploadFile = File(...),
    event_type: str = Form("hand_out"),
    camera_id: str = Form("table_cam_1"),
    operator_id: Optional[str] = Form(None)
):
    """Обработка одного изображения"""
    return await process_files([file], event_type, camera_id, operator_id)

# Multiple images prediction
@app.post("/predict/batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    event_type: str = Form("hand_out"),
    camera_id: str = Form("table_cam_1"),
    operator_id: Optional[str] = Form(None)
):
    """Обработка нескольких изображений - КАЖДОЕ ИЗ ИЗОБРАЖЕНИЙ ИМЕЕТ СВОЙ ОТЧЕТ"""
    return await process_files(files, event_type, camera_id, operator_id)

# ZIP folder prediction
@app.post("/predict/zip")
async def predict_zip(
    zip_file: UploadFile = File(...),
    event_type: str = Form("hand_out"),
    camera_id: str = Form("table_cam_1"),
    operator_id: Optional[str] = Form(None)
):
    """Обработка ZIP архива с изображениями - КАЖДОЕ ИЗ ИЗОБРАЖЕНИЙ ИМЕЕТ СВОЙ ОТЧЕТ"""
    contents = await zip_file.read()
    with BytesIO(contents) as zip_buffer:
        with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
            image_files = []
            for file_info in zip_ref.infolist():
                if file_info.filename.lower().endswith(('.jpg', '.jpeg', '.png')) and not file_info.is_dir():
                    extracted = zip_ref.read(file_info.filename)
                    image_files.append(UploadFile(
                        filename=os.path.basename(file_info.filename),
                        file=BytesIO(extracted)
                    ))
    
    if not image_files:
        raise HTTPException(400, "No images in ZIP")

    return await process_files(image_files, event_type, camera_id, operator_id)

async def process_files(files: List[UploadFile], event_type: str, camera_id: str, operator_id: Optional[str]):
    """Основная логика обработки файлов - КАЖДЫЙ ФАЙЛ ОБРАБАТЫВАЕТСЯ НЕЗАВИСИМО С УНИКАЛЬНОЙ ТРАНЗАКЦИЕЙ"""
    if not files:
        raise HTTPException(400, "No files provided")
    
    if model is None:
        raise HTTPException(500, "Model not loaded")

    tasks = [process_single_image(file, event_type, camera_id, operator_id) for file in files]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            print(f"❌ Processing error: {str(result)}")
            continue
        processed_results.append(result)

    return JSONResponse(content=processed_results)

async def process_single_image(file: UploadFile, event_type: str, camera_id: str, operator_id: Optional[str]) -> Dict:
    """Обрабатывает ОДНО изображение и возвращает результат"""
    start_time = datetime.datetime.now()
    
    image_name_without_ext = os.path.splitext(file.filename)[0]
    transaction_id = get_next_transaction_id(image_name_without_ext)
    
    try:
        contents = await file.read()
        img_array = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return {
                "transaction_id": transaction_id,
                "sequence_number": int(datetime.datetime.now().timestamp()),
                "event_type": event_type,
                "camera_id": camera_id,
                "operator_id": operator_id,
                "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
                "filename": file.filename,
                "error": "Failed to decode image",
                "status": "error",
                "raw_url": "",
                "viz_url": "",
                "processing_time_ms": 0,
                "summary": {
                    "total_expected": len(TOOL_CLASSES),
                    "total_detected": 0,
                    "match_percent": 0,
                    "status": "error"
                },
                "detected_items": [],
                "missing_items": [{"class_id": tool["class_id"], "name": tool["name"], "missing_qty": 1} for tool in TOOL_CLASSES],
                "alerts": ["manual_count_required", "missing_tools"]
            }

        orig_height, orig_width = img.shape[:2]
        raw_key = f"{transaction_id}/raw/{file.filename}"
        raw_url = save_to_minio(MINIO_BUCKET_RAW, raw_key, contents)
        
        results = model(img, verbose=False, conf=0.25)
        
        detected_tools = {}
        viz_img = img.copy()
        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]
        all_detections = []
        
        for r in results:
            if hasattr(r, 'boxes') and r.boxes is not None:
                boxes = r.boxes
                for i, box in enumerate(boxes):
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    if 0 <= cls < len(model.names):
                        model_class_name = model.names[cls]
                        tool = next((t for t in TOOL_CLASSES if t["model_class"] == model_class_name), None)
                        if tool:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            if hasattr(r, 'orig_shape'):
                                model_height, model_width = r.orig_shape
                            else:
                                model_height, model_width = img.shape[:2]
                            
                            scale_x = orig_width / model_width
                            scale_y = orig_height / model_height
                            
                            bbox_scaled = [
                                int(x1 * scale_x),
                                int(y1 * scale_y), 
                                int(x2 * scale_x),
                                int(y2 * scale_y)
                            ]
                            
                            all_detections.append({
                                'tool_key': tool["class_id"],
                                'bbox': bbox_scaled,
                                'confidence': conf,
                                'class_name': tool["name"]
                            })

        # Улучшенная логика IoU для выбора только одной детекции
        final_detections = []
        used_indices = set()
        
        for i, det in enumerate(all_detections):
            if i in used_indices:
                continue
                
            current_bbox = det['bbox']
            current_tool = det['tool_key']
            current_conf = det['confidence']
            
            # Проверяем пересечение только с детекциями того же инструмента
            overlapping = False
            for j, other_det in enumerate(all_detections):
                if j in used_indices or i == j:
                    continue
                if other_det['tool_key'] == current_tool:
                    iou = calculate_iou(current_bbox, other_det['bbox'])
                    if iou > 0.5:  # Порог IoU = 50%
                        overlapping = True
                        if other_det['confidence'] > current_conf:
                            used_indices.add(i)
                            break
                        else:
                            used_indices.add(j)
            
            if not overlapping and i not in used_indices:
                final_detections.append(det)
                used_indices.add(i)

        # Группируем финальные детекции по инструментам
        for det in final_detections:
            tool_key = det['tool_key']
            bbox = det['bbox']
            conf = det['confidence']
            
            if tool_key in detected_tools:
                detected_tools[tool_key]["qty"] += 1
                detected_tools[tool_key]["confidences"].append(conf)
                detected_tools[tool_key]["bboxes"].append(bbox)
                detected_tools[tool_key]["aggregated_confidence"] = sum(detected_tools[tool_key]["confidences"]) / detected_tools[tool_key]["qty"]
            else:
                detected_tools[tool_key] = {
                    "class_id": tool_key,
                    "class_name": det['class_name'],
                    "qty": 1,
                    "confidences": [conf],
                    "aggregated_confidence": conf,
                    "bboxes": [bbox],
                    "label": None,
                    "evidence_url": raw_url
                }

        # Рисуем bounding boxes на изображении
        for tool_key, tool_data in detected_tools.items():
            for bbox in tool_data["bboxes"]:
                color = colors[hash(tool_key) % len(colors)]
                x1, y1, x2, y2 = bbox
                cv2.rectangle(viz_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                label = f"{tool_data['class_name']} {tool_data['aggregated_confidence']:.2f}"
                cv2.putText(viz_img, label, (int(x1), int(y1)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Save visualization
        _, viz_buffer = cv2.imencode(".jpg", viz_img)
        viz_key = f"{transaction_id}/viz/{file.filename}"
        viz_url = save_to_minio(MINIO_BUCKET_VIZ, viz_key, viz_buffer.tobytes())
        
        # Create COMPLETE report for this single image
        detected_items = list(detected_tools.values())
        detected_class_ids = set(detected_tools.keys())
        
        # Find missing tools for THIS IMAGE
        missing_items = [
            {"class_id": tool["class_id"], "name": tool["name"], "missing_qty": 1}
            for tool in TOOL_CLASSES
            if tool["class_id"] not in detected_class_ids
        ]
        
        # Calculate summary for THIS IMAGE
        total_expected = len(TOOL_CLASSES)
        total_detected = sum(item["qty"] for item in detected_items)
        match_percent = round((total_detected / total_expected) * 100, 2) if total_expected > 0 else 0
        
        # Alerts for THIS IMAGE
        alerts = []
        if match_percent < 95:
            alerts.append("manual_count_required")
        if missing_items:
            alerts.append("missing_tools")
        
        processing_time_ms = int((datetime.datetime.now() - start_time).total_seconds() * 1000)
        
        return {
            "transaction_id": transaction_id,
            "sequence_number": int(datetime.datetime.now().timestamp()),
            "event_type": event_type,
            "camera_id": camera_id,
            "operator_id": operator_id,
            "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "filename": file.filename,
            "status": "success",
            "raw_url": raw_url,
            "viz_url": viz_url,
            "processing_time_ms": processing_time_ms,
            "summary": {
                "total_expected": total_expected,
                "total_detected": total_detected,
                "match_percent": match_percent,
                "status": "success" if match_percent >= 95 else "needs_manual_check"
            },
            "detected_items": detected_items,
            "missing_items": missing_items,
            "alerts": alerts
        }
        
    except Exception as e:
        processing_time_ms = int((datetime.datetime.now() - start_time).total_seconds() * 1000)
        return {
            "transaction_id": transaction_id,
            "sequence_number": int(datetime.datetime.now().timestamp()),
            "event_type": event_type,
            "camera_id": camera_id,
            "operator_id": operator_id,
            "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "filename": file.filename,
            "error": str(e),
            "status": "error",
            "raw_url": "",
            "viz_url": "",
            "processing_time_ms": processing_time_ms,
            "summary": {
                "total_expected": len(TOOL_CLASSES),
                "total_detected": 0,
                "match_percent": 0,
                "status": "error"
            },
            "detected_items": [],
            "missing_items": [{"class_id": tool["class_id"], "name": tool["name"], "missing_qty": 1} for tool in TOOL_CLASSES],
            "alerts": ["manual_count_required", "missing_tools"]
        }

# Get presigned MinIO URL
@app.get("/minio/presigned")
def get_presigned_url(bucket: str, key: str):
    """Получить presigned URL для MinIO файла"""
    try:
        url = minio_client.presigned_get_object(bucket, key, expires=datetime.timedelta(days=1))
        return {"url": url}
    except Exception as e:
        raise HTTPException(500, str(e))

# Get model info
@app.get("/model/info")
def get_model_info():
    if model is None:
        return {"status": "not_loaded", "message": "Model failed to load"}
    
    return {
        "status": "loaded",
        "names": model.names,
        "task": getattr(model, 'task', 'unknown')
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)