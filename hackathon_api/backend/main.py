# # import os
# # import uuid
# # import datetime
# # import json
# # from typing import List, Optional
# # from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# # from fastapi.responses import JSONResponse
# # from sqlalchemy import create_engine, Column, String, Text, DateTime
# # from sqlalchemy.ext.declarative import declarative_base
# # from sqlalchemy.orm import sessionmaker
# # from minio import Minio
# # from minio.error import S3Error
# # from ultralytics import YOLO
# # import cv2
# # import numpy as np
# # from io import BytesIO

# # # Environment variables
# # MINIO_URL = os.getenv("MINIO_URL", "http://minio:9000")
# # MINIO_KEY = os.getenv("MINIO_KEY", "minioadmin")
# # MINIO_SECRET = os.getenv("MINIO_SECRET", "minioadmin")
# # MINIO_BUCKET_RAW = os.getenv("MINIO_BUCKET_RAW", "raw-images")
# # MINIO_BUCKET_VIZ = os.getenv("MINIO_BUCKET_VIZ", "viz-images")
# # DB_DSN = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@db:5432/postgres")

# # # FastAPI app
# # app = FastAPI(title="Tool Inference Service", version="0.1.0")

# # # SQLAlchemy setup
# # engine = create_engine(DB_DSN)
# # SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# # Base = declarative_base()

# # class Transaction(Base):
# #     __tablename__ = "transactions"
# #     id = Column(String, primary_key=True, index=True)
# #     created_at = Column(DateTime, default=datetime.datetime.utcnow)
# #     data = Column(Text, nullable=False)

# # Base.metadata.create_all(bind=engine)

# # # MinIO client
# # minio_client = Minio(
# #     MINIO_URL.replace("http://", "").replace("https://", ""),
# #     access_key=MINIO_KEY,
# #     secret_key=MINIO_SECRET,
# #     secure=False  # Для локального MinIO
# # )

# # # Create buckets if not exist
# # for bucket in [MINIO_BUCKET_RAW, MINIO_BUCKET_VIZ]:
# #     try:
# #         if not minio_client.bucket_exists(bucket):
# #             minio_client.make_bucket(bucket)
# #         print(f"✅ Bucket {bucket} ready")
# #     except Exception as e:
# #         print(f"⚠️ Bucket {bucket} error: {e}")

# # # Load YOLO model
# # try:
# #     model = YOLO("best.pt")  # Убедись что файл в правильной папке
# #     print("✅ YOLO model loaded successfully")
# #     print(f"📊 Model names: {model.names}")
# # except Exception as e:
# #     print(f"❌ Failed to load model: {e}")
# #     model = None

# # # Tool classes with 11 expected tools
# # TOOL_CLASSES = [
# #     {"class_id": "P10", "name": "Ключ рожковый/накидной"},
# #     {"class_id": "P11", "name": "Бокорезы"},
# #     {"class_id": "P01", "name": "Отвертка «-»"},
# #     {"class_id": "P02", "name": "Отвертка «+»"},
# #     {"class_id": "P03", "name": "Отвертка на смещенный крест"},
# #     {"class_id": "P04", "name": "Коловорот"},
# #     {"class_id": "P05", "name": "Пассатижи контровочные"},
# #     {"class_id": "P06", "name": "Пассатижи"},
# #     {"class_id": "P07", "name": "Шэрница"},
# #     {"class_id": "P08", "name": "Разводной ключ"},
# #     {"class_id": "P09", "name": "Открывашка для банок с маслом"}
# # ]

# # # Health check
# # @app.get("/")
# # def root():
# #     return {"message": "Tool Inference API", "version": "0.1.0"}

# # @app.get("/health")
# # def health():
# #     return {"status": "healthy", "model_loaded": model is not None}

# # # Predict endpoint
# # @app.post("/predict")
# # async def predict(
# #     files: List[UploadFile] = File(...),
# #     event_type: str = Form("hand_out"),
# #     camera_id: str = Form("table_cam_1"),
# #     operator_id: Optional[str] = Form(None)
# # ):
# #     if not files:
# #         raise HTTPException(status_code=400, detail="No files uploaded")
    
# #     if model is None:
# #         raise HTTPException(status_code=500, detail="YOLO model not loaded")

# #     transaction_id = str(uuid.uuid4())
# #     timestamp_utc = datetime.datetime.utcnow().isoformat() + "Z"
    
# #     # Expected tools list (always 11 tools)
# #     expected_list = [{"class_id": tool["class_id"], "name": tool["name"], "expected_qty": 1} for tool in TOOL_CLASSES]
    
# #     detected_items = []
# #     frames_processed = 0
# #     start_time = datetime.datetime.now()

# #     detected_tools = {}  # To aggregate detections across images

# #     for idx, file in enumerate(files):
# #         try:
# #             contents = await file.read()
# #             img_array = np.frombuffer(contents, np.uint8)
# #             img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

# #             if img is None:
# #                 print(f"❌ Failed to decode image {file.filename}")
# #                 continue

# #             # Run inference
# #             results = model(img, verbose=False, conf=0.25)  # YOLO inference

# #             # Save raw image to MinIO
# #             raw_key = f"{transaction_id}/raw/{file.filename}"
# #             try:
# #                 minio_client.put_object(
# #                     MINIO_BUCKET_RAW, raw_key, 
# #                     BytesIO(contents), len(contents), 
# #                     content_type="image/jpeg"
# #                 )
# #                 raw_url = f"{MINIO_URL}/{MINIO_BUCKET_RAW}/{raw_key}"
# #             except Exception as e:
# #                 print(f"❌ MinIO raw save error: {e}")
# #                 raw_url = None

# #             # Process results
# #             viz_img = img.copy()
# #             colors = [
# #                 (255, 0, 0), (0, 255, 0), (0, 0, 255),
# #                 (255, 255, 0), (255, 0, 255), (0, 255, 255)
# #             ]

# #             for r_idx, r in enumerate(results):
# #                 # Обрабатываем сегментацию
# #                 if hasattr(r, 'masks') and r.masks is not None:
# #                     for i, (mask, box) in enumerate(zip(r.masks.data, r.boxes.xyxy)):
# #                         if i >= len(r.boxes.conf):
# #                             continue
                            
# #                         conf = float(r.boxes.conf[i])
# #                         cls = int(r.boxes.cls[i])
                        
# #                         # Маппинг class_id на наши P01, P02... по индексу
# #                         if 0 <= cls < len(TOOL_CLASSES):
# #                             tool = TOOL_CLASSES[cls]
# #                             tool_key = tool["class_id"]
                            
# #                             # Получаем bbox из маски
# #                             mask_np = mask.cpu().numpy()
# #                             y_indices, x_indices = np.where(mask_np > 0)
                            
# #                             if len(x_indices) > 0 and len(y_indices) > 0:
# #                                 x1, x2 = np.min(x_indices), np.max(x_indices)
# #                                 y1, y2 = np.min(y_indices), np.max(y_indices)
# #                                 bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                                
# #                                 if tool_key not in detected_tools:
# #                                     detected_tools[tool_key] = {
# #                                         "class_id": tool["class_id"],
# #                                         "class_name": tool["name"],
# #                                         "frame_first_seen": idx,
# #                                         "frame_last_seen": idx,
# #                                         "frames_seen": 1,
# #                                         "bbox_last": bbox,
# #                                         "confidences": [conf],
# #                                         "aggregated_confidence": conf,
# #                                         "label": None,
# #                                         "evidence_url": raw_url
# #                                     }
# #                                 else:
# #                                     det = detected_tools[tool_key]
# #                                     det["frame_last_seen"] = idx
# #                                     det["frames_seen"] += 1
# #                                     det["bbox_last"] = bbox
# #                                     det["confidences"].append(conf)
# #                                     det["aggregated_confidence"] = sum(det["confidences"]) / len(det["confidences"])

# #                             # Визуализация маски
# #                             color = colors[r_idx % len(colors)]
# #                             mask_resized = cv2.resize(mask_np, (img.shape[1], img.shape[0]))
# #                             viz_img[mask_resized > 0.5] = (
# #                                 0.6 * viz_img[mask_resized > 0.5] + 0.4 * np.array(color)
# #                             ).astype(np.uint8)

# #             # Сохраняем визуализацию
# #             try:
# #                 _, viz_buffer = cv2.imencode(".jpg", viz_img)
# #                 viz_key = f"{transaction_id}/viz/{file.filename}"
# #                 minio_client.put_object(
# #                     MINIO_BUCKET_VIZ, viz_key,
# #                     BytesIO(viz_buffer.tobytes()), len(viz_buffer.tobytes()),
# #                     content_type="image/jpeg"
# #                 )
# #                 viz_url = f"{MINIO_URL}/{MINIO_BUCKET_VIZ}/{viz_key}"
                
# #                 # Обновляем evidence_url на визуализацию
# #                 for tool_key in detected_tools:
# #                     detected_tools[tool_key]["evidence_url"] = viz_url
                    
# #             except Exception as e:
# #                 print(f"❌ MinIO viz save error: {e}")

# #             frames_processed += 1
# #             print(f"✅ Processed {file.filename}, detected {len(detected_tools)} tools")

# #         except Exception as e:
# #             print(f"❌ Error processing {file.filename}: {e}")
# #             continue

# #     # Формируем финальный список detected_items
# #     detected_items = list(detected_tools.values())

# #     # Summary calculation
# #     expected_total = len(expected_list)  # Always 11
# #     detected_total = len(detected_items)
# #     match_percent = round((detected_total / expected_total * 100) if expected_total > 0 else 0.0, 2)
    
# #     missing = [
# #         {"class_id": tool["class_id"], "name": tool["name"], "missing_qty": 1} 
# #         for tool in expected_list 
# #         if tool["class_id"] not in detected_tools
# #     ]
    
# #     alerts = []
# #     if match_percent < 95:  # Порог 95%
# #         alerts.append("manual_count_required")
# #     if missing:
# #         alerts.append("missing_tools")

# #     summary = {
# #         "expected_total": expected_total,
# #         "detected_total": detected_total,
# #         "match_percent": match_percent,
# #         "missing": missing,
# #         "alerts": alerts
# #     }

# #     processing_latency_ms = int((datetime.datetime.now() - start_time).total_seconds() * 1000)

# #     raw_metrics = {
# #         "processing_latency_ms": processing_latency_ms,
# #         "frames_processed": frames_processed,
# #         "model_version": "yolov8_segmentation",
# #         "aggregator_window_s": None
# #     }

# #     # Формируем транзакцию
# #     transaction_data = {
# #         "transaction_id": transaction_id,
# #         "event_type": event_type,
# #         "timestamp_utc": timestamp_utc,
# #         "camera_id": camera_id,
# #         "operator_id": operator_id,
# #         "expected_list": expected_list,
# #         "detected_items": detected_items,
# #         "summary": summary,
# #         "raw_metrics": raw_metrics
# #     }

# #     # Save to Postgres
# #     db = SessionLocal()
# #     try:
# #         tx = Transaction(id=transaction_id, data=json.dumps(transaction_data, ensure_ascii=False))
# #         db.add(tx)
# #         db.commit()
# #         print(f"✅ Transaction {transaction_id} saved to database")
# #     except Exception as e:
# #         db.rollback()
# #         print(f"❌ Database error: {e}")
# #         # Не падаем, просто логируем ошибку БД
# #     finally:
# #         db.close()

# #     return JSONResponse(content=transaction_data)

# # # Get transaction by ID
# # @app.get("/transactions/{transaction_id}")
# # def get_transaction(transaction_id: str):
# #     db = SessionLocal()
# #     try:
# #         tx = db.query(Transaction).filter(Transaction.id == transaction_id).first()
# #         if not tx:
# #             raise HTTPException(status_code=404, detail="Transaction not found")
# #         return json.loads(tx.data)
# #     finally:
# #         db.close()

# # # Get model info
# # @app.get("/model/info")
# # def get_model_info():
# #     if model is None:
# #         return {"status": "not_loaded", "message": "Model failed to load"}
    
# #     return {
# #         "status": "loaded",
# #         "names": model.names,
# #         "task": getattr(model, 'task', 'unknown')
# #     }

# # if __name__ == "__main__":
# #     import uvicorn
# #     uvicorn.run(app, host="0.0.0.0", port=8000)

# import os
# import uuid
# import datetime
# import json
# from typing import List, Optional
# from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# from fastapi.responses import JSONResponse
# from sqlalchemy import create_engine, Column, String, Text, DateTime
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker
# from minio import Minio
# from minio.error import S3Error
# from ultralytics import YOLO
# import cv2
# import numpy as np
# from io import BytesIO

# # Environment variables
# MINIO_URL = os.getenv("MINIO_URL", "http://minio:9000")
# MINIO_KEY = os.getenv("MINIO_KEY", "minioadmin")
# MINIO_SECRET = os.getenv("MINIO_SECRET", "minioadmin")
# MINIO_BUCKET_RAW = os.getenv("MINIO_BUCKET_RAW", "raw-images")
# MINIO_BUCKET_VIZ = os.getenv("MINIO_BUCKET_VIZ", "viz-images")
# DB_DSN = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@db:5432/postgres")

# # FastAPI app
# app = FastAPI(title="Tool Inference Service", version="0.1.0")

# # SQLAlchemy setup
# engine = create_engine(DB_DSN)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base = declarative_base()

# class Transaction(Base):
#     __tablename__ = "transactions"
#     id = Column(String, primary_key=True, index=True)
#     created_at = Column(DateTime, default=datetime.datetime.utcnow)
#     data = Column(Text, nullable=False)

# Base.metadata.create_all(bind=engine)

# # MinIO client
# minio_client = Minio(
#     MINIO_URL.replace("http://", "").replace("https://", ""),
#     access_key=MINIO_KEY,
#     secret_key=MINIO_SECRET,
#     secure=False  # Для локального MinIO
# )

# # Create buckets if not exist
# for bucket in [MINIO_BUCKET_RAW, MINIO_BUCKET_VIZ]:
#     try:
#         if not minio_client.bucket_exists(bucket):
#             minio_client.make_bucket(bucket)
#         print(f"✅ Bucket {bucket} ready")
#     except Exception as e:
#         print(f"⚠️ Bucket {bucket} error: {e}")

# # Load YOLO model
# try:
#     model = YOLO("best.pt")  # Убедись что файл в правильной папке
#     print("✅ YOLO model loaded successfully")
#     print(f"📊 Model names: {model.names}")
# except Exception as e:
#     print(f"❌ Failed to load model: {e}")
#     model = None

# # Tool classes with 11 expected tools, mapped to model indices 0-10
# TOOL_CLASSES = [
#     {"class_id": "P01", "name": "Отвертка «-»"},      # Index 0 (class0)
#     {"class_id": "P02", "name": "Отвертка «+»"},      # Index 1 (class1)
#     {"class_id": "P03", "name": "Отвертка на смещенный крест"},  # Index 2 (class2)
#     {"class_id": "P04", "name": "Коловорот"},          # Index 3 (class3)
#     {"class_id": "P05", "name": "Пассатижи контровочные"},  # Index 4 (class4)
#     {"class_id": "P06", "name": "Пассатижи"},          # Index 5 (class5)
#     {"class_id": "P07", "name": "Шэрница"},            # Index 6 (class6)
#     {"class_id": "P08", "name": "Разводной ключ"},     # Index 7 (class7)
#     {"class_id": "P09", "name": "Открывашка для банок с маслом"},  # Index 8 (class8)
#     {"class_id": "P10", "name": "Ключ рожковый/накидной"},  # Index 9 (class9)
#     {"class_id": "P11", "name": "Бокорезы"}            # Index 10 (class99)
# ]

# # Health check
# @app.get("/")
# def root():
#     return {"message": "Tool Inference API", "version": "0.1.0"}

# @app.get("/health")
# def health():
#     return {"status": "healthy", "model_loaded": model is not None}

# # Predict endpoint
# @app.post("/predict")
# async def predict(
#     files: List[UploadFile] = File(...),
#     event_type: str = Form("hand_out"),
#     camera_id: str = Form("table_cam_1"),
#     operator_id: Optional[str] = Form(None)
# ):
#     if not files:
#         raise HTTPException(status_code=400, detail="No files uploaded")
    
#     if model is None:
#         raise HTTPException(status_code=500, detail="YOLO model not loaded")

#     transaction_id = str(uuid.uuid4())
#     timestamp_utc = datetime.datetime.utcnow().isoformat() + "Z"
    
#     # Expected tools list (always 11 tools)
#     expected_list = [{"class_id": tool["class_id"], "name": tool["name"], "expected_qty": 1} for tool in TOOL_CLASSES]
    
#     detected_items = []
#     frames_processed = 0
#     start_time = datetime.datetime.now()

#     detected_tools = {}  # To aggregate detections across images

#     for idx, file in enumerate(files):
#         try:
#             contents = await file.read()
#             img_array = np.frombuffer(contents, np.uint8)
#             img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

#             if img is None:
#                 print(f"❌ Failed to decode image {file.filename}")
#                 continue

#             # Run inference
#             results = model(img, verbose=False, conf=0.25)  # YOLO inference

#             # Save raw image to MinIO
#             raw_key = f"{transaction_id}/raw/{file.filename}"
#             try:
#                 minio_client.put_object(
#                     MINIO_BUCKET_RAW, raw_key, 
#                     BytesIO(contents), len(contents), 
#                     content_type="image/jpeg"
#                 )
#                 raw_url = f"{MINIO_URL}/{MINIO_BUCKET_RAW}/{raw_key}"
#             except Exception as e:
#                 print(f"❌ MinIO raw save error: {e}")
#                 raw_url = None

#             # Process results
#             viz_img = img.copy()
#             colors = [
#                 (255, 0, 0), (0, 255, 0), (0, 0, 255),
#                 (255, 255, 0), (255, 0, 255), (0, 255, 255)
#             ]

#             for r_idx, r in enumerate(results):
#                 # Обрабатываем bounding boxes
#                 boxes = r.boxes.xyxy  # Координаты bounding boxes [x1, y1, x2, y2]
#                 confs = r.boxes.conf  # Уверенности
#                 clss = r.boxes.cls    # Классы

#                 for i in range(len(boxes)):
#                     conf = float(confs[i])
#                     cls = int(clss[i])
                    
#                     # Маппинг class_id на наши P01, P02... по индексу модели (0-10)
#                     if 0 <= cls < len(TOOL_CLASSES):
#                         tool = TOOL_CLASSES[cls]
#                         tool_key = tool["class_id"]
#                         bbox = [int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2] - boxes[i][0]), int(boxes[i][3] - boxes[i][1])]  # [x1, y1, width, height]
                        
#                         if tool_key not in detected_tools:
#                             detected_tools[tool_key] = {
#                                 "class_id": tool["class_id"],
#                                 "class_name": tool["name"],
#                                 "frame_first_seen": idx,
#                                 "frame_last_seen": idx,
#                                 "frames_seen": 1,
#                                 "bbox_last": bbox,
#                                 "confidences": [conf],
#                                 "aggregated_confidence": conf,
#                                 "label": None,
#                                 "evidence_url": raw_url
#                             }
#                         else:
#                             det = detected_tools[tool_key]
#                             det["frame_last_seen"] = idx
#                             det["frames_seen"] += 1
#                             det["bbox_last"] = bbox
#                             det["confidences"].append(conf)
#                             det["aggregated_confidence"] = sum(det["confidences"]) / len(det["confidences"])

#                         # Визуализация bounding box
#                         color = colors[r_idx % len(colors)]
#                         x1, y1, x2, y2 = map(int, boxes[i])
#                         cv2.rectangle(viz_img, (x1, y1), (x2, y2), color, 2)
#                         label = f"{tool['name']} {conf:.2f}"
#                         cv2.putText(viz_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#             # Сохраняем визуализацию
#             try:
#                 _, viz_buffer = cv2.imencode(".jpg", viz_img)
#                 viz_key = f"{transaction_id}/viz/{file.filename}"
#                 minio_client.put_object(
#                     MINIO_BUCKET_VIZ, viz_key,
#                     BytesIO(viz_buffer.tobytes()), len(viz_buffer.tobytes()),
#                     content_type="image/jpeg"
#                 )
#                 viz_url = f"{MINIO_URL}/{MINIO_BUCKET_VIZ}/{viz_key}"
                
#                 # Обновляем evidence_url на визуализацию
#                 for tool_key in detected_tools:
#                     detected_tools[tool_key]["evidence_url"] = viz_url
                    
#             except Exception as e:
#                 print(f"❌ MinIO viz save error: {e}")

#             frames_processed += 1
#             print(f"✅ Processed {file.filename}, detected {len(detected_tools)} tools")

#         except Exception as e:
#             print(f"❌ Error processing {file.filename}: {e}")
#             continue

#     # Формируем финальный список detected_items
#     detected_items = list(detected_tools.values())

#     # Summary calculation
#     expected_total = len(expected_list)  # Always 11
#     detected_total = len(detected_items)
#     match_percent = round((detected_total / expected_total * 100) if expected_total > 0 else 0.0, 2)
    
#     missing = [
#         {"class_id": tool["class_id"], "name": tool["name"], "missing_qty": 1} 
#         for tool in expected_list 
#         if tool["class_id"] not in detected_tools
#     ]
    
#     alerts = []
#     if match_percent < 95:  # Порог 95%
#         alerts.append("manual_count_required")
#     if missing:
#         alerts.append("missing_tools")

#     summary = {
#         "expected_total": expected_total,
#         "detected_total": detected_total,
#         "match_percent": match_percent,
#         "missing": missing,
#         "alerts": alerts
#     }

#     processing_latency_ms = int((datetime.datetime.now() - start_time).total_seconds() * 1000)

#     raw_metrics = {
#         "processing_latency_ms": processing_latency_ms,
#         "frames_processed": frames_processed,
#         "model_version": "yolov8_detection",
#         "aggregator_window_s": None
#     }

#     # Формируем транзакцию
#     transaction_data = {
#         "transaction_id": transaction_id,
#         "event_type": event_type,
#         "timestamp_utc": timestamp_utc,
#         "camera_id": camera_id,
#         "operator_id": operator_id,
#         "expected_list": expected_list,
#         "detected_items": detected_items,
#         "summary": summary,
#         "raw_metrics": raw_metrics
#     }

#     # Save to Postgres
#     db = SessionLocal()
#     try:
#         tx = Transaction(id=transaction_id, data=json.dumps(transaction_data, ensure_ascii=False))
#         db.add(tx)
#         db.commit()
#         print(f"✅ Transaction {transaction_id} saved to database")
#     except Exception as e:
#         db.rollback()
#         print(f"❌ Database error: {e}")
#         # Не падаем, просто логируем ошибку БД
#     finally:
#         db.close()

#     return JSONResponse(content=transaction_data)

# # Get transaction by ID
# @app.get("/transactions/{transaction_id}")
# def get_transaction(transaction_id: str):
#     db = SessionLocal()
#     try:
#         tx = db.query(Transaction).filter(Transaction.id == transaction_id).first()
#         if not tx:
#             raise HTTPException(status_code=404, detail="Transaction not found")
#         return json.loads(tx.data)
#     finally:
#         db.close()

# # Get model info
# @app.get("/model/info")
# def get_model_info():
#     if model is None:
#         return {"status": "not_loaded", "message": "Model failed to load"}
    
#     return {
#         "status": "loaded",
#         "names": model.names,
#         "task": getattr(model, 'task', 'unknown')
#     }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

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
from sqlalchemy import create_engine, Column, String, DateTime, Integer, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from minio import Minio
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
import asyncio

# Environment variables
MINIO_URL = os.getenv("MINIO_URL", "http://minio:9000")
MINIO_KEY = os.getenv("MINIO_KEY", "minioadmin")
MINIO_SECRET = os.getenv("MINIO_SECRET", "minioadmin")
MINIO_BUCKET_RAW = os.getenv("MINIO_BUCKET_RAW", "raw-images")
MINIO_BUCKET_VIZ = os.getenv("MINIO_BUCKET_VIZ", "viz-images")
DB_DSN = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@db:5432/postgres")

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

# SQLAlchemy setup
engine = create_engine(DB_DSN)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Transaction(Base):
    __tablename__ = "transactions"
    id = Column(String, primary_key=True, index=True)
    sequence_number = Column(Integer, unique=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    data = Column(JSON)  # JSONB для запросов

Base.metadata.create_all(bind=engine)

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

# Load YOLO model
try:
    model = YOLO("best.pt")
    print("✅ YOLO model loaded successfully")
    print(f"📊 Model names: {model.names}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

# Tool classes mapping - 11 инструментов
TOOL_CLASSES = [
    {"class_id": "P01", "name": "Отвертка «-»"},
    {"class_id": "P02", "name": "Отвертка «+»"},
    {"class_id": "P03", "name": "Отвертка на смещенный крест"},
    {"class_id": "P04", "name": "Коловорот"},
    {"class_id": "P05", "name": "Пассатижи контровочные"},
    {"class_id": "P06", "name": "Пассатижи"},
    {"class_id": "P07", "name": "Шэрница"},
    {"class_id": "P08", "name": "Разводной ключ"},
    {"class_id": "P09", "name": "Открывашка для банок с маслом"},
    {"class_id": "P10", "name": "Ключ рожковый/накидной"},
    {"class_id": "P11", "name": "Бокорезы"}
]

# Utility functions
def get_next_sequence_number() -> int:
    db = SessionLocal()
    try:
        last_tx = db.query(Transaction).order_by(Transaction.sequence_number.desc()).first()
        return last_tx.sequence_number + 1 if last_tx else 1
    except:
        return 1
    finally:
        db.close()

def save_to_minio(bucket: str, key: str, data: bytes, content_type: str = "image/jpeg") -> str:
    try:
        minio_client.put_object(bucket, key, BytesIO(data), len(data), content_type)
        return f"{MINIO_URL}/{bucket}/{key}"
    except Exception as e:
        print(f"MinIO error: {e}")
        return ""

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
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, zip_file.filename)
        with open(zip_path, "wb") as f:
            f.write(await zip_file.read())
        
        image_files = []
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            
            for file_info in zip_ref.infolist():
                if file_info.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(temp_dir, file_info.filename)
                    image_files.append({
                        'path': full_path,
                        'filename': file_info.filename
                    })
        
        # Process each image
        files_to_process = []
        for img_info in image_files:
            with open(img_info['path'], 'rb') as f:
                files_to_process.append(UploadFile(
                    filename=img_info['filename'],
                    file=f
                ))
        
        return await process_files(files_to_process, event_type, camera_id, operator_id)

async def process_files(files: List[UploadFile], event_type: str, camera_id: str, operator_id: Optional[str]):
    """Основная логика обработки файлов - КАЖДЫЙ ФАЙЛ ОБРАБАТЫВАЕТСЯ НЕЗАВИСИМО"""
    if not files:
        raise HTTPException(400, "No files provided")
    
    if model is None:
        raise HTTPException(500, "Model not loaded")

    # Create human-readable transaction ID with 7 digits
    sequence_number = get_next_sequence_number()
    transaction_id = f"TX_{sequence_number:07d}"
    start_time = datetime.datetime.now()

    # Process each file asynchronously
    tasks = [process_single_image(file, transaction_id, file_idx + 1) for file_idx, file in enumerate(files)]
    image_results = await asyncio.gather(*tasks)

    # Filter successful results
    successful_results = [result for result in image_results if result["status"] == "success"]
    
    # Build final response with INDIVIDUAL image reports
    result = {
        "transaction_id": transaction_id,
        "sequence_number": sequence_number,
        "event_type": event_type,
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "processing_info": {
            "mode": "cpu_based",
            "total_processing_time_ms": int((datetime.datetime.now() - start_time).total_seconds() * 1000),
            "avg_processing_time_ms": int(sum(result["processing_time_ms"] for result in image_results) / len(image_results)) if image_results else 0,
            "total_images_processed": len(image_results),
            "model_version": "yolov8_detection",
            "flops": "~15 GFLOPs"
        },
        "expected_tools": [{"class_id": t["class_id"], "name": t["name"], "expected_qty": 1} for t in TOOL_CLASSES],
        "images_processed": image_results,  # ← КАЖДОЕ ИЗОБРАЖЕНИЕ ИМЕЕТ СВОЙ ОТЧЕТ
        "overall_summary": calculate_overall_summary(image_results)
    }
    
    # Save to database
    save_to_database(transaction_id, sequence_number, result)
    
    return JSONResponse(content=result)

async def process_single_image(file: UploadFile, transaction_id: str, file_index: int) -> Dict:
    """Обрабатывает ОДНО изображение и возвращает ПОЛНЫЙ ОТЧЕТ для этого изображения"""
    start_time = datetime.datetime.now()
    try:
        contents = await file.read()
        img_array = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return {
                "filename": file.filename,
                "file_index": file_index,
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

        # Save original image
        raw_key = f"{transaction_id}/raw/{file.filename}"
        raw_url = save_to_minio(MINIO_BUCKET_RAW, raw_key, contents)
        
        # Run model inference
        results = model(img, verbose=False, conf=0.25)
        
        # Process detections for THIS IMAGE ONLY
        detected_tools = {}
        viz_img = img.copy()
        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]
        
        for r_idx, r in enumerate(results):
            if r.boxes is not None:
                boxes = r.boxes.xyxy
                confs = r.boxes.conf
                clss = r.boxes.cls
                
                for i in range(len(boxes)):
                    conf = float(confs[i])
                    cls = int(clss[i])
                    
                    if 0 <= cls < len(TOOL_CLASSES):
                        tool = TOOL_CLASSES[cls]
                        tool_key = tool["class_id"]
                        bbox = [int(boxes[i][0]), int(boxes[i][1]), 
                               int(boxes[i][2] - boxes[i][0]), 
                               int(boxes[i][3] - boxes[i][1])]
                        
                        # Handle duplicates: increment qty
                        if tool_key in detected_tools:
                            detected_tools[tool_key]["qty"] += 1
                            detected_tools[tool_key]["confidences"].append(conf)
                            detected_tools[tool_key]["bboxes"].append(bbox)
                            detected_tools[tool_key]["aggregated_confidence"] = sum(detected_tools[tool_key]["confidences"]) / detected_tools[tool_key]["qty"]
                        else:
                            detected_tools[tool_key] = {
                                "class_id": tool["class_id"],
                                "class_name": tool["name"],
                                "qty": 1,
                                "confidences": [conf],
                                "aggregated_confidence": conf,
                                "bboxes": [bbox],
                                "label": None,
                                "evidence_url": raw_url
                            }
                        
                        # Draw bounding box
                        color = colors[r_idx % len(colors)]
                        x1, y1, x2, y2 = map(int, boxes[i])
                        cv2.rectangle(viz_img, (x1, y1), (x2, y2), color, 2)
                        label = f"{tool['name']} {conf:.2f}"
                        cv2.putText(viz_img, label, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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
            "filename": file.filename,
            "file_index": file_index,
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
            "filename": file.filename,
            "file_index": file_index,
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

def calculate_overall_summary(image_results: List[Dict]) -> Dict:
    """Рассчитывает общую статистику по всем изображениям"""
    successful_images = [img for img in image_results if img["status"] == "success"]
    
    if not successful_images:
        return {
            "total_images": len(image_results),
            "successful_images": 0,
            "average_match_percent": 0,
            "images_with_issues": [img["file_index"] for img in image_results],
            "overall_status": "error"
        }
    
    total_match_percent = sum(img["summary"]["match_percent"] for img in successful_images)
    avg_match_percent = round(total_match_percent / len(successful_images), 2)
    
    images_with_issues = [
        img["file_index"] for img in image_results 
        if img["status"] != "success" or img["summary"]["match_percent"] < 95
    ]
    
    return {
        "total_images": len(image_results),
        "successful_images": len(successful_images),
        "average_match_percent": avg_match_percent,
        "images_with_issues": images_with_issues,
        "overall_status": "success" if avg_match_percent >= 95 else "needs_manual_check"
    }

def save_to_database(transaction_id: str, sequence_number: int, data: dict):
    """Сохраняет транзакцию в БД"""
    db = SessionLocal()
    try:
        tx = Transaction(
            id=transaction_id,
            sequence_number=sequence_number,
            data=data  # JSONB automatically handles
        )
        db.add(tx)
        db.commit()
        print(f"✅ Transaction {transaction_id} saved to database")
    except Exception as e:
        db.rollback()
        print(f"❌ Database error: {e}")
    finally:
        db.close()

# Get all transactions
@app.get("/transactions")
def get_transactions(limit: int = 10):
    """Получить список транзакций"""
    db = SessionLocal()
    try:
        transactions = db.query(Transaction).order_by(Transaction.sequence_number.desc()).limit(limit).all()
        return [
            {
                "transaction_id": tx.id,
                "sequence_number": tx.sequence_number,
                "created_at": tx.created_at.isoformat(),
                "overall_summary": tx.data["overall_summary"]
            }
            for tx in transactions
        ]
    finally:
        db.close()

# Get transaction details
@app.get("/transactions/{transaction_id}")
def get_transaction(transaction_id: str):
    """Получить детали транзакции"""
    db = SessionLocal()
    try:
        tx = db.query(Transaction).filter(Transaction.id == transaction_id).first()
        if not tx:
            raise HTTPException(404, "Transaction not found")
        return tx.data
    finally:
        db.close()

# Get transactions with alerts
@app.get("/transactions/alerts")
def get_transactions_with_alerts(limit: int = 10):
    """Получить транзакции с alerts"""
    db = SessionLocal()
    try:
        transactions = db.query(Transaction).order_by(Transaction.sequence_number.desc()).limit(limit).all()
        alerted = []
        for tx in transactions:
            images_with_issues = tx.data["overall_summary"]["images_with_issues"]
            if images_with_issues:
                alerted.append({
                    "transaction_id": tx.id,
                    "created_at": tx.created_at.isoformat(),
                    "images_with_issues": images_with_issues,
                    "average_match_percent": tx.data["overall_summary"]["average_match_percent"]
                })
        return alerted
    finally:
        db.close()

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