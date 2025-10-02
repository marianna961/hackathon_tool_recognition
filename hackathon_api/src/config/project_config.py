import os
from typing import List

class Settings:
    """Настройки проекта"""
    
    def __init__(self):
        # FastAPI настройки
        self.debug = os.getenv("DEBUG", "False").lower() == "true"
        self.allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
        
        # База данных
        self.database_url = os.getenv(
            "DATABASE_URL", 
            "postgresql://postgres:postgres@db:5432/postgres"
        )
        
        # MinIO настройки
        self.minio_url = os.getenv("MINIO_URL", "http://minio:9000")
        self.minio_key = os.getenv("MINIO_KEY", "minioadmin")
        self.minio_secret = os.getenv("MINIO_SECRET", "minioadmin")
        self.minio_bucket = os.getenv("MINIO_BUCKET", "evidence")
        
        # ML настройки
        # self.model_version = os.getenv("MODEL_VERSION", "yolov8_stub_v1")
        # self.min_frames = int(os.getenv("MIN_FRAMES", "5"))
        
        # Пороги
        self.match_threshold = float(os.getenv("MATCH_THRESHOLD", "95.0"))

def get_settings() -> Settings:
    """Возвращает настройки проекта"""
    return Settings()