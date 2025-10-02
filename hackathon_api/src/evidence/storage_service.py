import os
from pathlib import Path
from typing import Optional
import boto3
from botocore.exceptions import BotoCoreError, NoCredentialsError
from src.config.project_config import settings

# Локальная папка для fallback
DATA_DIR = Path(settings.DATA_DIR)
EVIDENCE_DIR = DATA_DIR / "evidence"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

# Настройка клиента MinIO (через boto3)
def get_s3_client():
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=settings.MINIO_URL,
            aws_access_key_id=settings.MINIO_KEY,
            aws_secret_access_key=settings.MINIO_SECRET,
            region_name="us-east-1",
        )
        return s3
    except Exception as e:
        print(f"[MinIO ERROR] client init failed: {e}")
        return None

s3_client = get_s3_client()

def ensure_bucket():
    """
    Проверяем/создаём bucket в MinIO.
    """
    if not s3_client:
        return False
    try:
        s3_client.head_bucket(Bucket=settings.MINIO_BUCKET)
        return True
    except Exception:
        try:
            s3_client.create_bucket(Bucket=settings.MINIO_BUCKET)
            return True
        except Exception as e:
            print(f"[MinIO ERROR] bucket create failed: {e}")
            return False

ensure_bucket()

def save_local(tx_id: str, filename: str, content: bytes) -> str:
    """
    Сохраняем файл локально (fallback).
    """
    folder = EVIDENCE_DIR / tx_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    path.write_bytes(content)
    return str(path.resolve())

def save_to_minio(tx_id: str, filename: str, content: bytes, content_type: str = "application/octet-stream") -> Optional[str]:
    """
    Сохраняем файл в MinIO и возвращаем presigned URL.
    """
    if not s3_client:
        return None
    key = f"{tx_id}/{filename}"
    try:
        s3_client.put_object(
            Bucket=settings.MINIO_BUCKET,
            Key=key,
            Body=content,
            ContentType=content_type
        )
        return get_presigned_url(key)
    except (BotoCoreError, NoCredentialsError, Exception) as e:
        print(f"[MinIO ERROR] save failed: {e}")
        return None

def get_presigned_url(key: str, expires: int = 3600) -> str:
    """
    Генерация presigned URL для доступа к объекту.
    """
    if not s3_client:
        raise RuntimeError("MinIO client not initialized")
    try:
        return s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': settings.MINIO_BUCKET, 'Key': key},
            ExpiresIn=expires
        )
    except Exception as e:
        raise Exception(f"Ошибка генерации presigned URL: {str(e)}")

def store_evidence(tx_id: str, filename: str, content: bytes, content_type: Optional[str] = None) -> str:
    """
    Универсальный метод — пробуем сохранить в MinIO, если не получилось — локально.
    """
    # MIME-тип подбираем автоматически
    if not content_type:
        if filename.lower().endswith((".jpg", ".jpeg")):
            content_type = "image/jpeg"
        elif filename.lower().endswith(".png"):
            content_type = "image/png"
        elif filename.lower().endswith(".mp4"):
            content_type = "video/mp4"
        else:
            content_type = "application/octet-stream"

    url = save_to_minio(tx_id, filename, content, content_type)
    if url:
        return url
    # fallback
    return save_local(tx_id, filename, content)
