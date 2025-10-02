# src/transactions/transaction_controller.py
from fastapi import APIRouter, UploadFile, File, Form
from typing import Optional
from src.transactions.transaction_service import process_image_transaction, process_batch_archive

router = APIRouter()

@router.post("/predict")
async def predict(
    image: UploadFile = File(...),
    event_type: str = Form("hand_out"),
    camera_id: str = Form("table_cam_1"),
    operator_id: Optional[str] = Form(None),
    expected_list_json: Optional[str] = Form(None)
):
    return await process_image_transaction(image, event_type, camera_id, operator_id, expected_list_json)

@router.post("/batch_predict")
async def batch_predict(
    archive: UploadFile = File(...),
    event_type: str = Form("hand_out"),
    camera_id: str = Form("table_cam_1"),
    operator_id: Optional[str] = Form(None),
    expected_list_json: Optional[str] = Form(None)
):
    return await process_batch_archive(archive, event_type, camera_id, operator_id, expected_list_json)
