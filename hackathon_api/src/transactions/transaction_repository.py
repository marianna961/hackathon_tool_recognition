# src/transactions/transaction_repository.py
import json
from pathlib import Path
from sqlalchemy.exc import SQLAlchemyError
from src.config.project_config import settings
from src.config.database.db_helper import get_session
from src.transactions.transaction_model import TransactionDB

DATA_DIR = Path(settings.DATA_DIR)
TX_DIR = DATA_DIR / "transactions"
TX_DIR.mkdir(parents=True, exist_ok=True)

def save_transaction_to_file(transaction: dict) -> str:
    tx_id = transaction.get("transaction_id")
    path = TX_DIR / f"{tx_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(transaction, f, ensure_ascii=False, indent=2)
    return str(path.resolve())

def save_transaction_to_db(transaction: dict) -> None:
    try:
        with get_session() as db:
            # prevent duplicates
            from sqlalchemy import select
            stmt = select(TransactionDB).where(TransactionDB.tx_id == transaction["transaction_id"])
            existing = db.execute(stmt).scalar_one_or_none()
            if existing:
                return
            db_obj = TransactionDB(
                tx_id=transaction["transaction_id"],
                payload=transaction
            )
            db.add(db_obj)
            db.commit()
    except SQLAlchemyError as e:
        print(f"[DB ERROR] {e}")
