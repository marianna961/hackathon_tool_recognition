from contextlib import contextmanager
from src.config.database.db_config import SessionLocal

@contextmanager
def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
