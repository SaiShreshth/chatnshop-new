# app/db_logger/logger.py
from datetime import datetime
from db_logger.db import SessionLocal
from db_logger.models import LLMLog

def log_llm_response(
    response_html: str,
    model: str,
    request_ts: datetime,
    response_ts: datetime,
    latency_ms: int,
    error: str | None = None
) -> None:
    """
    Save an LLM response log into the database.
    """
    db = SessionLocal()
    try:
        log_entry = LLMLog(
            response_html=response_html,
            model=model,
            request_ts=request_ts,
            response_ts=response_ts,
            latency_ms=latency_ms,
            error=error,
        )
        db.add(log_entry)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[DB-LOGGING] Failed to log: {e}")
    finally:
        db.close()

