# app/db_logger/models.py
from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.sql import func
from db_logger.db import Base

class LLMLog(Base):
    __tablename__ = "llm_logs"

    id = Column(Integer, primary_key=True, index=True)
    response_html = Column(Text, nullable=False)
    model = Column(String(100), nullable=False)
    request_ts = Column(DateTime(timezone=True), server_default=func.now())
    response_ts = Column(DateTime(timezone=True), nullable=False)
    latency_ms = Column(Integer, nullable=False)
    error = Column(Text, nullable=True)
