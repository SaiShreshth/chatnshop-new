import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()

# Prefer DSN over URL (more flexible, works with PostgreSQL, MySQL, etc.)
DB_DSN = os.getenv(
    "DB_DSN",
    "postgresql+psycopg2://postgres:password@localhost:5432/chatnshop"
)

# Create the engine using DSN
engine = create_engine(DB_DSN, pool_pre_ping=True)

# Standard session and base setup
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def get_db():
    """Dependency-style generator for DB sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
