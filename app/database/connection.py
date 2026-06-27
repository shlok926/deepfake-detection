from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config.config import settings

# Determine database engine parameters based on dialect
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    # SQLite-specific optimization for multi-threaded FastAPI calls
    connect_args = {"check_same_thread": False}

# Create connection pool
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    connect_args=connect_args
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Declarative base model
Base = declarative_base()

def get_db() -> Generator:
    """
    FastAPI dependency that provides a transactional database session.
    Closes the connection automatically after response completion.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
