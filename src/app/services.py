"""
Business Logic — Services Layer
=================================
Database ORM models, audit logging, and inference persistence.
Used by the Streamlit dashboard to store and query records.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base

from src.app.schemas import InferenceResult, FHIRDiagnosticReport

# ---------------------------------------------------------------------------
# Database setup (PostgreSQL via SQLAlchemy)
# ---------------------------------------------------------------------------
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://user:password@localhost:5433/medview",
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


class AuditLog(Base):
    """Audit log table — maps anonymous IDs back to original PHI."""
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    anonymous_id = Column(String(64), nullable=False, index=True)
    original_patient_name = Column(String(256))
    original_patient_id = Column(String(128))
    institution = Column(String(256))
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    action = Column(String(64), default="de-identification")


class InferenceRecord(Base):
    """Persists every inference result for downstream analytics."""
    __tablename__ = "inference_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    anonymous_id = Column(String(64), nullable=False, index=True)
    top_finding = Column(String(128))
    top_confidence = Column(String(16))
    raw_scores = Column(Text)  # JSON string
    dicom_url = Column(Text)
    heatmap_url = Column(Text)
    fhir_report = Column(Text)  # JSON string
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def init_db() -> None:
    """Create tables if they don't exist."""
    try:
        Base.metadata.create_all(bind=engine)
        print("[✓] Database tables initialised")
    except Exception as exc:
        print(f"[⚠] Database init skipped (will retry on first request): {exc}")


# ---------------------------------------------------------------------------
# Database helpers (soft-fail so app works without Postgres during dev)
# ---------------------------------------------------------------------------

def _save_audit_log(anonymous_id: str, phi: dict[str, Any]) -> None:
    session = None
    try:
        session = SessionLocal()
        entry = AuditLog(
            anonymous_id=anonymous_id,
            original_patient_name=phi.get("PatientName"),
            original_patient_id=phi.get("PatientID"),
            institution=phi.get("InstitutionName"),
        )
        session.add(entry)
        session.commit()
    except Exception as exc:
        if session:
            session.rollback()
        print(f"[⚠] Audit log write failed: {exc}")
    finally:
        if session:
            session.close()


def _save_inference_record(
    anonymous_id: str,
    inference: InferenceResult,
    dicom_url: str,
    heatmap_url: str,
    fhir_report: FHIRDiagnosticReport,
) -> None:
    session = None
    try:
        session = SessionLocal()
        record = InferenceRecord(
            anonymous_id=anonymous_id,
            top_finding=inference.top_finding,
            top_confidence=str(inference.top_confidence),
            raw_scores=json.dumps(inference.raw_scores),
            dicom_url=dicom_url,
            heatmap_url=heatmap_url,
            fhir_report=fhir_report.model_dump_json(),
        )
        session.add(record)
        session.commit()
    except Exception as exc:
        if session:
            session.rollback()
        print(f"[⚠] Inference record write failed: {exc}")
    finally:
        if session:
            session.close()
