"""
Pydantic Schemas — Data Models
===============================
Defines data contracts for the MedView-AI platform including the
FHIR R4 DiagnosticReport structure.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


# ===================================================================
# Internal models
# ===================================================================

class PredictionItem(BaseModel):
    """A single class prediction with confidence."""
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class InferenceResult(BaseModel):
    """Full inference output from the ensemble model."""
    predictions: list[PredictionItem] = []
    raw_scores: list[float] = []
    top_finding: str = ""
    top_confidence: float = 0.0


# ===================================================================
# FHIR R4 — DiagnosticReport (simplified)
# ===================================================================

class FHIRCoding(BaseModel):
    system: str = "http://loinc.org"
    code: str = "58718-8"
    display: str = "Automated analysis of Chest X-ray"


class FHIRCode(BaseModel):
    coding: list[FHIRCoding] = [FHIRCoding()]


class FHIRDiagnosticReport(BaseModel):
    """FHIR R4 DiagnosticReport resource."""
    resourceType: str = "DiagnosticReport"
    id: str | None = None
    status: str = "final"
    code: FHIRCode = FHIRCode()
    issued: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    conclusion: str = ""


