"""
DICOM De-identification Utility
================================
Implements the HIPAA Safe Harbor method for removing Protected Health
Information (PHI) from DICOM files.  Follows DICOM PS3.15 Basic Profile.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import pydicom
from pydicom.dataset import Dataset

# ---------------------------------------------------------------------------
# The 18 HIPAA Safe-Harbor identifiers mapped to their DICOM tags.
# Tags not present in a particular file are silently skipped.
# ---------------------------------------------------------------------------
HIPAA_TAGS: list[tuple[int, int]] = [
    (0x0010, 0x0010),  # PatientName
    (0x0010, 0x0020),  # PatientID
    (0x0010, 0x0030),  # PatientBirthDate
    (0x0010, 0x0040),  # PatientSex
    (0x0010, 0x1000),  # OtherPatientIDs
    (0x0010, 0x1001),  # OtherPatientNames
    (0x0010, 0x1010),  # PatientAge
    (0x0010, 0x1020),  # PatientSize
    (0x0010, 0x1030),  # PatientWeight
    (0x0010, 0x1090),  # MedicalRecordLocator
    (0x0010, 0x2160),  # EthnicGroup
    (0x0010, 0x21B0),  # AdditionalPatientHistory
    (0x0010, 0x4000),  # PatientComments
    (0x0008, 0x0050),  # AccessionNumber
    (0x0008, 0x0080),  # InstitutionName
    (0x0008, 0x0081),  # InstitutionAddress
    (0x0008, 0x0090),  # ReferringPhysicianName
    (0x0008, 0x1070),  # OperatorsName
]


def extract_phi(ds: Dataset) -> dict[str, Any]:
    """Return a dict of all PHI values found in the DICOM dataset.

    The keys are human-readable tag names and the values are the raw
    element values (strings, dates, etc.).
    """
    phi: dict[str, Any] = {}
    tag_name_map = {
        (0x0010, 0x0010): "PatientName",
        (0x0010, 0x0020): "PatientID",
        (0x0010, 0x0030): "PatientBirthDate",
        (0x0010, 0x0040): "PatientSex",
        (0x0008, 0x0050): "AccessionNumber",
        (0x0008, 0x0080): "InstitutionName",
        (0x0008, 0x0081): "InstitutionAddress",
        (0x0008, 0x0090): "ReferringPhysicianName",
        (0x0008, 0x1070): "OperatorsName",
    }
    for tag in HIPAA_TAGS:
        if tag in ds:
            name = tag_name_map.get(tag, f"Tag{tag}")
            phi[name] = str(ds[tag].value)
    return phi


def generate_anonymous_id() -> str:
    """Generate a UUID-based anonymous patient identifier."""
    return str(uuid.uuid4())


def deidentify(ds: Dataset) -> tuple[Dataset, dict[str, Any], str]:
    """De-identify a DICOM dataset in place (Safe Harbor method).

    Parameters
    ----------
    ds : pydicom.Dataset
        The incoming DICOM dataset with PHI.

    Returns
    -------
    ds : pydicom.Dataset
        The same dataset with PHI tags wiped.
    phi : dict
        Extracted PHI values (for the audit log).
    anonymous_id : str
        The UUID assigned to this study.
    """
    # 1. Extract PHI before we wipe it
    phi = extract_phi(ds)

    # 2. Generate anonymous identifier
    anonymous_id = generate_anonymous_id()

    # 3. Wipe all HIPAA tags
    for tag in HIPAA_TAGS:
        if tag in ds:
            ds[tag].value = "ANONYMIZED"

    # 4. Stamp the anonymous ID into PatientID for traceability
    ds.PatientID = anonymous_id
    ds.PatientName = "ANONYMOUS"

    return ds, phi, anonymous_id


def read_dicom_bytes(file_bytes: bytes) -> Dataset:
    """Parse raw bytes into a pydicom Dataset."""
    from io import BytesIO
    return pydicom.dcmread(BytesIO(file_bytes))


def dicom_to_numpy(ds: Dataset) -> "numpy.ndarray":
    """Extract the pixel data from a DICOM dataset as a NumPy array.

    Returns a float32 array normalised to [0, 1].
    """
    import numpy as np

    pixel_array = ds.pixel_array.astype(np.float32)

    # Normalise to [0, 1]
    p_min, p_max = pixel_array.min(), pixel_array.max()
    if p_max - p_min > 0:
        pixel_array = (pixel_array - p_min) / (p_max - p_min)

    return pixel_array
