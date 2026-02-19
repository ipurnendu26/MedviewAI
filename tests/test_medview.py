"""
Tests for MedView-AI
=====================
Covers DICOM de-identification, schema validation, inference, and
the Streamlit pipeline core functions.
"""

from __future__ import annotations

import io
import uuid

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian
import pytest

# ---------------------------------------------------------------------------
# Helpers — create a minimal synthetic DICOM
# ---------------------------------------------------------------------------

def _make_synthetic_dicom() -> bytes:
    """Create a small valid DICOM file (64x64 greyscale) in memory."""
    filename = "synthetic.dcm"
    file_meta = pydicom.Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\x00" * 128)

    # Patient PHI (should be wiped by de-identification)
    ds.PatientName = "Doe^John"
    ds.PatientID = "PAT-001"
    ds.PatientBirthDate = "19800101"
    ds.PatientSex = "M"
    ds.InstitutionName = "Test Hospital"
    ds.ReferringPhysicianName = "Dr. Smith"
    ds.AccessionNumber = "ACC-123"

    # Image metadata
    ds.Rows = 64
    ds.Columns = 64
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"

    # Pixel data (random noise)
    pixel_array = np.random.randint(0, 4096, (64, 64), dtype=np.uint16)
    ds.PixelData = pixel_array.tobytes()

    # Serialise
    buf = io.BytesIO()
    ds.save_as(buf)
    buf.seek(0)
    return buf.read()


# ===================================================================
# DICOM de-identification tests
# ===================================================================

class TestDicomDeidentification:
    def test_phi_is_extracted(self):
        from src.utils.dicom_utils import read_dicom_bytes, extract_phi

        ds = read_dicom_bytes(_make_synthetic_dicom())
        phi = extract_phi(ds)

        assert "PatientName" in phi
        assert phi["PatientName"] == "Doe^John"
        assert phi["PatientID"] == "PAT-001"
        assert phi["InstitutionName"] == "Test Hospital"

    def test_deidentify_wipes_phi(self):
        from src.utils.dicom_utils import read_dicom_bytes, deidentify

        ds = read_dicom_bytes(_make_synthetic_dicom())
        ds, phi, anon_id = deidentify(ds)

        assert ds.PatientName == "ANONYMOUS"
        assert ds.PatientID == anon_id
        assert len(anon_id) == 36  # UUID format

    def test_pixel_data_survives(self):
        from src.utils.dicom_utils import read_dicom_bytes, deidentify, dicom_to_numpy

        ds = read_dicom_bytes(_make_synthetic_dicom())
        ds, _, _ = deidentify(ds)

        arr = dicom_to_numpy(ds)
        assert arr.shape == (64, 64)
        assert arr.dtype == np.float32
        assert 0.0 <= arr.min() <= arr.max() <= 1.0


# ===================================================================
# Schema validation tests
# ===================================================================

class TestSchemas:
    def test_fhir_report_structure(self):
        from src.app.schemas import FHIRDiagnosticReport

        report = FHIRDiagnosticReport(
            id="test-123",
            conclusion="Pneumonia: 98.2%",
        )
        data = report.model_dump()

        assert data["resourceType"] == "DiagnosticReport"
        assert data["status"] == "final"
        assert data["code"]["coding"][0]["system"] == "http://loinc.org"


# ===================================================================
# Inference smoke test
# ===================================================================

from src.ml.inference import get_ensemble, CLASS_NAMES


class TestInference:
    def test_ensemble_returns_valid_output(self):
        from src.ml.inference import get_ensemble

        ensemble = get_ensemble()
        dummy_image = np.random.rand(64, 64).astype(np.float32)
        result = ensemble.predict(dummy_image)

        assert "predictions" in result
        assert "raw_scores" in result
        assert "top_finding" in result
        assert len(result["raw_scores"]) == len(CLASS_NAMES)




# ===================================================================
# Grad-CAM / explainability tests
# ===================================================================

class TestExplainability:
    def test_heatmap_png_bytes_returned(self):
        """generate_heatmap_png should return valid PNG bytes."""
        from src.ml.explain import generate_heatmap_png

        dummy_image = np.random.rand(64, 64).astype(np.float32)
        png_bytes = generate_heatmap_png(
            original_image=dummy_image,
            raw_scores=[0.3, 0.7],
            class_index=1,
        )
        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 100
        # PNG magic header
        assert png_bytes[:4] == b"\x89PNG"

    def test_saliency_map_shape(self):
        """Saliency map should match input spatial dims."""
        from src.ml.explain import saliency_map_approx

        img = np.random.rand(128, 128).astype(np.float32)
        heatmap = saliency_map_approx([0.4, 0.6], img, 1)
        assert heatmap.shape == (128, 128)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0


# ===================================================================
# Streamlit pipeline integration test
# ===================================================================

class TestStreamlitPipeline:
    def test_full_pipeline_no_minio(self):
        """End-to-end: DICOM → de-id → inference → heatmap, no MinIO."""
        from src.utils.dicom_utils import read_dicom_bytes, deidentify, dicom_to_numpy
        from src.ml.inference import get_ensemble, CLASS_NAMES
        from src.ml.explain import generate_heatmap_png

        dicom_bytes = _make_synthetic_dicom()
        ds = read_dicom_bytes(dicom_bytes)
        ds, phi, anon_id = deidentify(ds)

        assert ds.PatientName == "ANONYMOUS"

        pixel_array = dicom_to_numpy(ds)
        ensemble = get_ensemble()
        result = ensemble.predict(pixel_array)

        assert "top_finding" in result
        assert result["top_finding"] in CLASS_NAMES

        top_idx = CLASS_NAMES.index(result["top_finding"])
        heatmap_bytes = generate_heatmap_png(
            original_image=pixel_array,
            raw_scores=result["raw_scores"],
            class_index=top_idx,
        )
        assert isinstance(heatmap_bytes, bytes)
        assert heatmap_bytes[:4] == b"\x89PNG"
