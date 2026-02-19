"""
MedView-AI — Production Streamlit Dashboard
=============================================
Upload a DICOM chest X-ray (.dcm), view the de-identified image,
ensemble prediction (NORMAL / PNEUMONIA), and Grad-CAM heatmap —
all rendered directly in the browser.

Usage
-----
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ── Project imports ──────────────────────────────────────────────
from src.utils.dicom_utils import (
    read_dicom_bytes,
    deidentify,
    extract_phi,
    dicom_to_numpy,
)
from src.ml.inference import get_ensemble, CLASS_NAMES
from src.ml.explain import (
    generate_heatmap_png,
    overlay_heatmap,
    saliency_map_approx,
)

# DB layer (soft-fail if Postgres is not running)
try:
    from src.app.services import (
        init_db,
        _save_audit_log,
        _save_inference_record,
        SessionLocal,
        AuditLog,
        InferenceRecord as InferenceRecordORM,
    )
    from src.app.schemas import InferenceResult, PredictionItem, FHIRDiagnosticReport

    _DB_AVAILABLE = True
except Exception:
    _DB_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────
APP_VERSION = "1.0.0"
_LABEL_TOP_FINDING = "Top Finding"
_LABEL_ANON_ID = "Anonymous ID"

# ─────────────────────────────────────────────────────────────────
# Page config — MUST be the first Streamlit command
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedView-AI  |  Chest X-Ray Analysis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# Custom CSS — production-grade theming
# ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Import Google Font ───────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Root variables ───────────────────────────────────────── */
    :root {
        --primary: #0066FF;
        --primary-light: #E8F0FE;
        --success: #00C48C;
        --danger: #FF4757;
        --warning: #FFA502;
        --surface: #FFFFFF;
        --surface-alt: #F8FAFC;
        --border: #E2E8F0;
        --text-primary: #1A202C;
        --text-secondary: #64748B;
        --text-muted: #94A3B8;
        --radius: 12px;
        --shadow-sm: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
        --shadow-md: 0 4px 12px rgba(0,0,0,0.08);
        --shadow-lg: 0 10px 30px rgba(0,0,0,0.10);
    }

    /* ── Global typography ────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* ── Main container ───────────────────────────────────────── */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1400px !important;
    }

    /* ── Sidebar styling ──────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%) !important;
        border-right: 1px solid #334155 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #E2E8F0 !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h1 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #334155 !important;
    }
    section[data-testid="stSidebar"] .stSlider > div > div > div {
        color: #94A3B8 !important;
    }

    /* ── Tab styling ──────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background: var(--surface-alt);
        border-radius: var(--radius);
        padding: 4px;
        border: 1px solid var(--border);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 500;
        font-size: 0.9rem;
        color: var(--text-secondary);
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        background: var(--surface) !important;
        color: var(--primary) !important;
        box-shadow: var(--shadow-sm) !important;
        font-weight: 600;
    }

    /* ── Metric cards ─────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 16px 20px;
        box-shadow: var(--shadow-sm);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: var(--text-secondary) !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
    }

    /* ── DataFrame styling ────────────────────────────────────── */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        overflow: hidden;
    }

    /* ── Button styling ───────────────────────────────────────── */
    .stDownloadButton > button,
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        padding: 8px 20px !important;
        border: 1px solid var(--border) !important;
        transition: all 0.15s ease !important;
    }
    .stDownloadButton > button:hover,
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: var(--shadow-sm) !important;
    }

    /* ── File uploader ────────────────────────────────────────── */
    [data-testid="stFileUploader"] {
        border: 2px dashed var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 2rem !important;
        background: var(--surface-alt) !important;
        transition: border-color 0.2s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--primary) !important;
    }

    /* ── Alert boxes ──────────────────────────────────────────── */
    .stAlert {
        border-radius: var(--radius) !important;
    }

    /* ── Expander ─────────────────────────────────────────────── */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }

    /* ── JSON viewer ──────────────────────────────────────────── */
    [data-testid="stJson"] {
        background: #F8FAFC !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 1rem !important;
    }

    /* ── Custom badge helpers ─────────────────────────────────── */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .badge-ok   { background: #DCFCE7; color: #166534; }
    .badge-warn { background: #FEF3C7; color: #92400E; }
    .badge-err  { background: #FEE2E2; color: #991B1B; }

    /* ── Section header helper ────────────────────────────────── */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.4rem;
    }
    .section-caption {
        font-size: 0.82rem;
        color: var(--text-secondary);
        margin-bottom: 1.2rem;
        line-height: 1.5;
    }

    /* ── Card container ───────────────────────────────────────── */
    .card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.5rem;
        box-shadow: var(--shadow-sm);
    }

    /* ── Prediction result banner ─────────────────────────────── */
    .result-banner {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 20px 28px;
        border-radius: var(--radius);
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: var(--shadow-sm);
        margin-bottom: 0.8rem;
    }
    .result-pneumonia {
        background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
        border: 1px solid #FECACA;
        color: #991B1B;
    }
    .result-normal {
        background: linear-gradient(135deg, #DCFCE7 0%, #BBF7D0 100%);
        border: 1px solid #BBF7D0;
        color: #166534;
    }
    .result-icon { font-size: 2rem; }
    .result-label { font-size: 1.25rem; font-weight: 700; }
    .result-conf { font-size: 1rem; font-weight: 400; opacity: 0.85; }

    /* ── Empty state ──────────────────────────────────────────── */
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
    }
    .empty-state-icon { font-size: 4rem; margin-bottom: 1rem; }
    .empty-state-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }
    .empty-state-desc {
        font-size: 0.9rem;
        color: var(--text-secondary);
        max-width: 500px;
        margin: 0 auto 1.5rem auto;
        line-height: 1.6;
    }

    /* ── Footer ───────────────────────────────────────────────── */
    .app-footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid var(--border);
        margin-top: 3rem;
        font-size: 0.78rem;
        color: var(--text-muted);
    }
    .app-footer a { color: var(--primary); text-decoration: none; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────
# Cached resource loaders
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading EfficientNet-B4 + ViT models …")
def load_models():
    """Load the ensemble once and cache across reruns."""
    ensemble = get_ensemble()
    if not ensemble._has_any_model():
        st.warning(
            "No trained models found in `models/`. "
            "Run `python -m src.ml.train --data_dir ./data/chest_xray` first."
        )
    return ensemble


@st.cache_resource(show_spinner=False)
def maybe_init_db():
    if _DB_AVAILABLE:
        try:
            init_db()
            return True
        except Exception:
            return False
    return False


# ─────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── Branding ────────────────────────────────────────────────
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">
            <img src="https://img.icons8.com/fluency/96/lungs.png" width="42"/>
            <div>
                <div style="font-size:1.4rem;font-weight:700;letter-spacing:-0.02em;color:#fff !important;">MedView-AI</div>
                <div style="font-size:0.72rem;color:#94A3B8 !important;letter-spacing:0.05em;text-transform:uppercase;">Chest X-Ray Intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Model info ──────────────────────────────────────────────
    st.markdown(
        """
        <div style="font-size:0.78rem;line-height:1.8;color:#CBD5E1 !important;">
            <b style="color:#E2E8F0 !important;">Ensemble</b><br/>
            &nbsp;&nbsp;EfficientNet-B4 &nbsp;·&nbsp; Hybrid ViT<br/>
            <b style="color:#E2E8F0 !important;">Classes</b><br/>
            &nbsp;&nbsp;NORMAL &nbsp;·&nbsp; PNEUMONIA<br/>
            <b style="color:#E2E8F0 !important;">Explainability</b><br/>
            &nbsp;&nbsp;Gradient-weighted Class Activation Map
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Settings ────────────────────────────────────────────────
    st.markdown(
        '<div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;'
        'letter-spacing:0.1em;color:#94A3B8 !important;margin-bottom:8px;">Settings</div>',
        unsafe_allow_html=True,
    )
    show_phi = st.checkbox("Show extracted PHI (before de-id)", value=False)
    show_dicom_meta = st.checkbox("Show DICOM metadata", value=False)
    overlay_alpha = st.slider("Heatmap overlay opacity", 0.1, 0.9, 0.4, 0.05)

    st.divider()

    # ── System Status ───────────────────────────────────────────
    ensemble = load_models()
    db_ok = maybe_init_db()
    models_ok = ensemble._has_any_model()

    st.markdown(
        '<div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;'
        'letter-spacing:0.1em;color:#94A3B8 !important;margin-bottom:8px;">System Status</div>',
        unsafe_allow_html=True,
    )
    _badge = lambda ok, label: (
        f'<span class="status-badge {"badge-ok" if ok else "badge-err"}">'
        f'{"●" if ok else "○"} {label}</span>'
    )
    st.markdown(_badge(models_ok, "ML Models"), unsafe_allow_html=True)
    st.markdown(_badge(db_ok, "PostgreSQL"), unsafe_allow_html=True)

    # ── Version footer ──────────────────────────────────────────
    st.markdown("<br/>" * 2, unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:0.68rem;color:#475569 !important;text-align:center;">'
        f'v{APP_VERSION} &nbsp;·&nbsp; Python {__import__("sys").version.split()[0]}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────
# Main area — Tabbed layout
# ─────────────────────────────────────────────────────────────────
tab_analysis, tab_audit = st.tabs(["🫁  Analysis", "📋  Audit Logs"])


# =================================================================
# TAB 1 — Analysis
# =================================================================
with tab_analysis:
    st.markdown(
        '<div class="section-header">Chest X-Ray Analysis</div>'
        '<div class="section-caption">'
        'Upload a DICOM (<code>.dcm</code>) file to receive an ensemble AI prediction '
        'with Grad-CAM visual explainability.</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose a .dcm file",
        type=["dcm"],
        help="Upload a DICOM chest X-ray image",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()

        if len(file_bytes) == 0:
            st.error("Uploaded file is empty.")
            st.stop()

        # ── 1. Parse DICOM ──────────────────────────────────────
        try:
            ds = read_dicom_bytes(file_bytes)
        except Exception as exc:
            st.error(f"Failed to parse DICOM file: {exc}")
            st.stop()

        # ── 2. Extract PHI & de-identify ────────────────────────
        phi = extract_phi(ds)
        ds, _, anonymous_id = deidentify(ds)

        # ── 3. Extract pixel array ──────────────────────────────
        try:
            pixel_array = dicom_to_numpy(ds)
        except Exception as exc:
            st.error(f"Could not extract pixel data: {exc}")
            st.stop()

        # ── 4. Run inference ────────────────────────────────────
        with st.spinner("Running ensemble inference …"):
            result = ensemble.predict(pixel_array)

        top_finding = result["top_finding"]
        top_confidence = result["top_confidence"]
        raw_scores = result["raw_scores"]

        # ── 5. Generate Grad-CAM heatmap ────────────────────────
        with st.spinner("Generating Grad-CAM heatmap …"):
            top_class_idx = CLASS_NAMES.index(top_finding)
            heatmap_png_bytes = generate_heatmap_png(
                original_image=pixel_array,
                raw_scores=raw_scores,
                class_index=top_class_idx,
            )
            heatmap_raw = saliency_map_approx(raw_scores, pixel_array, top_class_idx)
            overlay_img = overlay_heatmap(pixel_array, heatmap_raw, alpha=overlay_alpha)

        # ── 6. Audit log (soft-fail) ────────────────────────────
        if _DB_AVAILABLE and db_ok:
            try:
                _save_audit_log(anonymous_id, phi)
                inference_obj = InferenceResult(
                    predictions=[PredictionItem(**p) for p in result["predictions"]],
                    raw_scores=raw_scores,
                    top_finding=top_finding,
                    top_confidence=top_confidence,
                )
                fhir_report = FHIRDiagnosticReport(
                    id=anonymous_id,
                    conclusion=f"Findings — {top_finding}: {top_confidence*100:.1f}%",
                )
                _save_inference_record(anonymous_id, inference_obj, "", "", fhir_report)
            except Exception:
                pass  # non-critical

        # ─────────────────────────────────────────────────────────
        # DISPLAY RESULTS
        # ─────────────────────────────────────────────────────────

        # ── Prediction banner ───────────────────────────────────
        if top_finding == "PNEUMONIA":
            st.markdown(
                f'<div class="result-banner result-pneumonia">'
                f'<span class="result-icon">⚠️</span>'
                f'<div><span class="result-label">PNEUMONIA Detected</span><br/>'
                f'<span class="result-conf">Confidence: {top_confidence*100:.1f}%</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="result-banner result-normal">'
                f'<span class="result-icon">✅</span>'
                f'<div><span class="result-label">NORMAL</span><br/>'
                f'<span class="result-conf">Confidence: {top_confidence*100:.1f}%</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.write("")  # spacer

        # ── Metrics row ─────────────────────────────────────────
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("NORMAL", f"{raw_scores[0]*100:.1f}%")
        col_m2.metric("PNEUMONIA", f"{raw_scores[1]*100:.1f}%")
        col_m3.metric(_LABEL_TOP_FINDING, top_finding)
        col_m4.metric(_LABEL_ANON_ID, anonymous_id[:8] + "…")

        st.write("")  # spacer

        # ── Images: Original | Heatmap overlay ──────────────────
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown(
                '<div class="section-header">Original X-Ray</div>',
                unsafe_allow_html=True,
            )
            if pixel_array.dtype != np.uint8:
                display_img = np.uint8(255 * np.clip(pixel_array, 0, 1))
            else:
                display_img = pixel_array
            if display_img.ndim == 2:
                display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2RGB)
            st.image(display_img, use_container_width=True)

        with col2:
            st.markdown(
                '<div class="section-header">Grad-CAM Heatmap</div>',
                unsafe_allow_html=True,
            )
            overlay_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
            st.image(overlay_rgb, use_container_width=True)

        # ── Detailed Grad-CAM (matplotlib figure) ──────────────
        with st.expander("🔬 Full Grad-CAM Figure (matplotlib)", expanded=False):
            heatmap_pil = Image.open(io.BytesIO(heatmap_png_bytes))
            st.image(heatmap_pil, use_container_width=True)

        # ── Class-by-class scores ──────────────────────────────
        st.write("")
        st.markdown(
            '<div class="section-header">Class Scores</div>',
            unsafe_allow_html=True,
        )

        score_col1, score_col2 = st.columns([1, 1], gap="large")
        with score_col1:
            scores_df = pd.DataFrame({
                "Class": CLASS_NAMES,
                "Confidence": [f"{s*100:.2f}%" for s in raw_scores],
                "Score": raw_scores,
            })
            st.dataframe(scores_df, use_container_width=True, hide_index=True)

        with score_col2:
            chart_df = pd.DataFrame({"Class": CLASS_NAMES, "Score": raw_scores})
            st.bar_chart(chart_df.set_index("Class"), height=220, color=["#0066FF"])

        # ── FHIR DiagnosticReport ──────────────────────────────
        st.write("")
        st.markdown(
            '<div class="section-header">FHIR R4 DiagnosticReport</div>'
            '<div class="section-caption">'
            'HL7 FHIR-compliant diagnostic report generated automatically.</div>',
            unsafe_allow_html=True,
        )
        fhir_data = {
            "resourceType": "DiagnosticReport",
            "id": anonymous_id,
            "status": "final",
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": "58718-8",
                    "display": "Automated analysis of Chest X-ray",
                }]
            },
            "issued": datetime.now(timezone.utc).isoformat(),
            "conclusion": f"Findings — {top_finding}: {top_confidence*100:.1f}%",
        }
        st.json(fhir_data)

        # ── Optional: PHI display ──────────────────────────────
        if show_phi and phi:
            st.write("")
            st.markdown(
                '<div class="section-header">🔒 Extracted PHI (before de-identification)</div>',
                unsafe_allow_html=True,
            )
            st.warning(
                "This data was extracted and then wiped from the DICOM. "
                "Shown for demo only."
            )
            st.json(phi)

        # ── Optional: DICOM metadata ───────────────────────────
        if show_dicom_meta:
            st.write("")
            st.markdown(
                '<div class="section-header">📋 DICOM Metadata (after de-identification)</div>',
                unsafe_allow_html=True,
            )
            meta_dict = {}
            for elem in ds:
                if elem.keyword and elem.keyword != "PixelData":
                    meta_dict[elem.keyword] = str(elem.value)
            st.json(meta_dict)

        # ── Download buttons ───────────────────────────────────
        st.write("")
        st.markdown(
            '<div class="section-header">Downloads</div>'
            '<div class="section-caption">'
            'Export analysis artifacts for your records.</div>',
            unsafe_allow_html=True,
        )
        dl_col1, dl_col2, dl_col3 = st.columns(3)
        with dl_col1:
            st.download_button(
                "⬇  Heatmap (PNG)",
                data=heatmap_png_bytes,
                file_name=f"heatmap_{anonymous_id}.png",
                mime="image/png",
                use_container_width=True,
            )
        with dl_col2:
            st.download_button(
                "⬇  FHIR Report (JSON)",
                data=json.dumps(fhir_data, indent=2),
                file_name=f"fhir_report_{anonymous_id}.json",
                mime="application/json",
                use_container_width=True,
            )
        with dl_col3:
            st.download_button(
                "⬇  Scores (JSON)",
                data=json.dumps(result, indent=2),
                file_name=f"scores_{anonymous_id}.json",
                mime="application/json",
                use_container_width=True,
            )

    else:
        # ── Empty state ─────────────────────────────────────────
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-state-icon">🫁</div>
                <div class="empty-state-title">No DICOM file uploaded</div>
                <div class="empty-state-desc">
                    Drag &amp; drop or browse for a <strong>.dcm</strong> chest X-ray
                    file above to begin the automated analysis pipeline.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("ℹ️  How it works", expanded=True):
            st.markdown("""
| Step | Description |
|------|-------------|
| **1. Upload** | Select a DICOM chest X-ray (`.dcm` file) |
| **2. De-identification** | All 18 HIPAA Safe Harbor PHI tags are wiped |
| **3. Ensemble Inference** | EfficientNet-B4 + Hybrid ViT predict NORMAL vs PNEUMONIA |
| **4. Grad-CAM Heatmap** | Highlights which image regions drove the prediction |
| **5. Results** | View the prediction, heatmap, scores, and FHIR report |
            """)


# =================================================================
# TAB 2 — Audit Logs
# =================================================================
with tab_audit:
    st.markdown(
        '<div class="section-header">PostgreSQL Audit Logs</div>'
        '<div class="section-caption">'
        'Browse de-identification records and inference history stored in PostgreSQL. '
        'Use filters to narrow down results, or export data as CSV.</div>',
        unsafe_allow_html=True,
    )

    if not _DB_AVAILABLE or not db_ok:
        st.warning(
            "PostgreSQL is not available. Start the database with "
            "`docker compose up db -d` and reload the page."
        )
    else:
        # ── Refresh ─────────────────────────────────────────────
        ref_col1, ref_col2 = st.columns([6, 1])
        with ref_col2:
            if st.button("🔄 Refresh", key="refresh_logs", use_container_width=True):
                st.rerun()

        # ── Query helpers ───────────────────────────────────────
        def _fetch_audit_logs() -> pd.DataFrame:
            """Return all audit_log rows as a DataFrame."""
            session = SessionLocal()
            try:
                rows = (
                    session.query(AuditLog)
                    .order_by(AuditLog.timestamp.desc())
                    .all()
                )
                data = [
                    {
                        "ID": r.id,
                        _LABEL_ANON_ID: r.anonymous_id,
                        "Patient Name": r.original_patient_name or "—",
                        "Patient ID": r.original_patient_id or "—",
                        "Institution": r.institution or "—",
                        "Action": r.action,
                        "Timestamp": r.timestamp,
                    }
                    for r in rows
                ]
                return pd.DataFrame(data) if data else pd.DataFrame()
            finally:
                session.close()

        def _fetch_inference_logs() -> pd.DataFrame:
            """Return all inference_results rows as a DataFrame."""
            session = SessionLocal()
            try:
                rows = (
                    session.query(InferenceRecordORM)
                    .order_by(InferenceRecordORM.timestamp.desc())
                    .all()
                )
                data = [
                    {
                        "ID": r.id,
                        _LABEL_ANON_ID: r.anonymous_id,
                        _LABEL_TOP_FINDING: r.top_finding or "—",
                        "Confidence": r.top_confidence or "—",
                        "Raw Scores": r.raw_scores or "—",
                        "FHIR Report": r.fhir_report or "—",
                        "Timestamp": r.timestamp,
                    }
                    for r in rows
                ]
                return pd.DataFrame(data) if data else pd.DataFrame()
            finally:
                session.close()

        # ── Sub-tabs ────────────────────────────────────────────
        audit_sub, inference_sub = st.tabs(
            ["🔒  De-identification Audit Log", "🧠  Inference Results"]
        )

        # ── Audit Log table ─────────────────────────────────────
        with audit_sub:
            st.markdown(
                '<div class="section-header">De-identification Audit Log</div>'
                '<div class="section-caption">'
                'Every uploaded DICOM is de-identified per HIPAA Safe Harbor. '
                'This table records the mapping between anonymous IDs and the '
                'original PHI that was scrubbed.</div>',
                unsafe_allow_html=True,
            )
            audit_df = _fetch_audit_logs()
            if audit_df.empty:
                st.info(
                    "No audit log entries yet. Upload a DICOM on the "
                    "Analysis tab to create records."
                )
            else:
                # ── Filters ─────────────────────────────────────
                fc1, fc2, fc3 = st.columns([2, 2, 1])
                with fc1:
                    search_anon = st.text_input(
                        "🔍 Filter by Anonymous ID",
                        key="audit_search_anon",
                        placeholder="e.g. a1b2c3d4",
                    )
                with fc2:
                    search_action = st.selectbox(
                        "Filter by Action",
                        options=["All"]
                        + sorted(audit_df["Action"].unique().tolist()),
                        key="audit_search_action",
                    )
                with fc3:
                    st.write("")  # vertical alignment spacer
                    st.write("")
                    csv_audit_full = audit_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇ CSV",
                        data=csv_audit_full,
                        file_name="audit_log.csv",
                        mime="text/csv",
                        key="dl_audit_csv",
                        use_container_width=True,
                    )

                filtered = audit_df.copy()
                if search_anon:
                    filtered = filtered[
                        filtered[_LABEL_ANON_ID].str.contains(
                            search_anon, case=False, na=False
                        )
                    ]
                if search_action != "All":
                    filtered = filtered[filtered["Action"] == search_action]

                st.metric("Total records", len(filtered))
                st.dataframe(
                    filtered,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Timestamp": st.column_config.DatetimeColumn(
                            format="YYYY-MM-DD HH:mm:ss"
                        ),
                        _LABEL_ANON_ID: st.column_config.TextColumn(
                            width="medium"
                        ),
                    },
                )

        # ── Inference Results table ─────────────────────────────
        with inference_sub:
            st.markdown(
                '<div class="section-header">Inference Results</div>'
                '<div class="section-caption">'
                'Every prediction made by the ensemble model is persisted, '
                'including raw scores and the generated FHIR DiagnosticReport.</div>',
                unsafe_allow_html=True,
            )
            inf_df = _fetch_inference_logs()
            if inf_df.empty:
                st.info(
                    "No inference records yet. Upload a DICOM on the "
                    "Analysis tab to create records."
                )
            else:
                # ── Filters ─────────────────────────────────────
                fc3, fc4, fc5 = st.columns([2, 2, 1])
                with fc3:
                    search_inf_anon = st.text_input(
                        "🔍 Filter by Anonymous ID",
                        key="inf_search_anon",
                        placeholder="e.g. a1b2c3d4",
                    )
                with fc4:
                    search_finding = st.selectbox(
                        "Filter by Finding",
                        options=["All"]
                        + sorted(
                            inf_df[_LABEL_TOP_FINDING].unique().tolist()
                        ),
                        key="inf_search_finding",
                    )
                with fc5:
                    st.write("")
                    st.write("")
                    csv_inf_full = inf_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇ CSV",
                        data=csv_inf_full,
                        file_name="inference_results.csv",
                        mime="text/csv",
                        key="dl_inf_csv",
                        use_container_width=True,
                    )

                filtered_inf = inf_df.copy()
                if search_inf_anon:
                    filtered_inf = filtered_inf[
                        filtered_inf[_LABEL_ANON_ID].str.contains(
                            search_inf_anon, case=False, na=False
                        )
                    ]
                if search_finding != "All":
                    filtered_inf = filtered_inf[
                        filtered_inf[_LABEL_TOP_FINDING] == search_finding
                    ]

                # ── Summary metrics ─────────────────────────────
                mc1, mc2, mc3 = st.columns(3)
                pneumonia_count = len(
                    filtered_inf[filtered_inf[_LABEL_TOP_FINDING] == "PNEUMONIA"]
                )
                normal_count = len(
                    filtered_inf[filtered_inf[_LABEL_TOP_FINDING] == "NORMAL"]
                )
                mc1.metric("Total Predictions", len(filtered_inf))
                mc2.metric("PNEUMONIA", pneumonia_count)
                mc3.metric("NORMAL", normal_count)

                st.dataframe(
                    filtered_inf,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Timestamp": st.column_config.DatetimeColumn(
                            format="YYYY-MM-DD HH:mm:ss"
                        ),
                        "FHIR Report": st.column_config.TextColumn(
                            width="large"
                        ),
                        _LABEL_ANON_ID: st.column_config.TextColumn(
                            width="medium"
                        ),
                    },
                )

                # ── Finding distribution chart ──────────────────
                st.write("")
                st.markdown(
                    '<div class="section-header">Finding Distribution</div>',
                    unsafe_allow_html=True,
                )
                chart_data = (
                    filtered_inf[_LABEL_TOP_FINDING]
                    .value_counts()
                    .reset_index()
                )
                chart_data.columns = ["Finding", "Count"]
                st.bar_chart(
                    chart_data.set_index("Finding"),
                    height=250,
                    color=["#0066FF"],
                )


# ─────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="app-footer">
        <strong>MedView-AI</strong> v{APP_VERSION} &nbsp;·&nbsp;
        For research and educational purposes only &nbsp;·&nbsp;
        Not for clinical diagnosis<br/>
        Built with Streamlit · TensorFlow · PostgreSQL &nbsp;·&nbsp;
        &copy; {datetime.now().year} MedView-AI
    </div>
    """,
    unsafe_allow_html=True,
)
