# Project Specification: MedView-AI

## 1. Project Overview

MedView-AI is an end-to-end medical imaging platform for **binary classification of chest X-rays** (NORMAL vs PNEUMONIA). The system:

1. Ingests **DICOM** (`.dcm`) files via a Streamlit browser UI.
2. **De-identifies** them in memory following the **HIPAA Safe Harbor** method (18 PHI tag categories).
3. Runs **ensemble deep-learning inference** using EfficientNet-B4 + Hybrid Vision Transformer (ViT), both loaded as native Keras models.
4. Generates **Grad-CAM heatmaps** highlighting the image regions that most influenced the prediction.
5. Produces an **HL7 FHIR R4 DiagnosticReport** JSON for healthcare interoperability.
6. Persists every de-identification event and inference result to **PostgreSQL** for full audit trail.
7. Renders all results — prediction banner, confidence metrics, heatmap overlay, class-score chart, FHIR report, downloadable artifacts — in a single-page **Streamlit dashboard**.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          User's Browser                             │
│                    http://localhost:8501                             │
└────────────────────────────┬────────────────────────────────────────┘
                             │  DICOM upload
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard (streamlit_app.py)            │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌────────────────────┐ │
│  │  Upload   │→ │ De-ident  │→ │ Inference│→ │ Grad-CAM + Display │ │
│  │ (pydicom) │  │ (HIPAA)   │  │ (Keras)  │  │ (OpenCV/mpl)       │ │
│  └──────────┘  └─────┬─────┘  └────┬─────┘  └────────────────────┘ │
│                       │             │                                │
│                       ▼             ▼                                │
│               ┌─────────────────────────┐                           │
│               │     SQLAlchemy ORM      │                           │
│               │  (services.py)          │                           │
│               └────────────┬────────────┘                           │
└────────────────────────────┼────────────────────────────────────────┘
                             │  TCP :5432
                             ▼
                   ┌──────────────────┐
                   │  PostgreSQL 15   │
                   │  (Docker)        │
                   │  - audit_log     │
                   │  - inference_    │
                   │    results       │
                   └──────────────────┘
```

### Architectural Constraints

- **Cloud Agnostic** — runs entirely on a local machine via Docker Compose.
- **No external API gateway** — Streamlit serves both UI and processing logic.
- **Keras-native inference** — models are loaded as `.keras` files; no ONNX runtime.
- **Multi-stage Docker build** — optimised image size for the dashboard container.
- **Soft-fail design** — the app works without PostgreSQL (logs a warning); models generate dummy predictions when `.keras` files are absent.

---

## 3. Technology Stack

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| **UI / Dashboard** | Streamlit | ≥ 1.40.0 | Single-page web app — file upload, result display, audit log viewer |
| **Deep Learning** | TensorFlow / Keras | ≥ 2.18.0 | Model training (EfficientNet-B4 + Hybrid ViT) and native `.keras` inference |
| **Image Processing** | OpenCV (headless) | ≥ 4.10.0 | CLAHE contrast enhancement, image resizing, Grad-CAM heatmap overlay |
| **Medical Imaging** | pydicom | ≥ 2.4.4 | DICOM file parsing, PHI tag extraction, de-identification |
| **Explainability** | Grad-CAM (custom) | — | Gradient-weighted Class Activation Map via `tf.GradientTape`; Sobel-based saliency fallback |
| **Visualisation** | matplotlib | ≥ 3.8.3 | Heatmap figure rendering to PNG bytes |
| **Image I/O** | Pillow | ≥ 10.2.0 | PNG/JPEG image conversion for Streamlit display |
| **Database** | PostgreSQL | 15-alpine | Persistent storage for audit logs and inference records |
| **ORM** | SQLAlchemy | ≥ 2.0.27 | Declarative ORM models, session management, table auto-creation |
| **DB Adapter** | psycopg2-binary | ≥ 2.9.9 | PostgreSQL wire-protocol driver for SQLAlchemy |
| **Data Contracts** | Pydantic | ≥ 2.6.1 | Schema validation for `InferenceResult`, `PredictionItem`, and FHIR R4 `DiagnosticReport` |
| **DataFrames** | pandas | ≥ 2.2.0 | Tabular display of audit logs / inference history in Streamlit |
| **Numerical** | NumPy | ≥ 2.1.0 | Array operations throughout DICOM → pixel → preprocessing → inference pipeline |
| **Class Balancing** | scikit-learn | ≥ 1.4.0 | `compute_class_weight("balanced")` to handle NORMAL/PNEUMONIA imbalance during training |
| **Testing** | pytest | ≥ 8.0.1 | 8 tests across 5 test classes (DICOM, schemas, inference, explainability, integration) |
| **Env Config** | python-dotenv | ≥ 1.0.1 | Load `.env` variables (e.g. `DATABASE_URL`) |
| **Containerisation** | Docker + Docker Compose | — | Two-service stack: `dashboard` (Streamlit) + `db` (PostgreSQL) |

### How the Tools Link Together

```
pydicom ──→ NumPy ──→ OpenCV (CLAHE + resize) ──→ TensorFlow/Keras (inference)
                                                         │
                                  ┌──────────────────────┴───────────────────────┐
                                  ▼                                              ▼
                       matplotlib + OpenCV                               Pydantic (FHIR)
                       (Grad-CAM PNG)                                    (DiagnosticReport)
                                  │                                              │
                                  └──────────────┬───────────────────────────────┘
                                                 ▼
                                           Streamlit UI
                                      (display + download)
                                                 │
                                                 ▼
                                     SQLAlchemy → PostgreSQL
                                     (audit_log + inference_results)
```

- **pydicom** parses raw `.dcm` bytes into a `Dataset` object.
- **NumPy** extracts `pixel_array` and normalises to `[0, 1]` float32.
- **OpenCV** applies CLAHE contrast enhancement and resizes to 224×224 — matching the exact preprocessing used during training.
- **TensorFlow/Keras** loads both `.keras` model files, runs `model.predict()`, and averages the scores (50/50 ensemble weight).
- **matplotlib** renders the Grad-CAM overlay to PNG bytes; **OpenCV** does the colour-map blending.
- **Pydantic** validates the prediction output and constructs the FHIR R4 `DiagnosticReport` JSON.
- **pandas** converts database query results into DataFrames for Streamlit's `st.dataframe()`.
- **SQLAlchemy + psycopg2** map Python objects to PostgreSQL tables (`audit_log`, `inference_results`) and handle all CRUD.
- **Streamlit** orchestrates the entire pipeline per file-upload event and renders all outputs.

---

## 4. Project File Structure

```
medical-imaging-detection/
│
├── streamlit_app.py            # Main Streamlit dashboard (~990 lines)
│                                 — Upload, de-id, inference, Grad-CAM, FHIR, audit logs
│
├── src/
│   ├── __init__.py
│   │
│   ├── app/                    # Application layer (DB + schemas)
│   │   ├── __init__.py
│   │   ├── services.py         # SQLAlchemy ORM models, DB init, audit/inference persistence
│   │   └── schemas.py          # Pydantic models: PredictionItem, InferenceResult,
│   │                             FHIRDiagnosticReport (FHIR R4)
│   │
│   ├── ml/                     # Machine Learning layer
│   │   ├── __init__.py
│   │   ├── train.py            # Training script — EfficientNet-B4 (two-phase) + Hybrid ViT
│   │   │                         CLI: python -m src.ml.train --data_dir ./data/chest_xray
│   │   ├── inference.py        # KerasEnsemble class — loads .keras models, runs prediction
│   │   └── explain.py          # Grad-CAM (TF GradientTape) + Sobel saliency fallback +
│   │                             heatmap overlay + PNG export
│   │
│   └── utils/                  # Utility layer
│       ├── __init__.py
│       └── dicom_utils.py      # HIPAA Safe Harbor de-identification (18 tags),
│                                 PHI extraction, DICOM→NumPy conversion
│
├── models/                     # Trained model artefacts (git-tracked .keras only)
│   ├── efficientnet_b4.keras   # Final EfficientNet-B4 model (91.5% val accuracy, 0.975 AUC)
│   ├── vit.keras               # Final Hybrid ViT model (86.7% val accuracy, 0.945 AUC)
│   ├── model_meta.json         # {"class_names": ["NORMAL","PNEUMONIA"], "img_size": [224,224]}
│   ├── eff_phase1_best.keras   # Best checkpoint — EfficientNet Phase 1 (head-only)
│   ├── eff_phase2_best.keras   # Best checkpoint — EfficientNet Phase 2 (fine-tune)
│   ├── vit_best.keras          # Best checkpoint — ViT
│   ├── eff_phase1_history.csv  # Training metrics log — Phase 1
│   ├── eff_phase2_history.csv  # Training metrics log — Phase 2
│   └── vit_history.csv         # Training metrics log — ViT
│
├── data/                       # Training data (git-ignored)
│   └── chest_xray/             # Kaggle Chest X-Ray Pneumonia dataset
│       ├── train/              # 5,216 images (NORMAL: 1,341 · PNEUMONIA: 3,875)
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       ├── val/                # 16 images (unused — too small)
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       └── test/               # 624 images (used as validation set)
│           ├── NORMAL/
│           └── PNEUMONIA/
│
├── tests/
│   ├── __init__.py
│   └── test_medview.py         # 8 tests across 5 classes:
│                                 TestDicomDeidentification (3)
│                                 TestSchemas (1)
│                                 TestInference (1)
│                                 TestExplainability (2)
│                                 TestStreamlitPipeline (1)
│
├── docker/
│   └── dashboard.Dockerfile    # Multi-stage build: Python 3.11-slim → Streamlit
│
├── docker-compose.yml          # 2 services: dashboard + db
├── requirements.txt            # 14 Python packages
├── .env                        # DATABASE_URL=postgresql://user:password@localhost:5433/medview
├── .gitignore
├── .dockerignore
└── PROJECT_SPEC.md             # This file
```

---

## 5. Data Standards

| Standard | Usage |
|----------|-------|
| **DICOM** (Digital Imaging and Communications in Medicine) | Input format — `.dcm` chest X-ray files parsed by pydicom |
| **HIPAA Safe Harbor** (45 CFR §164.514(b)(2)) | 18 PHI identifier categories wiped from DICOM tags before any processing |
| **DICOM PS3.15 Basic Profile** | Attribute Confidentiality — tags set to `"ANONYMIZED"`, patient ID replaced with UUID |
| **HL7 FHIR R4** (`DiagnosticReport`) | Output format — standardised healthcare JSON with LOINC code `58718-8` |
| **LOINC 58718-8** | "Automated analysis of Chest X-ray" — coded into the FHIR `DiagnosticReport.code.coding` |

---

## 6. End-to-End Data Flow

Below is the complete lifecycle of a single DICOM upload, from the moment the user drops a file to every output rendered on the dashboard.

### Step 1 — Upload (Streamlit + pydicom)

```
User drags chest_xray.dcm into the Streamlit file uploader widget
  → st.file_uploader reads raw bytes into memory
  → pydicom.dcmread(BytesIO(file_bytes)) parses bytes into a Dataset object
```

**File:** `streamlit_app.py` → calls `read_dicom_bytes()` from `src/utils/dicom_utils.py`

### Step 2 — PHI Extraction (pydicom)

```
extract_phi(dataset)
  → Iterates all 18 HIPAA Safe Harbor DICOM tags
  → Returns dict: {"PatientName": "Doe^John", "PatientID": "PAT-001",
                    "InstitutionName": "Hospital", ...}
```

**File:** `src/utils/dicom_utils.py` → `extract_phi()`

### Step 3 — De-identification (pydicom + uuid)

```
deidentify(dataset)
  → Calls extract_phi() to capture original PHI
  → Generates UUID via uuid.uuid4() — e.g. "550e8400-e29b-..."
  → Overwrites all 18 HIPAA tags with "ANONYMIZED"
  → Stamps PatientID = UUID, PatientName = "ANONYMOUS"
  → Returns (cleaned_dataset, phi_dict, anonymous_id)
```

**Result:** The AI never sees the patient's real identity.  
**File:** `src/utils/dicom_utils.py` → `deidentify()`

### Step 4 — Pixel Extraction (pydicom + NumPy)

```
dicom_to_numpy(dataset)
  → dataset.pixel_array → raw uint16 array (e.g. 1024×1024)
  → Cast to float32, min-max normalise to [0.0, 1.0]
  → Returns numpy.ndarray shape (H, W)
```

**File:** `src/utils/dicom_utils.py` → `dicom_to_numpy()`

### Step 5 — Preprocessing (OpenCV + NumPy)

```
KerasEnsemble.preprocess(image)
  → Convert greyscale (H,W) to RGB (H,W,3) by stacking channels
  → Scale to [0, 255] uint8
  → Resize to (224, 224) via cv2.resize
  → CLAHE contrast enhancement (cv2.createCLAHE on LAB L-channel)
  → EfficientNet normalisation: pixel / 127.5 - 1.0 → range [-1, 1]
  → Expand dims → (1, 224, 224, 3) batch tensor
```

**Why CLAHE:** Chest X-rays are low-contrast greyscale; CLAHE brings out subtle density differences (infiltrates, consolidation) that the models need.  
**File:** `src/ml/inference.py` → `KerasEnsemble.preprocess()`

### Step 6 — Ensemble Inference (TensorFlow / Keras)

```
KerasEnsemble.predict(image)
  → Calls preprocess(image)
  → EfficientNet-B4:  model.predict(batch) → [P(NORMAL), P(PNEUMONIA)]  × 0.5 weight
  → Hybrid ViT:       model.predict(batch) → [P(NORMAL), P(PNEUMONIA)]  × 0.5 weight
  → Weighted average of both score vectors
  → Returns dict:
      {
        "predictions": [{"label": "PNEUMONIA", "confidence": 0.982}, ...],
        "raw_scores": [0.018, 0.982],
        "top_finding": "PNEUMONIA",
        "top_confidence": 0.982
      }
```

**File:** `src/ml/inference.py` → `KerasEnsemble.predict()`

### Step 7 — Grad-CAM Heatmap (TensorFlow + OpenCV + matplotlib)

```
generate_heatmap_png(original_image, raw_scores, class_index)
  → If a Keras model is available:
      → Build sub-model: input → last Conv2D output + final prediction
      → tf.GradientTape: compute gradients of class score w.r.t. conv feature map
      → Global-average-pool gradients → channel weights
      → Weighted sum of feature maps → ReLU → normalise → heatmap (H, W) in [0, 1]
  → Else (fallback):
      → saliency_map_approx(): Sobel edge magnitude × Gaussian blur × class confidence

overlay_heatmap(original, heatmap, alpha)
  → cv2.resize heatmap to original dimensions
  → cv2.applyColorMap (JET colourmap) → colour heatmap
  → cv2.addWeighted blend: (1-α) × original + α × heatmap
  → Returns overlaid BGR image as uint8

matplotlib renders overlay → fig.savefig(BytesIO, format="png") → raw PNG bytes
```

**File:** `src/ml/explain.py` → `grad_cam_tf()`, `saliency_map_approx()`, `overlay_heatmap()`, `generate_heatmap_png()`

### Step 8 — Audit Persistence (SQLAlchemy → PostgreSQL)

```
_save_audit_log(anonymous_id, phi_dict)
  → INSERT INTO audit_log (anonymous_id, original_patient_name,
      original_patient_id, institution, timestamp, action)

_save_inference_record(anonymous_id, inference_result, ...)
  → INSERT INTO inference_results (anonymous_id, top_finding,
      top_confidence, raw_scores, fhir_report, timestamp)
```

**Soft-fail:** If PostgreSQL is unreachable, the app logs a warning and continues — all visual outputs still work.  
**File:** `src/app/services.py`

### Step 9 — FHIR Report Construction (Pydantic)

```json
{
    "resourceType": "DiagnosticReport",
    "id": "550e8400-e29b-...",
    "status": "final",
    "code": {
        "coding": [{
            "system": "http://loinc.org",
            "code": "58718-8",
            "display": "Automated analysis of Chest X-ray"
        }]
    },
    "issued": "2026-02-18T10:30:00+00:00",
    "conclusion": "Findings — PNEUMONIA: 98.2%"
}
```

**File:** `src/app/schemas.py` → `FHIRDiagnosticReport` (Pydantic BaseModel)

### Step 10 — Dashboard Rendering (Streamlit)

All outputs are rendered in the **Analysis** tab:

| Dashboard Element | Streamlit Widget | Data Source |
|-------------------|-----------------|-------------|
| **Prediction banner** (PNEUMONIA ⚠️ / NORMAL ✅) | `st.markdown` (custom HTML/CSS) | `top_finding`, `top_confidence` |
| **Metric cards** (NORMAL %, PNEUMONIA %, Top Finding, Anon ID) | `st.metric` × 4 columns | `raw_scores`, `top_finding`, `anonymous_id` |
| **Original X-Ray** image | `st.image` | `pixel_array` converted to uint8 RGB |
| **Grad-CAM Heatmap Overlay** | `st.image` | `overlay_img` from `overlay_heatmap()` |
| **Full Grad-CAM Figure** (expandable) | `st.image` inside `st.expander` | `heatmap_png_bytes` from matplotlib |
| **Class Scores table** | `st.dataframe` | pandas DataFrame of `CLASS_NAMES` × `raw_scores` |
| **Class Scores bar chart** | `st.bar_chart` | pandas DataFrame |
| **FHIR R4 DiagnosticReport** | `st.json` | `fhir_data` dict |
| **Extracted PHI** (optional toggle) | `st.json` inside `st.warning` | `phi` dict from `extract_phi()` |
| **DICOM Metadata** (optional toggle) | `st.json` | All non-PixelData DICOM elements |
| **Download: Heatmap PNG** | `st.download_button` | `heatmap_png_bytes` |
| **Download: FHIR Report JSON** | `st.download_button` | `json.dumps(fhir_data)` |
| **Download: Scores JSON** | `st.download_button` | `json.dumps(result)` |

The **Audit Logs** tab (second tab) queries PostgreSQL via SQLAlchemy and displays:

- **De-identification Audit Log** — filterable table of all PHI→UUID mappings with CSV export
- **Inference Results** — filterable table of all predictions with summary metrics (total, PNEUMONIA count, NORMAL count), distribution bar chart, and CSV export

---

## 7. ML Models — Training Details

### Dataset

**Kaggle Chest X-Ray Pneumonia** — 5,840 images total

- Train: 5,216 (NORMAL: 1,341 · PNEUMONIA: 3,875 — **2.9:1 class imbalance**)
- Validation: 624 images (uses `test/` split; official `val/` has only 16 images)
- Resolution: resized to **224×224** (from variable original sizes)

### Preprocessing Pipeline (identical for training and inference)

1. **RGB conversion** — greyscale → 3 channels
2. **Resize** to 224×224
3. **CLAHE** (Contrast-Limited Adaptive Histogram Equalisation) on LAB L-channel — `clipLimit=2.0, tileGridSize=(8,8)`
4. **EfficientNet normalisation** — `pixel / 127.5 - 1.0` → range `[-1, 1]`

### Model 1: EfficientNet-B4 (texture / pattern detector)

- **Base:** `EfficientNetB4(weights="imagenet", include_top=False, pooling="avg")`
- **Head:** BatchNorm → Dropout(0.4) → Dense(256, relu) → BatchNorm → Dropout(0.3) → Dense(2, softmax)
- **Phase 1 (head-only):** Backbone frozen, Adam(lr=1e-3), ReduceLROnPlateau, ~5 epochs
- **Phase 2 (fine-tune):** Top 20 layers unfrozen, Adam(WarmupCosineDecay, base_lr=5e-5), ~10 epochs
- **Result:** 91.5% validation accuracy, 0.975 AUC

### Model 2: Hybrid Vision Transformer (geometric / structural detector)

- **Architecture:** 4-layer ConvStem (stride-2 each → 14×14 = 196 tokens) + 2 Transformer blocks
- **Design rationale:** "Early Convolutions Help Transformers See Better" (Xiao et al.) — ConvStem gives spatial inductive bias so the ViT converges on small datasets (~5K images)
- **Key components:** `ConvStem` → positional embedding → `TransformerBlock` × 2 → LayerNorm → GAP → Dense head
- **Regularisation:** StochasticDepth (linearly increasing drop-path), label smoothing (0.1), gradient clipping (clipnorm=1.0), AdamW(weight_decay=1e-4)
- **LR Schedule:** WarmupCosineDecay (2-epoch warmup → cosine decay)
- **Result:** 86.7% validation accuracy, 0.945 AUC

### Ensemble Strategy

- **Weighted average:** EfficientNet × 0.5 + ViT × 0.5 (equal weighting)
- **Confidence threshold:** 0.5 for positive class inclusion in results
- **Fallback:** If no models are found on disk, returns random dummy scores for development

### Custom Keras Classes (registered with `@keras.utils.register_keras_serializable()`)

These are required for loading `vit.keras` — they are imported by `inference.py` from `train.py`:

- `ConvStem` — 4-layer convolutional patch embedding
- `StochasticDepth` — drop-path regularisation
- `TransformerBlock` — pre-norm MHA + FFN encoder block
- `WarmupCosineDecay` — LR schedule

---

## 8. Database Schema

**PostgreSQL** (running in Docker on port 5433 → container port 5432)

- Database: `medview`
- User: `user` / Password: `password`

### Table: `audit_log`

| Column | Type | Description |
|--------|------|-------------|
| `id` | `INTEGER` (PK, auto) | Row identifier |
| `anonymous_id` | `VARCHAR(64)` (indexed) | UUID assigned during de-identification |
| `original_patient_name` | `VARCHAR(256)` | Original `PatientName` from DICOM (before wipe) |
| `original_patient_id` | `VARCHAR(128)` | Original `PatientID` from DICOM |
| `institution` | `VARCHAR(256)` | Original `InstitutionName` from DICOM |
| `timestamp` | `DATETIME` | UTC timestamp of de-identification |
| `action` | `VARCHAR(64)` | Always `"de-identification"` |

### Table: `inference_results`

| Column | Type | Description |
|--------|------|-------------|
| `id` | `INTEGER` (PK, auto) | Row identifier |
| `anonymous_id` | `VARCHAR(64)` (indexed) | Links to `audit_log.anonymous_id` |
| `top_finding` | `VARCHAR(128)` | `"NORMAL"` or `"PNEUMONIA"` |
| `top_confidence` | `VARCHAR(16)` | Confidence score as string (e.g. `"0.982"`) |
| `raw_scores` | `TEXT` | JSON array of all class scores |
| `dicom_url` | `TEXT` | Reserved (empty string — no object storage) |
| `heatmap_url` | `TEXT` | Reserved (empty string — no object storage) |
| `fhir_report` | `TEXT` | Full FHIR DiagnosticReport as JSON string |
| `timestamp` | `DATETIME` | UTC timestamp of inference |

---

## 9. Docker Infrastructure

### docker-compose.yml — 2 Services

```yaml
services:
  dashboard:          # Streamlit app
    build: docker/dashboard.Dockerfile
    ports: 8501:8501
    env: DATABASE_URL=postgresql://user:password@db:5432/medview
    depends_on: db
    volumes: ./models:/app/models

  db:                 # PostgreSQL
    image: postgres:15-alpine
    ports: 5433:5432  # host:container
    env: POSTGRES_USER=user, POSTGRES_PASSWORD=password, POSTGRES_DB=medview
    volumes: pg_data:/var/lib/postgresql/data
```

### dashboard.Dockerfile — Multi-stage Build

- **Stage 1 (builder):** `python:3.11-slim` → `pip install --prefix=/install -r requirements.txt`
- **Stage 2 (runtime):** `python:3.11-slim` → copy installed packages + source code + models → `CMD streamlit run streamlit_app.py`
- Installs `libgl1` and `libglib2.0-0` for OpenCV headless runtime

---

## 10. Testing

**Framework:** pytest  
**Test file:** `tests/test_medview.py`  
**Total:** 8 tests across 5 classes

| Test Class | Tests | What It Validates |
|------------|-------|-------------------|
| `TestDicomDeidentification` | 3 | PHI extraction correctness, HIPAA tag wiping, pixel array survival after de-id |
| `TestSchemas` | 1 | FHIR DiagnosticReport structure — resourceType, status, LOINC coding |
| `TestInference` | 1 | Ensemble returns valid dict with `predictions`, `raw_scores`, `top_finding` |
| `TestExplainability` | 2 | Heatmap PNG bytes validity (magic header), saliency map shape matches input |
| `TestStreamlitPipeline` | 1 | Full end-to-end: DICOM bytes → parse → de-id → inference → heatmap PNG |

Run: `python -m pytest tests/test_medview.py -v`

---

## 11. How to Run

### Prerequisites

- Python 3.11+ (developed on 3.13.3)
- Docker + Docker Compose
- ~2 GB disk for TensorFlow and model weights

### Local Development (without Docker)

```bash
# 1. Create virtual environment
python -m venv venv
.\venv\Scripts\activate          # Windows
# source venv/bin/activate       # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start PostgreSQL (Docker)
docker compose up db -d

# 4. (Optional) Train models — requires data/chest_xray/ dataset
python -m src.ml.train --data_dir ./data/chest_xray --epochs 25

# 5. Launch Streamlit dashboard
streamlit run streamlit_app.py --server.port 8501

# 6. Open browser
#    → http://localhost:8501
```

### Docker (full stack)

```bash
# Build and start both services
docker compose up -d --build

# Dashboard: http://localhost:8501
# PostgreSQL: localhost:5433 (user/password, database: medview)
```

### Run Tests

```bash
python -m pytest tests/test_medview.py -v
```

---

## 12. Sidebar Controls & System Status

The Streamlit sidebar provides:

| Control | Type | Effect |
|---------|------|--------|
| **Show extracted PHI** | Checkbox | Displays the original PHI that was wiped (demo/debug only) |
| **Show DICOM metadata** | Checkbox | Displays all non-PixelData DICOM tags after de-identification |
| **Heatmap overlay opacity** | Slider (0.1–0.9) | Controls the alpha blending of the Grad-CAM heatmap on the X-ray |
| **ML Models** status | Badge (● / ○) | Green if at least one `.keras` model was loaded |
| **PostgreSQL** status | Badge (● / ○) | Green if database connection and table creation succeeded |

---

## 13. FHIR R4 DiagnosticReport Structure

```json
{
    "resourceType": "DiagnosticReport",
    "id": "<anonymous_uuid>",
    "status": "final",
    "code": {
        "coding": [{
            "system": "http://loinc.org",
            "code": "58718-8",
            "display": "Automated analysis of Chest X-ray"
        }]
    },
    "issued": "2026-02-18T10:30:00.000000+00:00",
    "conclusion": "Findings — PNEUMONIA: 98.2%"
}
```

---

## 14. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Streamlit over FastAPI** | Single-page dashboard eliminates the need for a separate frontend; users interact directly via browser upload |
| **Keras-native over ONNX** | ONNX export fails with custom ViT layers on TF 2.20+; native `.keras` loading is reliable and supports Grad-CAM via `tf.GradientTape` |
| **CLAHE preprocessing** | Chest X-rays are inherently low-contrast; adaptive histogram equalisation significantly improves model accuracy |
| **Two-phase EfficientNet training** | Freeze→unfreeze strategy with warmup prevents catastrophic forgetting of ImageNet features |
| **ConvStem ViT** | Pure patch-projection ViT fails to converge on 5K images; convolutional stem provides spatial inductive bias |
| **Soft-fail database** | Dashboard remains fully functional without PostgreSQL — critical for local dev and demo scenarios |
| **Equal ensemble weights** | Both models contribute complementary signals (texture vs geometry); 50/50 averaging is robust without a calibration set |
| **test/ as validation** | Official Kaggle val/ split has only 16 images — statistically meaningless; test/ (624 images) provides stable metrics |
