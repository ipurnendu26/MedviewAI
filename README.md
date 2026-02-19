# MedView-AI

MedView-AI is an end-to-end medical imaging platform for **binary classification of chest X-rays** (NORMAL vs PNEUMONIA) using deep learning. It features DICOM upload, in-memory de-identification (HIPAA Safe Harbor), ensemble inference (EfficientNet-B4 + Hybrid Vision Transformer), Grad-CAM explainability, HL7 FHIR R4 report generation, and full audit logging to PostgreSQL, all in a modern Streamlit dashboard.

---

## Features
- Upload and de-identify DICOM (.dcm) files (HIPAA Safe Harbor, 18 PHI tag categories)
- Ensemble deep learning inference (EfficientNet-B4 + Hybrid Vision Transformer)
- Grad-CAM heatmaps for explainability
- HL7 FHIR R4 DiagnosticReport JSON output for interoperability
- PostgreSQL audit log and inference result storage (via SQLAlchemy ORM)
- Single-page Streamlit dashboard with interactive results, confidence metrics, and downloadable artifacts
- Soft-fail: works without PostgreSQL (logs warning, dummy predictions if models missing)
- Docker Compose for local deployment (Streamlit + PostgreSQL)

---

## Architecture Overview

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

**Architectural Constraints:**
- Cloud agnostic: runs entirely on a local machine via Docker Compose
- No external API gateway: Streamlit serves both UI and processing logic
- Keras-native inference: models are loaded as `.keras` files; no ONNX runtime
- Multi-stage Docker build: optimised image size for the dashboard container
- Soft-fail design: the app works without PostgreSQL (logs a warning); models generate dummy predictions when `.keras` files are absent

---

## Technology Stack

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

---

## Quick Start
1. Clone the repo and install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Start PostgreSQL (optional, for audit logging):
   ```sh
   docker compose up db
   ```
3. Run the app:
   ```sh
   streamlit run streamlit_app.py
   ```
4. Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Testing
Run all tests with:
```sh
pytest
```

---

## CI/CD
GitHub Actions workflow runs tests and checks on every push/pull request. See `.github/workflows/ci.yml` for details.

---

## License
MIT
