# MedView-AI

MedView-AI is an end-to-end medical imaging platform for binary classification of chest X-rays (NORMAL vs PNEUMONIA) using deep learning. It features DICOM upload, de-identification, ensemble inference (EfficientNet-B4 + ViT), Grad-CAM explainability, FHIR R4 report generation, and full audit logging to PostgreSQL, all in a modern Streamlit dashboard.

## Features
- Upload and de-identify DICOM (.dcm) files (HIPAA Safe Harbor)
- Ensemble deep learning inference (EfficientNet-B4 + ViT)
- Grad-CAM heatmaps for explainability
- HL7 FHIR R4 DiagnosticReport JSON output
- PostgreSQL audit log and inference result storage
- Single-page Streamlit dashboard with interactive results
- Soft-fail: works without PostgreSQL (logs warning)
- Docker Compose for local deployment

## Technology Stack
- Streamlit, TensorFlow/Keras, OpenCV, pydicom, matplotlib, Pillow
- PostgreSQL, SQLAlchemy, psycopg2-binary, Pydantic, pandas, scikit-learn
- Docker, Docker Compose

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

## Testing
Run all tests with:
```sh
pytest
```

## CI/CD
GitHub Actions workflow runs tests and checks on every push/pull request.

## License
MIT
