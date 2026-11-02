# Manufacturing Manual Assistant

AI-powered manufacturing assistant that answers operator questions strictly grounded in verified machine manuals with original PDF page citations. SQL-enforced machine isolation prevents cross-machine contamination. If documentation lacks evidence, the system provides an honest refusal instead of guessing.

---

## Team Integration & Handoff

This repository is a fully integrated, production-ready prototype configured for seamless handoff between team members:

* **Member 1 (Backend & Infrastructure — HP Omen)**:
  * Full Django REST API with PostgreSQL and pgvector drivers (`psycopg[binary]`, `pgvector`).
  * `docker-compose.yml` for automated `pgvector/pgvector:pg16` containerization.
  * `python manage.py setup_pgvector` command to activate vector extensions.
  * SQL-level machine isolation and hybrid search (BM25 + Cosine similarity).
  * `python manage.py run_rag_evaluation` benchmark suite reporting accuracy and latency metrics.
* **Member 2 (Frontend & UI/UX Design)**:
  * Sleek dark engineering dashboard theme located in [`static/css/dashboard.css`](file:///c:/Users/giris/manual%20assistant/static/css/dashboard.css).
  * Real-time telemetry monitoring [`/api/v1/health/`](http://127.0.0.1:8000/api/v1/health/).
  * Drag-and-drop PDF ingestion studio with progress feedback.
  * Citation inspector with page badges, revision tracking, and verbatim context excerpts.
  * Interactive scenario chips (lubrication, pressure, spindle warm-up, adversarial isolation).

---

## 10-Phase Roadmap & Status

| Phase | Milestone | Status | Key Deliverables |
|---|---|---|---|
| **1** | Environment Verification | **Completed** | Windows PowerShell setup, PyTorch CPU/CUDA, Python 3.13 compatibility. |
| **2** | PDF Ingestion & OCR | **Completed** | PyMuPDF text extraction + RapidOCR fallback for scanned pages. |
| **3** | Chunking & Embeddings | **Completed** | Section-aware chunking, MiniLM-L6-v2 embeddings, offline HF caching. |
| **4** | Grounded Generation | **Completed** | Extractive synthesis, verbatim citations, zero-hallucination refusal. |
| **5** | Django REST Framework API | **Completed** | `/ingest/`, `/ask/`, `/manuals/`, unit test suite. |
| **6** | Revisions & Hybrid Search | **Completed** | Latest-revision resolver, BM25 + dense vector ranking (`0.65 / 0.35`). |
| **7** | Database Chunk Persistence | **Completed** | `ManualChunk` model, SQL `machine_id` pre-filtering, `/health/` API. |
| **8** | Modern UI & Telemetry | **Completed** | Glassmorphic dashboard, latency badge (`12.9 ms`), drag-and-drop. |
| **9** | Local LLM & Benchmarks | **Completed** | Ollama/vLLM OpenAI endpoint support, `run_rag_evaluation` command (100% pass). |
| **10**| PostgreSQL + pgvector Ready| **Completed** | `docker-compose.yml`, `psycopg`, `pgvector`, and `setup_pgvector` tool. |

---

## Quickstart (Zero-Setup SQLite)

### 1. Activate Environment & Install Dependencies
```powershell
cd "C:\Users\giris\manual assistant"
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. Apply Migrations & Load Sample Manuals
```powershell
python manage.py migrate
python manage.py load_sample_manuals
```

### 3. Launch the Application
```powershell
python manage.py runserver 127.0.0.1:8000
```
Open **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)** in your browser.

---

## Live PostgreSQL + pgvector Deployment

When ready to switch from SQLite to PostgreSQL + pgvector:

1. **Start the database container**:
   ```bash
   docker compose up -d
   ```
2. **Update `.env`**:
   ```env
   DATABASE_URL=postgresql://manuals:manuals@127.0.0.1:5432/manuals
   ```
3. **Migrate and verify pgvector**:
   ```powershell
   python manage.py migrate
   python manage.py setup_pgvector
   python manage.py load_sample_manuals
   ```
   The `/api/v1/health/` endpoint will immediately update `pgvector_ready: true` and `vector_backend: django_pgvector`.

---

## Optional Local LLM Integration (Ollama / vLLM)

By default, the assistant synthesizes verified answers purely from retrieved manual passages (no API key required). To enable natural language synthesis via a local LLM:

1. Start Ollama with your chosen model:
   ```bash
   ollama run llama3.2
   ```
2. Set the following in `.env`:
   ```env
   LLM_BASE_URL=http://127.0.0.1:11434/v1
   LLM_MODEL=llama3.2
   LLM_API_KEY=ollama
   ```
If the LLM endpoint is unavailable or times out, the system automatically falls back to extractive generation without interrupting the user.

---

## Evaluation & Test Suite

### Run All 25 Automated Tests
```powershell
pytest
```

### Run Grounding & Isolation Benchmarks
```powershell
python manage.py run_rag_evaluation
```
Reports accuracy, citation precision, isolation pass rate, and P50/P95 latency percentiles.

---

## API Specifications

### `POST /api/v1/assistant/ask/`
```json
{
  "machine_id": "M-HP200",
  "question": "What oil grade does the manual specify for routine lubrication?",
  "min_score": 0.35
}
```
**Response (200 OK)**:
```json
{
  "status": "grounded",
  "machine_id": "M-HP200",
  "question": "What oil grade does the manual specify for routine lubrication?",
  "answer": "Based only on the retrieved manual passages...",
  "citations": [
    {
      "manual_id": "MAN-HP200",
      "title": "Hydraulic Press HP-200 Operator Manual",
      "revision": "R2",
      "page_number": 3,
      "excerpt": "Lubricate the ram guide every 40 operating hours using ISO VG 68 oil."
    }
  ],
  "latency_ms": 12.9
}
```

### `GET /api/v1/health/`
Reports database status, backend engine, chunk counts, and pgvector readiness:
```json
{
  "status": "ok",
  "database": { "ok": true, "engine": "sqlite", "error": null },
  "vector_backend": "django_chunks",
  "pgvector_ready": false,
  "chunk_count": 7,
  "manual_count": 2
}
```
