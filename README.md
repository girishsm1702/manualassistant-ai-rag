# Manufacturing Manual Assistant

AI assistant that answers questions about manufacturing machines using uploaded PDF manuals. Answers are grounded in retrieved passages and include original PDF page citations. If the manuals do not contain enough evidence, the system says so instead of guessing.

## Team roles

This repository currently includes Member 2's RAG module plus a thin Django REST API so backend and frontend work can plug in.

Proposed API contract:

`POST /api/v1/assistant/ingest/` accepts a multipart PDF (`file`) or a local `pdf_path`.

`GET /api/v1/assistant/manuals/` lists ingested manuals.

`POST /api/v1/assistant/ask/`

```json
{
  "machine_id": "M-HP200",
  "question": "What does the manual say about routine maintenance?"
}
```

Response includes `status`, `answer`, `citations` (manual id, title, revision, original PDF page), and `evidence`.

## Local setup (Windows PowerShell)

Use Python 3.13. The system Python 3.14 is newer than many ML wheels.

```powershell
cd "C:\Users\giris\manual assistant"
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
copy .env.example .env
python manage.py migrate
python manage.py load_sample_manuals
python manage.py runserver
```

Open http://127.0.0.1:8000/

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Notes

- Django ingest stores chunks in `ManualChunk` and filters by `machine_id` in SQL before scoring. SQLite is the default local database. `docker-compose.yml` starts PostgreSQL + pgvector when Docker is available; set `DATABASE_URL` and run migrations.
- `GET /api/v1/health/` reports database status and chunk counts. Ask responses include `latency_ms`.
- Unit tests still use the file-backed numpy store so RAG tests can run without Django.
- Embeddings use `sentence-transformers/all-MiniLM-L6-v2` on CPU unless `EMBEDDING_DEVICE=cuda` is set *and* CUDA PyTorch is installed and verified.
- No paid LLM is required. If `LLM_BASE_URL` and `LLM_MODEL` are unset, answers are assembled from retrieved excerpts only.
- Scanned/image-only pages are OCR'd with RapidOCR when `OCR_ENABLED=true`.
- If several revisions of the same manual are ingested, questions use the latest revision unless `revision` is sent in the ask request.
