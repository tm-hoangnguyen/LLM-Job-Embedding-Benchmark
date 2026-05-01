# Job Posting Embeddings & Retrieval Benchmark

## Overview
A standalone Python toolkit for job-posting embeddings, FAISS retrieval, ground-truth generation, and evaluation workflows.

---

## Data
- Source: Automated retrieval from LinkedIn and Indeed via the JobSpy library.
- Deduplication: Managed through a CockroachDB backend to track job lifecycle (active/inactive status) and avoid duplicates.
- Processing: HTML/Markdown stripping via `BeautifulSoup` to clean raw job descriptions before modeling.

---

## Methodologies
- Embedding Models:
  - API-based: Google `gemini-embedding-001`
  - Locally hosted: `Qwen3-Embedding-0.6B` (1024d) and `Qwen3-Embedding-8B` (4096d) via Hugging Face
- Feature Representations:
  - Title-only: Baseline for fast role identity
  - Title + Description: Full context, though it introduces more semantic noise
  - Title + LLM Summary: Uses `gemma-3-27b-it` to pull out key skills and responsibilities, reducing noise
- Evaluation: Retrieval is indexed via FAISS (cosine similarity) and measured using Precision@k, Recall@k, F1@k, and NDCG@k across seven functional categories.

---

## Key Results
- Retrieval depth (top-k): Title + LLM summary improved F1@k over title-only embeddings when k was large (k ≥ 30), where richer text starts to matter more than it does for small candidate lists.
- Model choice: On narrow verticals like Finance & Accounting, Qwen3-Embedding-8B came out ahead — Precision@100 reached ~0.96, beating both Gemini and Qwen3-Embedding-0.6B on those slices.
- When smaller embeddings win: For broad, loosely defined categories like Entry Level — where seniority signals are inconsistent and titles don't map cleanly to a single semantic niche — the 1024-d (0.6B) model sometimes outperformed the 4096-d (8B) model. Extra dimensions can amplify irrelevant variance when category boundaries are fuzzy, so bigger isn't universally better here.

---

## Tech Stack
- Languages/Tools: Python, PyTorch, FAISS, scikit-learn
- LLMs: Google Gemini (API), Qwen3 & Gemma-3 (via Transformers)
- Storage: CockroachDB (SQLAlchemy)

---

## Setup & Usage
1. Environment: Create a `.venv` and install `requirements.txt`.
2. Secrets: Add `GEMINI_API_KEY` and DB credentials to a `.env` file.
3. Execution:
   - `python read_data.py`: Fetch and clean data.
   - `python ground_truth_generation.py`: Generate LLM-based labels.
   - `train.ipynb`: Run end-to-end evaluation and retrieval experiments.
