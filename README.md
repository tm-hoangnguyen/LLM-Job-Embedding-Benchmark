# Job LLM / Embeddings Benchmark

## Overview
This repository is a **standalone Python toolkit** for **job-posting embeddings**, **FAISS retrieval**, **ground-truth generation**, and **evaluation workflows**.

---

## Data
- **Source**: Automated retrieval from LinkedIn and Indeed via the JobSpy library.
- **Deduplication**: Managed through a CockroachDB backend to ensure database integrity and track job lifecycle (active/inactive status).
- **Processing**: HTML/Markdown stripping using `BeautifulSoup` to clean raw job descriptions for downstream modeling.

---

## Methodologies
- **Embedding Models**: 
  - **API-based**: Google `gemini-embedding-001`.
  - **Locally Hosted**: `Qwen3-Embedding-0.6B` (1024d) and `Qwen3-Embedding-8B` (4096d) via Hugging Face.
- **Feature Representations**: 
  - **Title-only**: Baseline for fast role identity.
  - **Title + Description**: Full context but introduces more semantic noise.
  - **Title + LLM Summary**: Uses `gemma-3-27b-it` to extract key skills and responsibilities to minimize noise.
- **Evaluation**: Performance is indexed via **FAISS** (cosine similarity) and measured using Precision@k, Recall@k, F1@k, and NDCG@k across seven functional categories.

---

## Key Results
- **Retrieval depth (top‑k)**: With **top‑k**, metrics look at the first *k* retrieved jobs for each query. **Title + LLM summary** improved **F1@k** versus title‑only embeddings when **k was large (k ≥ 30)**—where richer query/job text matters more than for very small candidate lists.
- **Model choice**: On **narrow verticals** (e.g. **Finance & Accounting** among our labeled categories), **Qwen3‑Embedding‑8B** achieved the strongest numbers we observed (**Precision@100** up to **~0.96**), ahead of **Gemini** and **Qwen3‑Embedding‑0.6B** on those slices.
- **When smaller embeddings win**: For **broad, loosely defined categories** (e.g. **Entry Level**, where seniority cues vary wildly—“junior,” “associate,” “senior I/II,” numeric bands—and titles rarely map cleanly to one semantic niche), the **1024‑d (0.6B)** model sometimes **outperformed the 4096‑d (8B)** model—consistent with extra dimensions amplifying irrelevant variance when boundaries are fuzzy—so **model size is not universally better** here.

---

## Tech Stack
- **Languages/Tools**: Python, PyTorch, FAISS, scikit-learn.
- **LLMs**: Google Gemini (API), Qwen3 & Gemma-3 (via Transformers).
- **Storage**: CockroachDB (SQLAlchemy).

---

## Setup & Usage
1. **Environment**: Create a `.venv` and install `requirements.txt`.
2. **Secrets**: Configure `GEMINI_API_KEY` and DB credentials in a `.env` file.
3. **Execution**:
   - `python read_data.py`: Fetch and clean data.
   - `python ground_truth_generation.py`: Generate LLM-based labels.
   - `train.ipynb`: Run end-to-end evaluation and retrieval experiments.