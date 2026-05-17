# Job Posting Embeddings & Retrieval Benchmark

## Overview
- This project supports personalized job search and recommendation using embeddings and FAISS for semantic matching, ranking, and querying of job postings
---

## Data
- Source: Automated retrieval from LinkedIn and Indeed via the JobSpy library.
- Deduplication: Managed through a CockroachDB backend to track job lifecycle (active/inactive status) and avoid duplicates.
- Processing: HTML/Markdown stripping via `BeautifulSoup` to clean raw job descriptions before modeling.
- Dataflow Overview: </p>
  <img width="811" height="329" alt="image" src="https://github.com/user-attachments/assets/d55f500c-6769-46d1-9ca9-a7faa0e6a46e" />


---

## Methodologies
- Embedding Models:
  - API-based: Google `gemini-embedding-001`
  - Locally hosted: `Qwen3-Embedding-0.6B` (1024d) and `Qwen3-Embedding-8B` (4096d) via Hugging Face
- Data Labeling techniques:
  - Rule-based labeling: Built regex-based heuristics on job titles to generate proxy ground-truth datasets for retrieval evaluation across multiple job categories
  - LLM-based labeling: Implemented a `gemma-3-27b-it` multi-label classification pipeline using few-shot prompting and strict classification rules to label job postings from both titles and descriptions into seven functional categories
- Feature Representations:
  - Title-only: Baseline for fast role identity
  - Title + Description: Full context, though it introduces more semantic noise
  - Title + LLM Summary: Uses `gemma-3-27b-it` to retrieve key context and skills of a job description
- Evaluation: Retrieval is indexed via FAISS (cosine similarity) and measured using Precision@k, Recall@k, F1@k, and NDCG@k across seven functional categories

---

## Key Results
- Feature engineering: Title-only embeddings performed competitively at low retrieval depth (k ≤ 10), while title + LLM-generated summaries achieved stronger and more consistent F1@k at larger retrieval depths (k ≥ 30) by adding contextual information while reducing description noise.
- Model comparison: Qwen3-Embedding-8B outperformed both Gemini and Qwen3-Embedding-0.6B across most functional groups, reaching Precision@100 scores up to 0.96 and NDCG near 0.99 in Finance & Accounting and Software Engineering retrieval tasks.
- Dimensionality trade-off: The smaller Qwen3-Embedding-0.6B model occasionally outperformed the 4096-dimension 8B model on broad categories such as Entry Level, suggesting that higher-dimensional embeddings can amplify noisy or ambiguous semantic signals in loosely defined groups.

---

## Tech Stack
- Languages/Tools: Python, FAISS, scikit-learn
- LLMs: Google Gemini & Gemma-3 (API), Qwen3 (via Transformers)
- Storage: CockroachDB (SQLAlchemy)

---

## Setup & Usage
1. Environment: Create a `.venv` and install `requirements.txt`.
2. Secrets: Add `GEMINI_API_KEY` and DB credentials to a `.env` file.
3. Execution:
   - `python read_data.py`: Fetch and clean data.
   - `python ground_truth_generation.py`: Generate LLM-based labels.
   - `train.ipynb`: Run end-to-end evaluation and retrieval experiments.
