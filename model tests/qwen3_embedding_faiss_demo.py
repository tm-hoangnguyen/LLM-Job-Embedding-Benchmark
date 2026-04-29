import glob
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
BATCH_SIZE = 16
SHUFFLE_SEED = 42
ROW_LIMIT = 1000
TEXT_COLUMNS: Sequence[str] = ("title", "description")


def load_job_postings_samples_v2(
    shuffle_seed: int = SHUFFLE_SEED,
    row_limit: int | None = ROW_LIMIT,
) -> pd.DataFrame:
    """
    Load job postings from assets/datasets/job_postings_samples_v2.csv.
    Mirrors preprocessing in ground_truth_generation.py: shuffle, dedup, drop null/empty, then first N rows.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(
        base_dir, "..", "assets", "datasets", "job_postings_samples_v2.csv"
    )
    csv_path = os.path.normpath(csv_path)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Expected CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)
    df = df.drop_duplicates(subset=["title", "description", "location"]).reset_index(drop=True)
    df = df.dropna(subset=["description"]).reset_index(drop=True)
    df = df[df["description"].astype(str).str.strip() != ""].reset_index(drop=True)

    if row_limit is not None:
        df = df[:row_limit].reset_index(drop=True)

    return df


def build_job_texts(
    df: pd.DataFrame, text_columns: Sequence[str] = TEXT_COLUMNS
) -> List[str]:
    """
    Build a per-job text string by concatenating the specified columns.
    """
    missing = [col for col in text_columns if col not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required text columns in DataFrame: {', '.join(missing)}"
        )

    cols = list(text_columns)
    texts: List[str] = []
    for _, row in df[cols].iterrows():
        parts = [str(row[col]).strip() for col in text_columns if pd.notna(row[col])]
        texts.append(" ".join(p for p in parts if p))

    return texts


def _last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool the last non-padding token — required for Qwen3-Embedding."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def _get_output_paths(base_dir: str, model_name: str = "qwen3_0.6b", features: str = "title_description") -> tuple[str, str]:
    """Return (checkpoint_path, out_dir) under assets/models/ds_v2/."""
    out_dir = os.path.normpath(os.path.join(base_dir, "..", "assets", "models", "test"))
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = os.path.join(out_dir, f"{model_name}_{features}_CHECKPOINT.pkl")
    return checkpoint_path, out_dir


def encode_jobs_incremental(
    df: pd.DataFrame,
    texts: List[str],
    out_file: str,
    model_name: str = MODEL_NAME,
    batch_size: int = BATCH_SIZE,
) -> pd.DataFrame:
    """
    Encode job texts using Qwen3-Embedding-0.6B and save to pickle after each batch.
    Resumes from checkpoint if out_file exists (skips rows already embedded).
    """
    # Check for existing final file (from a previous complete run)
    out_dir = os.path.dirname(out_file)
    base = os.path.splitext(out_file)[0].replace("_CHECKPOINT", "")
    final_matches = glob.glob(os.path.join(out_dir, os.path.basename(base) + "_*d.pkl"))
    for path in final_matches:
        existing = pd.read_pickle(path)
        if len(existing) == len(df) and existing["embedding"].notna().all():
            print(f"Already complete: {path} ({len(df)} rows). Skipping.")
            return existing

    # Load checkpoint or start fresh
    if os.path.exists(out_file):
        checkpoint = pd.read_pickle(out_file)
        if len(checkpoint) != len(df):
            print(f"Checkpoint row count ({len(checkpoint)}) differs from current df ({len(df)}). Starting fresh.")
            checkpoint = df.copy()
            checkpoint["embedding"] = [None] * len(checkpoint)
            n_done = 0
        else:
            n_done = checkpoint["embedding"].notna().sum()
            if n_done >= len(df):
                print(f"Checkpoint complete: all {len(df)} rows already embedded. Done.")
                return checkpoint
            print(f"Resuming: {n_done}/{len(df)} rows already embedded.")
    else:
        checkpoint = df.copy()
        checkpoint["embedding"] = [None] * len(checkpoint)
        n_done = 0

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right", local_files_only=True)
    model = AutoModel.from_pretrained(model_name, dtype=torch.float32, local_files_only=True)
    model.eval()

    for i in range(n_done, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**encoded)

        embeddings = _last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        embs = embeddings.numpy().astype("float32")

        for j, emb in enumerate(embs):
            checkpoint.at[i + j, "embedding"] = emb

        checkpoint.to_pickle(out_file)
        print(f"  Encoded {min(i + batch_size, len(texts))}/{len(texts)} → saved")

    # Save to final file (with dimension in name) and remove checkpoint
    dim = embs.shape[1]
    base = os.path.splitext(out_file)[0].replace("_CHECKPOINT", "")
    final_file = f"{base}_{dim}d.pkl"
    checkpoint.to_pickle(final_file)
    if os.path.exists(out_file) and out_file != final_file:
        os.remove(out_file)
    print(f"Saved to {final_file}")
    return checkpoint


def main() -> None:
    df = load_job_postings_samples_v2()
    print(f"Loaded {len(df)} job postings from job_postings_samples_v2.csv.")

    job_texts = build_job_texts(df)
    print(f"Building embeddings for {len(job_texts)} job texts with {MODEL_NAME}...")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_file, _ = _get_output_paths(base_dir)

    result = encode_jobs_incremental(df, job_texts, out_file)
    n_valid = result["embedding"].notna().sum()
    print(f"Done. Embeddings: {n_valid}/{len(result)} rows.")


def test_model_smoke() -> None:
    """
    Sanity-check that the Qwen3 embedding model is loaded and behaves correctly:

    1. Embedding shape — output dimension matches the expected 1024-d.
    2. Similarity ordering — a query about software engineering should score
       higher against a software-engineering job than against an unrelated one.
    3. FAISS round-trip — the most-similar document retrieved from a tiny
       in-memory index is the expected one.
    """
    import faiss
    import torch.nn.functional as F

    print("=" * 60)
    print("Running model smoke test …")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="right", local_files_only=True)
    model = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.float32, local_files_only=True)
    model.eval()

    texts = [
        # idx 0 — software engineering job
        "Software Engineer: design and implement scalable backend services "
        "using Python, REST APIs, and cloud infrastructure.",
        # idx 1 — unrelated job (finance)
        "Financial Analyst: build financial models, analyze P&L statements, "
        "and prepare quarterly forecasts for senior management.",
        # idx 2 — query that should match idx 0
        "software engineer backend python cloud API development",
    ]

    def _embed(text_list):
        encoded = tokenizer(text_list, padding=True, truncation=False, return_tensors="pt")
        with torch.no_grad():
            out = model(**encoded)
        embs = _last_token_pool(out.last_hidden_state, encoded["attention_mask"])
        return F.normalize(embs, p=2, dim=1).numpy().astype("float32")

    embs = _embed(texts)

    # --- 1. Shape check ---
    expected_dim = 1024
    assert embs.shape == (3, expected_dim), (
        f"Expected shape (3, {expected_dim}), got {embs.shape}"
    )
    print(f"[PASS] Embedding shape: {embs.shape}")

    # --- 2. Similarity ordering ---
    query_emb = embs[2:3]
    sim_se = (query_emb @ embs[0]).item()       # query ↔ software engineering job
    sim_finance = (query_emb @ embs[1]).item()  # query ↔ finance job
    assert sim_se > sim_finance, (
        f"Expected query to be closer to SE job (sim={sim_se:.4f}) "
        f"than finance job (sim={sim_finance:.4f})"
    )
    print(f"[PASS] Similarity ordering: SE job={sim_se:.4f} > finance job={sim_finance:.4f}")

    # --- 3. FAISS round-trip ---
    index = faiss.IndexFlatIP(expected_dim)
    corpus = embs[:2].copy()
    faiss.normalize_L2(corpus)
    index.add(corpus)

    query = query_emb.copy()
    faiss.normalize_L2(query)
    distances, indices = index.search(query, k=1)

    top_idx = int(indices[0][0])
    assert top_idx == 0, (
        f"Expected FAISS to return doc 0 (SE job), got doc {top_idx}"
    )
    print(f"[PASS] FAISS round-trip: top result is doc {top_idx} (SE job, score={distances[0][0]:.4f})")

    print("=" * 60)
    print("All smoke tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    test_model_smoke()