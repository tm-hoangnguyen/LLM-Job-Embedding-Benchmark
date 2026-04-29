import os
from typing import Any, Optional

import faiss as faiss_lib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from dotenv import load_dotenv
from google import genai
from transformers import AutoModel, AutoTokenizer

# Hugging Face loading:
# - Default: Hub id below; may download if missing (needs network once).
# - Already downloaded: set QWEN_LOCAL_ONLY=1 to load only from HF cache (no Hub calls), or set
#   QWEN_EMBEDDING_MODEL_PATH=/abs/path/to/model (folder with config.json / tokenizer files).
# - Or HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 (cache-only).
# - 8B: optional QWEN_EMBEDDING_8B_MODEL_PATH (local dir) mirrors QWEN_EMBEDDING_MODEL_PATH for 0.6B.

# Loaded (tokenizer, model) keyed by resolved pretrained ``src`` (Hub id or local dir).
_qwen_cache: dict[str, tuple[Any, Any]] = {}

# Hub repo ids (used for YAML ``embedding*.model_name`` matching). Override paths via env.
QWEN_EMBEDDING_06B_MODEL_NAME = "qwen/Qwen3-Embedding-0.6B"
QWEN_EMBEDDING_8B_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"

# Per-category YAML keys for *query* embeddings (distinct from job ``embedding`` column in DataFrames).
YAML_QUERY_EMBEDDING_KEY_QWEN = "embedding1"
YAML_QUERY_EMBEDDING_KEY_QWEN_8B = "embedding2"

# Job vector widths for Qwen checkpoints used in this project (must match FAISS index ``self.dimension``).
QWEN_JOB_EMBEDDING_DIM_0_6B = 1024
QWEN_JOB_EMBEDDING_DIM_8B = 4096


def _normalize_embedding_model_id(name: Optional[str]) -> str:
    """Compare HF ids loosely (org/repo vs repo-only, case, underscores)."""
    if not name:
        return ""
    s = str(name).strip().lower().replace("_", "-")
    if "/" in s:
        s = s.split("/", 1)[1]
    return s


def _embedding_model_names_match(a: Optional[str], b: Optional[str]) -> bool:
    return _normalize_embedding_model_id(a) == _normalize_embedding_model_id(b)


def _hf_local_files_only() -> bool:
    """True if Hugging Face libraries must not hit the network (cache-only load)."""
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        v = os.getenv(key, "").lower()
        if v in ("1", "true", "yes"):
            return True
    return False


def _qwen_local_only_preferred() -> bool:
    """Use only local files (HF cache or QWEN_EMBEDDING_MODEL_PATH); never contact the Hub."""
    v = os.getenv("QWEN_LOCAL_ONLY", "").lower()
    if v in ("1", "true", "yes"):
        return True
    return _hf_local_files_only()


def _qwen_pretrained_id_or_path() -> tuple[str, bool]:
    """
    Return (model_id_or_local_dir, local_files_only).

    If QWEN_EMBEDDING_MODEL_PATH points to an existing directory, load from there (always local).
    Otherwise use QWEN_EMBEDDING_06B_MODEL_NAME; local_files_only follows QWEN_LOCAL_ONLY / offline env.
    """
    raw = os.getenv("QWEN_EMBEDDING_MODEL_PATH", "").strip()
    if raw:
        expanded = os.path.expanduser(raw)
        if os.path.isdir(expanded):
            return expanded, True
        raise FileNotFoundError(
            f"QWEN_EMBEDDING_MODEL_PATH is set but not a directory: {expanded!r}"
        )
    return QWEN_EMBEDDING_06B_MODEL_NAME, _qwen_local_only_preferred()


def _qwen_8b_pretrained_id_or_path() -> tuple[str, bool]:
    """Like ``_qwen_pretrained_id_or_path`` but for the 8B checkpoint (optional local override)."""
    raw = os.getenv("QWEN_EMBEDDING_8B_MODEL_PATH", "").strip()
    if raw:
        expanded = os.path.expanduser(raw)
        if os.path.isdir(expanded):
            return expanded, True
        raise FileNotFoundError(
            f"QWEN_EMBEDDING_8B_MODEL_PATH is set but not a directory: {expanded!r}"
        )
    return QWEN_EMBEDDING_8B_MODEL_NAME, _qwen_local_only_preferred()


def _resolve_pretrained_src_for_model(canonical_hub_id: str) -> tuple[str, bool]:
    """Map a canonical Hub id to ``from_pretrained`` source and local_files_only flag."""
    if _embedding_model_names_match(canonical_hub_id, QWEN_EMBEDDING_06B_MODEL_NAME):
        return _qwen_pretrained_id_or_path()
    if _embedding_model_names_match(canonical_hub_id, QWEN_EMBEDDING_8B_MODEL_NAME):
        return _qwen_8b_pretrained_id_or_path()
    cid = (canonical_hub_id or "").strip()
    if not cid:
        return _qwen_pretrained_id_or_path()
    return cid, _qwen_local_only_preferred()


def _resolve_qwen_yaml_and_pretrained_from_index_dimension(index_dimension: int) -> tuple[str, str]:
    """
    Map FAISS job-embedding width to category_map slot + HF id.

    The index dimension comes from the job ``embedding`` column (e.g. whichever pickle you loaded
    into ``FAISSIndex``); it must match the query vector we build or load from YAML.
    """
    if index_dimension == QWEN_JOB_EMBEDDING_DIM_8B:
        return YAML_QUERY_EMBEDDING_KEY_QWEN_8B, QWEN_EMBEDDING_8B_MODEL_NAME
    if index_dimension == QWEN_JOB_EMBEDDING_DIM_0_6B:
        return YAML_QUERY_EMBEDDING_KEY_QWEN, QWEN_EMBEDDING_06B_MODEL_NAME
    raise ValueError(
        f"Unsupported Qwen index dimension {index_dimension}: expected "
        f"{QWEN_JOB_EMBEDDING_DIM_0_6B} (0.6B → {YAML_QUERY_EMBEDDING_KEY_QWEN}) or "
        f"{QWEN_JOB_EMBEDDING_DIM_8B} (8B → {YAML_QUERY_EMBEDDING_KEY_QWEN_8B})."
    )


def _try_load_cached_qwen_query_embedding(
    categories_cfg: dict,
    category: str,
    query_text: str,
    index_dimension: int,
    yaml_embedding_key: str,
    expected_model_name: str,
) -> Optional[np.ndarray]:
    """
    Return (1, D) float32 query vector from category_map if the stored query, YAML slot, model,
    and dimension match; otherwise None (caller should run Qwen).
    """
    if category not in categories_cfg:
        return None

    stored_query = categories_cfg[category].get("query") or ""
    if stored_query.strip() != (query_text or "").strip():
        return None

    emb_block: Any = categories_cfg[category].get(yaml_embedding_key)
    if not isinstance(emb_block, dict):
        return None

    if not _embedding_model_names_match(
        emb_block.get("model_name"), expected_model_name
    ):
        return None

    vec = emb_block.get("vector")
    if vec is None:
        return None
    arr = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    if arr.shape[1] != index_dimension:
        return None
    if arr.size == 0:
        return None
    return arr


def _get_qwen_model(canonical_hub_id: str):
    """Load Qwen tokenizer and model for the given canonical id; cache per resolved pretrained src."""
    src, local_only = _resolve_pretrained_src_for_model(canonical_hub_id)
    if src not in _qwen_cache:
        tokenizer = AutoTokenizer.from_pretrained(
            src,
            padding_side="right",
            local_files_only=local_only,
        )
        model = AutoModel.from_pretrained(
            src,
            dtype=torch.float32,
            local_files_only=local_only,
        )
        model.eval()
        _qwen_cache[src] = (tokenizer, model)
    return _qwen_cache[src]


class FAISSIndex:
    def __init__(self, df, query_text=None):
        self.df = df
        self.embed_col = np.array(df['embedding'].tolist()).astype('float32')
        faiss_lib.normalize_L2(self.embed_col)  
        self.dimension = self.embed_col.shape[1]
        self.index = None
        self.index_to_job_id = {}

    def build_index(self):
        # Build index with cosine similarity
        self.index = faiss_lib.IndexFlatIP(self.dimension)
        self.index.add(self.embed_col)

        # Create mapping from job_posting_id to index
        self.index_to_job_id = self.df['job_posting_id'].tolist()

        # print(f"Embeddings shape: {self.embed_col.shape}")
        # print(f"Number of job postings: {len(self.df)}")

    def search_gemini(self, category, query_text, top_k=10):
        """
        Search for the most similar job postings given a query text.
        
        Args:
            query_text (str): The query text (e.g., title + description)
            top_k (int): Number of top similar jobs to return (default: 10)
        
        Returns:
            list: List of dictionaries containing the top_k most similar job postings with similarity scores
        """
        # Load Gemini API key
        load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env.dev.llm"))
        
        # Initialize Gemini client
        gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
        # Generate embedding for query text (match dimension with index)
        result = gemini_client.models.embed_content(
            model="gemini-embedding-001",
            contents=[query_text],
            config=genai.types.EmbedContentConfig(output_dimensionality=self.dimension)
        )
        
        # Extract query embedding
        query_embedding = np.array(result.embeddings[0].values).astype('float32').reshape(1, -1)
        
        # Normalize query embedding for cosine similarity
        faiss_lib.normalize_L2(query_embedding)
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Convert FAISS indices to job_posting_ids and retrieve metadata
        results = []
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            job_id = self.index_to_job_id[idx]
            job_data = self.df[self.df['job_posting_id'] == job_id].iloc[0]
            
            results.append(job_data.to_dict())
            results[-1]['similarity_score'] = float(distance)
        
        return pd.DataFrame(results)


    def search_qwen(self, model_name, category, query_text, top_k=10):
        """
        Search for the most similar job postings given a query text using a Qwen3 embedding model.

        If ``category_map.yaml`` has a valid cached vector for this category (matching YAML
        ``query``, the slot implied by the index dimension, stored model id in that slot, and index
        dimension), it is reused and the Qwen model is not run.

        Args:
            model_name: Ignored for YAML/HF routing (callers may pass any placeholder). Slot and
                checkpoint follow ``self.dimension`` (job vectors in the index): 1024 →
                ``embedding1`` / 0.6B, 4096 → ``embedding2`` / 8B.
            category (str): Key under category_map.yaml ``categories``.
            query_text (str): Must equal YAML ``query`` (after strip) for that category to persist
                under the active slot. If that slot's ``model_name`` is set to another encoder,
                the file is not overwritten.
            top_k (int): Number of top similar jobs to return (default: 10)

        Returns:
            pd.DataFrame: DataFrame containing top_k most similar job postings with similarity scores
        """
        category_map_path = os.path.join(
            os.path.dirname(__file__), "assets", "category_map.yaml"
        )
        with open(category_map_path, "r", encoding="utf-8") as f:
            category_map = yaml.safe_load(f)

        categories_cfg = category_map.get("categories", {})
        if category not in categories_cfg:
            raise KeyError(
                f"Unknown category {category!r}; expected one of {list(categories_cfg)}"
            )

        _ = model_name  # API compatibility; YAML/HF choice follows job embedding width only.

        yaml_key, pretrained_for_gen = _resolve_qwen_yaml_and_pretrained_from_index_dimension(
            self.dimension
        )

        # Load cached query embedding if it exists
        query_embedding = _try_load_cached_qwen_query_embedding(
            categories_cfg,
            category,
            query_text,
            self.dimension,
            yaml_key,
            pretrained_for_gen,
        )

        # If no cached query embedding, generate a new one (only if the query text matches the YAML query)
        if query_embedding is None:
            print("Generating new query embedding")
            def _last_token_pool(last_hidden_states, attention_mask):
                left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
                if left_padding:
                    return last_hidden_states[:, -1]
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_states.shape[0]
                return last_hidden_states[
                    torch.arange(batch_size, device=last_hidden_states.device),
                    sequence_lengths,
                ]

            tokenizer, model = _get_qwen_model(pretrained_for_gen)

            encoded = tokenizer(
                [query_text],
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = model(**encoded)

            query_embedding = _last_token_pool(
                outputs.last_hidden_state, encoded["attention_mask"]
            )
            query_embedding = F.normalize(query_embedding, p=2, dim=1)
            query_embedding = query_embedding.numpy().astype("float32")

            stored_query = categories_cfg[category].get("query") or ""
            query_matches_yaml = stored_query.strip() == (query_text or "").strip()

            emb_block: Any = categories_cfg[category].get(yaml_key)
            if isinstance(emb_block, dict):
                yaml_model = emb_block.get("model_name")
                yaml_has_other_model = bool(yaml_model) and not _embedding_model_names_match(
                    yaml_model, pretrained_for_gen
                )
            else:
                yaml_has_other_model = False

            should_persist = query_matches_yaml and not yaml_has_other_model

            if should_persist:
                categories_cfg[category][yaml_key] = {
                    "model_name": pretrained_for_gen,
                    "vector": query_embedding.squeeze().tolist(),
                }
                with open(category_map_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(
                        category_map,
                        f,
                        default_flow_style=False,
                        allow_unicode=True,
                        sort_keys=False,
                    )
                print(
                    f"Cached query embedding for category {category} "
                    f"under {yaml_key} with model {pretrained_for_gen}"
                )

        faiss_lib.normalize_L2(query_embedding)

        distances, indices = self.index.search(query_embedding, top_k)

        valid_mask = (indices[0] >= 0) & (indices[0] < len(self.index_to_job_id))
        valid_indices = indices[0][valid_mask]
        valid_distances = distances[0][valid_mask]

        job_ids = [self.index_to_job_id[i] for i in valid_indices]

        id_to_row = self.df.set_index('job_posting_id')
        results_df = id_to_row.loc[job_ids].reset_index()
        results_df['similarity_score'] = valid_distances.tolist()

        return results_df