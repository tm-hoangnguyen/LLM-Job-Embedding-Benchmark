"""
Ground truth generation for job postings evaluation.

Provides regex-based and LLM-as-a-judge methods to label job postings
into categories. Designed for embedding evaluation (Recall@K, Precision@K).
"""

import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from tqdm import tqdm


class BaseGroundTruthGenerator(ABC):
    """Base class for ground truth generators. Subclass for regex, LLM-as-judge, etc."""

    @abstractmethod
    def generate(self, df: pd.DataFrame, title_col: str = "title") -> dict[str, pd.DataFrame]:
        """
        Generate ground truth subsets from a DataFrame of job postings.

        Args:
            df: DataFrame with job postings.
            title_col: Name of the column containing job titles.

        Returns:
            Dict mapping category names to filtered DataFrames.
        """
        pass


class RegexGroundTruthGenerator(BaseGroundTruthGenerator):
    """
    Generate ground truth subsets using regex patterns on job titles.

    Categories: entry_level, data_business_analytics, finance_analytics, marketing_analytics.
    """

    # Entry-level patterns (non-capturing groups to avoid str.contains warnings)
    ENTRY_TITLE_PATTERN = re.compile(
        r"(?:entry|new grad|junior|early career|trainee|recent grad|\bI\b|\b1\b)",
        flags=re.IGNORECASE,
    )
    EXCLUDE_PATTERN = re.compile(
        r"(?:senior|principal|lead|staff|director|manager|head of|chief|vp|vice president|executive|intern)",
        flags=re.IGNORECASE,
    )

    # Analytics cluster patterns
    DATA_ANALYTICS_PATTERN = re.compile(
        r"\b(?:data analyst|business analyst|business intelligence|business analytics|reporting|operations analyst|analytics|data science)\b",
        flags=re.IGNORECASE,
    )
    FINANCE_ANALYTICS_PATTERN = re.compile(
        r"\b(?:financial analyst|fp&a analyst|treasury analyst|finance analytics|budget analyst|investment analyst)\b",
        flags=re.IGNORECASE,
    )
    MARKETING_ANALYTICS_PATTERN = re.compile(
        r"\b(?:marketing analyst|marketing|product analyst|marketing analytics|growth analyst|campaign analyst|customer insights analyst)\b",
        flags=re.IGNORECASE,
    )

    def filter_entry_level(
        self,
        df: pd.DataFrame,
        title_col: str = "title",
    ) -> pd.DataFrame:
        """
        Filter job postings that appear to be entry-level (excluding interns).

        Matches titles with entry-level terms and excludes senior-level or intern terms.
        """
        mask_match = df[title_col].str.contains(
            self.ENTRY_TITLE_PATTERN, na=False, regex=True
        )
        mask_exclude = ~df[title_col].str.contains(
            self.EXCLUDE_PATTERN, na=False, regex=True
        )
        return df[mask_match & mask_exclude].reset_index(drop=True)

    def filter_data_business_analytics(
        self,
        df: pd.DataFrame,
        title_col: str = "title",
    ) -> pd.DataFrame:
        """Filter job postings matching Data + Business Analytics roles."""
        mask = df[title_col].str.contains(
            self.DATA_ANALYTICS_PATTERN, na=False, regex=True
        )
        return df[mask].reset_index(drop=True)

    def filter_finance_analytics(
        self,
        df: pd.DataFrame,
        title_col: str = "title",
    ) -> pd.DataFrame:
        """Filter job postings matching Finance Analytics roles."""
        mask = df[title_col].str.contains(
            self.FINANCE_ANALYTICS_PATTERN, na=False, regex=True
        )
        return df[mask].reset_index(drop=True)

    def filter_marketing_analytics(
        self,
        df: pd.DataFrame,
        title_col: str = "title",
    ) -> pd.DataFrame:
        """Filter job postings matching Marketing Analytics roles."""
        mask = df[title_col].str.contains(
            self.MARKETING_ANALYTICS_PATTERN, na=False, regex=True
        )
        return df[mask].reset_index(drop=True)

    def generate(
        self,
        df: pd.DataFrame,
        title_col: str = "title",
    ) -> dict[str, pd.DataFrame]:
        """
        Generate all regex-based ground truth subsets.

        Returns:
            Dict mapping category names to filtered DataFrames:
            - "entry_level"
            - "data_business_analytics"
            - "finance_analytics"
            - "marketing_analytics"
        """
        return {
            "entry_level": self.filter_entry_level(df, title_col=title_col),
            "data_business_analytics": self.filter_data_business_analytics(
                df, title_col=title_col
            ),
            "finance_analytics": self.filter_finance_analytics(df, title_col=title_col),
            "marketing_analytics": self.filter_marketing_analytics(
                df, title_col=title_col
            ),
        }



def save_ground_truths(
    ground_truths: dict[str, pd.DataFrame],
    output_dir: str,
    id_col: str = "job_posting_id",
    columns: Optional[list[str]] = None,
) -> None:
    """
    Save ground truth DataFrames to CSV files.

    Args:
        ground_truths: Dict of category name -> DataFrame.
        output_dir: Directory to write CSV files.
        id_col: Column containing job posting IDs.
        columns: Columns to include. Defaults to [id_col, title, description, location].
    """
    os.makedirs(output_dir, exist_ok=True)
    default_cols = [id_col, "title", "description", "location"]
    cols = columns or default_cols

    for name, subdf in ground_truths.items():
        avail = [c for c in cols if c in subdf.columns]
        subdf[avail].to_csv(
            os.path.join(output_dir, f"{name}.csv"),
            index=False,
        )


class LLMGroundTruthGenerator(BaseGroundTruthGenerator):
    """
    LLM-as-a-judge ground truth generator using binary per-category classification.

    For each job posting, calls the model once per category with a structured
    binary prompt (output: 0 or 1). Prompt config is loaded from prompts.yaml
    under models.<model>.tasks.ground_truth_labeling.

    Rate limiting: enforces a minimum interval between API calls to respect
    the free-tier cap (default: 30 req/min for gemma-3-27b-it).

    Checkpointing: writes a CSV after every labeled row so interrupted runs
    can be resumed automatically without re-labeling completed rows.
    """

    DEFAULT_MODEL = "gemma-3-27b-it"
    # Free-tier caps for gemma-3-27b-it: 30 req/min, 15k tokens/min, 14.4k req/day.
    # Each call is ~685 tokens (prompt + title + 1500-char description).
    # multilabel mode: 1 call/row -> 20 req/min ~= 13.7k TPM (safe headroom).
    # per_category mode: 6 calls/row -> same limit applies per individual call.
    REQUESTS_PER_MINUTE = 20
    MAX_RETRIES = 3
    DESCRIPTION_TRUNCATE_CHARS = 1500

    def __init__(
        self,
        prompts_path: str,
        model: str = DEFAULT_MODEL,
        mode: str = "multilabel",
        checkpoint_path: Optional[str] = None,
        requests_per_minute: int = REQUESTS_PER_MINUTE,
        description_col: str = "description",
        id_col: str = "job_posting_id",
    ) -> None:
        """
        Args:
            prompts_path: Path to prompts.yaml.
            model: Gemini/Gemma model ID (must match a key under models in prompts.yaml).
            mode: "multilabel" (1 API call per row, returns JSON) or
                  "per_category" (1 API call per row per category, returns 0/1).
                  "multilabel" is strongly preferred -- 6x fewer API calls.
            checkpoint_path: Optional CSV path to save/resume progress. If the file
                already exists, completed rows are skipped automatically.
            requests_per_minute: API rate limit; controls the minimum sleep between calls.
            description_col: DataFrame column containing job descriptions.
            id_col: DataFrame column containing unique job posting IDs.
        """
        if mode not in ("multilabel", "per_category"):
            raise ValueError(f"mode must be 'multilabel' or 'per_category', got '{mode}'")

        self.model = model
        self.mode = mode
        self.checkpoint_path = checkpoint_path
        self.description_col = description_col
        self.id_col = id_col
        self._min_interval = 60.0 / requests_per_minute

        cfg = self._load_prompt_config(prompts_path)
        self._prompt_template: str = cfg["template"]
        self._temperature: float = cfg["temperature"]
        self._max_output_tokens: int = cfg["max_output_tokens"]
        self.categories: dict[str, str] = cfg["categories"]

        # Load from the script's own directory so the correct .env is found
        # regardless of where the script is invoked from.
        load_dotenv(dotenv_path=Path(__file__).parent / ".env.dev.llm")
        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self._logger = logging.getLogger(self.__class__.__name__)

    # ── Config ──────────────────────────────────────────────────────────────

    def _load_prompt_config(self, prompts_path: str) -> dict:
        """Load prompt config from prompts.yaml based on the selected mode."""
        task_key = (
            "ground_truth_labeling_multilabel"
            if self.mode == "multilabel"
            else "ground_truth_labeling"
        )
        with open(prompts_path, "r") as f:
            config = yaml.safe_load(f)
        prompts = config["models"][self.model]["tasks"][task_key]["prompts"]
        return {
            "template": prompts["template"],
            "temperature": prompts["temperature"],
            "max_output_tokens": prompts["max_output_tokens"],
            "categories": prompts["categories"],
        }

    # ── LLM calls ───────────────────────────────────────────────────────────

    def _call_llm_per_category(self, title: str, description: str, category: str) -> int:
        """One API call for a single (job, category) pair. Returns 0 or 1."""
        prompt = self._prompt_template.format(
            title=title,
            description=str(description)[: self.DESCRIPTION_TRUNCATE_CHARS],
            category=category,
        )
        generate_config = genai_types.GenerateContentConfig(
            temperature=self._temperature,
            max_output_tokens=self._max_output_tokens,
        )
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self._client.models.generate_content(
                    model=self.model, contents=prompt, config=generate_config,
                )
                raw = response.text.strip()
                if raw in ("0", "1"):
                    return int(raw)
                self._logger.warning(
                    "Unexpected output '%s' for category='%s'; defaulting to 0.", raw, category
                )
                return 0
            except Exception as exc:
                wait = 5 * (2 ** attempt)
                self._logger.warning(
                    "API error on attempt %d/%d for category='%s': %s. Retrying in %ds.",
                    attempt + 1, self.MAX_RETRIES, category, exc, wait,
                )
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(wait)
                else:
                    self._logger.error("All retries exhausted for category='%s'. Defaulting to 0.", category)
                    return 0

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        """
        Extract the first JSON object from a free-form text response.

        Handles cases where the model wraps output in markdown code fences
        (```json ... ```) or adds prose before/after the JSON object.
        Returns None if no valid JSON object is found.
        """
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return None

    def _call_llm_multilabel(self, title: str, description: str) -> dict[str, int]:
        """
        One API call for all categories at once. Returns a dict of {category: 0|1}.

        Uses response_mime_type="application/json" for reliable JSON output.
        Missing or unparseable categories default to 0.
        """
        prompt = self._prompt_template.format(
            title=title,
            description=str(description)[: self.DESCRIPTION_TRUNCATE_CHARS],
        )
        # gemma-3-27b-it does not support response_mime_type="application/json";
        # we parse JSON from free-form text and fall back to regex extraction.
        generate_config = genai_types.GenerateContentConfig(
            temperature=self._temperature,
            max_output_tokens=self._max_output_tokens,
        )
        categories = list(self.categories.keys())
        default = {cat: 0 for cat in categories}

        for attempt in range(self.MAX_RETRIES):
            try:
                response = self._client.models.generate_content(
                    model=self.model, contents=prompt, config=generate_config,
                )
                raw = response.text.strip()
                parsed = self._extract_json(raw)
                if parsed is None:
                    self._logger.warning(
                        "No JSON found on attempt %d/%d. Raw: '%s'. Defaulting to 0.",
                        attempt + 1, self.MAX_RETRIES, raw[:120],
                    )
                    return default
                result = {cat: int(bool(parsed.get(cat, 0))) for cat in categories}
                # Enforce internship/entry_level mutual exclusion
                if result.get("internship") == 1:
                    result["entry_level"] = 0
                return result
            except Exception as exc:
                wait = 5 * (2 ** attempt)
                self._logger.warning(
                    "API error on attempt %d/%d: %s. Retrying in %ds.",
                    attempt + 1, self.MAX_RETRIES, exc, wait,
                )
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(wait)
                else:
                    self._logger.error("All retries exhausted. Defaulting to 0 for all categories.")
                    return default

    # ── Core labeling loop ───────────────────────────────────────────────────

    def _label_all(
        self,
        df: pd.DataFrame,
        title_col: str = "title",
    ) -> pd.DataFrame:
        """
        Label every row in df for every category. Returns a DataFrame with
        columns [id_col, category_1, category_2, ...] containing 0/1 labels.

        Loads an existing checkpoint (if set) and skips already-labeled rows,
        then writes to checkpoint after each completed row.
        """
        categories = list(self.categories.keys())

        # Load checkpoint or start fresh
        if self.checkpoint_path and Path(self.checkpoint_path).exists():
            results_df = pd.read_csv(self.checkpoint_path)
            done_ids = set(results_df[self.id_col])
            self._logger.info(
                "Resuming from checkpoint: %d/%d rows already labeled.",
                len(done_ids), len(df),
            )
        else:
            results_df = pd.DataFrame(columns=[self.id_col] + categories)
            done_ids: set = set()

        pending = df[~df[self.id_col].isin(done_ids)].reset_index(drop=True)
        self._logger.info(
            "Labeling %d pending rows x %d categories = %d API calls.",
            len(pending), len(categories), len(pending) * len(categories),
        )

        _last_call_at = 0.0
        api_calls_per_row = 1 if self.mode == "multilabel" else len(categories)
        self._logger.info(
            "Mode: %s | %d API calls per row | ~%d total calls.",
            self.mode, api_calls_per_row, len(pending) * api_calls_per_row,
        )

        for _, row in tqdm(pending.iterrows(), total=len(pending), desc="Labeling"):
            title = str(row[title_col])
            description = str(row[self.description_col])
            result: dict = {self.id_col: row[self.id_col]}

            if self.mode == "multilabel":
                # Single call returns all category labels at once
                wait = self._min_interval - (time.time() - _last_call_at)
                if wait > 0:
                    time.sleep(wait)
                result.update(self._call_llm_multilabel(title, description))
                _last_call_at = time.time()
            else:
                # One call per category
                for category in categories:
                    wait = self._min_interval - (time.time() - _last_call_at)
                    if wait > 0:
                        time.sleep(wait)
                    result[category] = self._call_llm_per_category(title, description, category)
                    _last_call_at = time.time()

            results_df = pd.concat(
                [results_df, pd.DataFrame([result])], ignore_index=True
            )

            if self.checkpoint_path:
                results_df.to_csv(self.checkpoint_path, index=False)

        return results_df

    # ── Public interface ─────────────────────────────────────────────────────

    def generate(
        self,
        df: pd.DataFrame,
        title_col: str = "title",
    ) -> dict[str, pd.DataFrame]:
        """
        Label all job postings and return per-category subsets.

        Returns:
            Dict mapping category name -> DataFrame of rows labeled 1 for that category.
            Columns include all original df columns plus the binary label columns.
        """
        labels_df = self._label_all(df, title_col=title_col)

        # Cast label columns to int (may be float after CSV round-trip)
        for cat in self.categories:
            if cat in labels_df.columns:
                labels_df[cat] = labels_df[cat].fillna(0).astype(int)

        labeled = df.merge(labels_df[[self.id_col] + list(self.categories)], on=self.id_col, how="left")

        return {
            cat: labeled[labeled[cat] == 1].reset_index(drop=True)
            for cat in self.categories
        }


# ── Main: generate LLM ground truths for ds_v2 ──────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    BASE_DIR = Path(__file__).parent
    load_dotenv(dotenv_path=BASE_DIR / ".env.dev.llm")
    SRC_CSV    = BASE_DIR / "assets/datasets/job_postings_samples_v2.csv"
    PROMPTS    = BASE_DIR / "assets/prompts.yaml"
    OUT_DIR    = BASE_DIR / "assets/ground_truths/ds_v2/llm_generated"
    CHECKPOINT = OUT_DIR / "checkpoint_labels.csv"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load & preprocess (same logic as notebook) ───────────────────────────
    df = pd.read_csv(SRC_CSV)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df.drop_duplicates(subset=["title", "description", "location"]).reset_index(drop=True)
    df = df.dropna(subset=["description"]).reset_index(drop=True)
    df = df[df["description"].str.strip() != ""].reset_index(drop=True)
    first_1000 = df[:1000].copy()
    print(f"Dataset: {len(first_1000)} rows after preprocessing")

    # ── Run LLM labeling ─────────────────────────────────────────────────────
    # multilabel: 1 API call/row (~50 min for 1k rows at 20 req/min)
    # per_category: 6 API calls/row (~5 hrs) — only use for accuracy comparison
    generator = LLMGroundTruthGenerator(
        prompts_path=str(PROMPTS),
        mode="multilabel",
        checkpoint_path=str(CHECKPOINT),
    )

    ground_truths = generator.generate(first_1000, title_col="title")

    # ── Save category CSVs ───────────────────────────────────────────────────
    save_ground_truths(
        ground_truths=ground_truths,
        output_dir=str(OUT_DIR),
        id_col="job_posting_id",
        columns=["job_posting_id", "title", "description", "location"],
    )

    # ── Write manifest ───────────────────────────────────────────────────────
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(SRC_CSV.relative_to(BASE_DIR)),
        "model": generator.model,
        "method": f"llm_{generator.mode}",
        "preprocessing": {
            "shuffle_seed": 42,
            "dedup_subset": ["title", "description", "location"],
            "drop_null_description": True,
            "drop_empty_description": True,
            "row_limit": 1000,
        },
        "categories": {
            name: {
                "description": desc,
                "count": len(ground_truths[name]),
                "file": f"{name}.csv",
                "job_posting_ids": ground_truths[name]["job_posting_id"].tolist(),
            }
            for name, desc in generator.categories.items()
        },
    }
    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\nDone. Summary:")
    for name, subdf in ground_truths.items():
        print(f"  {name}: {len(subdf)} jobs → {name}.csv")
    print(f"  manifest.json written")
