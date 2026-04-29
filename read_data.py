import os
import re
import pandas as pd
from bs4 import BeautifulSoup
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Repo root — use `.env` for DB credentials (optional `.env.dev.llm` for local overrides)
_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_ROOT, ".env"))
load_dotenv(os.path.join(_ROOT, ".env.dev.llm"))

# Get database credentials
db_host = os.getenv('DB_HOST', 'localhost')
db_port = os.getenv('DB_PORT', '5432')
db_name = os.getenv('DB_NAME', 'jobcrawler')
db_user = os.getenv('DB_USER', 'jobcrawler')
db_password = os.getenv('DB_PASSWORD', '')

# Create database URL
db_ssl = os.getenv('DB_SSL', 'require')
db_url = f"cockroachdb://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?sslmode={db_ssl}"

engine = create_engine(db_url, echo=False)

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'datasets')

# job_postings is staging table
FULL_QUERY = "SELECT * FROM job_postings"


def get_job_postings(query=FULL_QUERY):
    """
    Retrieve job postings from the database.
    Column names are derived automatically from the query result — no hardcoding needed.

    Args:
        query (str): SQL query string

    Returns:
        pd.DataFrame: Raw DataFrame with all columns returned by the query
    """
    try:
        df = pd.read_sql(sql=text(query), con=engine.connect())
        print(f"Retrieved {len(df)} job postings ({len(df.columns)} columns)\n")
        return df
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()


def deduplicate_and_clean_job_postings(df):
    """
    Removes duplicate job postings and cleans the data:
    - Keeps only the first occurrence of a combination of (title, description)
    - Removes rows where description is null or empty

    Args:
        df (pd.DataFrame): DataFrame containing job postings

    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df = df.drop_duplicates(subset=['title', 'description']).reset_index(drop=True)
    df = df.dropna(subset=['description']).reset_index(drop=True)
    df = df[df['description'].str.strip() != ''].reset_index(drop=True)

    print(f"Deduplicated dataset: {len(df)} rows\n")
    return df




def clean_description(text):
    """
    Convert a raw job description (HTML or markdown) to plain text.

    Pass 1 — HTML stripping (BeautifulSoup):
    - Parses and removes all HTML tags
    - Decodes HTML entities (&rsquo; &amp; &lt; etc.)
    - Preserves block-level spacing via newlines on <br>/<p>/<li>/headings

    Pass 2 — Markdown stripping (regex):
    - Markdown escape sequences (\\- \\& \\* etc.)
    - Bold / italic (** __ * _)
    - ATX and setext headings
    - Horizontal rules
    - Bullet and numbered list markers
    - Markdown links [label](url) → label
    - Inline code `code` → code
    - Excessive blank lines

    Args:
        text (str): Raw description string (HTML, markdown, or plain)

    Returns:
        str: Plain-text description
    """
    if not isinstance(text, str):
        return text

    # --- Pass 1: strip HTML if present ---
    if re.search(r'<[a-zA-Z/][^>]*>', text):
        soup = BeautifulSoup(text, 'html.parser')
        # Insert newlines at block-level boundaries before stripping tags
        for tag in soup.find_all(['br', 'p', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            tag.insert_before('\n')
        text = soup.get_text(separator=' ')

    # --- Pass 2: strip markdown ---

    # Unescape markdown escape sequences: \- \& \* \. \( \) \[ \] etc.
    text = re.sub(r'\\(.)', r'\1', text)

    # Remove setext-style heading underlines (===... or ---...)
    text = re.sub(r'^[=\-]{2,}\s*$', '', text, flags=re.MULTILINE)

    # Remove ATX heading markers (# ## ### ...)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

    # Remove bold/italic: ***text***, **text**, *text*, __text__, _text_
    text = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'_{1,2}(.+?)_{1,2}', r'\1', text, flags=re.DOTALL)

    # Remove markdown links [label](url) → label
    text = re.sub(r'\[([^\]]*)\]\([^\)]*\)', r'\1', text)

    # Remove inline code `code` → code
    text = re.sub(r'`([^`]*)`', r'\1', text)

    # Remove bullet list markers (* - + at line start)
    text = re.sub(r'^\s*[*\-+]\s+', '', text, flags=re.MULTILINE)

    # Remove numbered list markers (1. 2. ...)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

    # Strip each line, then flatten everything into a single space-separated string
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = ' '.join(lines)

    # Collapse any remaining multiple spaces
    text = re.sub(r' {2,}', ' ', text)

    return text.strip()


def plaintext_descriptions(df, save_filename='job_postings_samples_v2.csv'):
    """
    Apply clean_description to the description column and save the result.

    Args:
        df (pd.DataFrame): DataFrame with a 'description' column
        save_filename (str): Output CSV filename under assets/datasets/

    Returns:
        pd.DataFrame: DataFrame with plain-text descriptions
    """
    df = df.copy()
    df['description'] = df['description'].apply(clean_description)

    # Drop rows that became empty after cleaning
    df = df[df['description'].str.strip() != ''].reset_index(drop=True)

    os.makedirs(ASSETS_DIR, exist_ok=True)
    save_path = os.path.join(ASSETS_DIR, save_filename)
    df.to_csv(save_path, index=False)

    print(f"Plaintext dataset: {len(df)} rows → saved to {save_path}\n")
    print(df['description'].iloc[0][:500])

    return df


raw_cache = os.path.join(ASSETS_DIR, 'job_postings_raw.csv')

df_raw = get_job_postings()
if df_raw.empty:
    if os.path.exists(raw_cache):
        print(f"DB unavailable — loading from cache: {raw_cache}\n")
        df_raw = pd.read_csv(raw_cache)
    else:
        print("No data available. Run with DB connection first.")
        df_raw = pd.DataFrame()

if not df_raw.empty:
    df_step2 = deduplicate_and_clean_job_postings(df_raw)
    df_step3 = plaintext_descriptions(df_step2)
    print('Done')

