from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
import pandas as pd
from google import genai
from read_data import get_job_postings


# read csv 
df = pd.read_csv('job_postings_samples.csv')

# set seed, shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# remove duplications of title + description + location in df and only remain the first occurrence of job_posting_id
df = df.drop_duplicates(subset=['title', 'description', 'location']).reset_index(drop=True)

# remove rows with null description
df = df.dropna(subset=['description']).reset_index(drop=True)

# remove rows with empty description
df = df[df['description'].str.strip() != ''].reset_index(drop=True)

# ============================================================================
# Summarize job description using Gemini 2.5 Flash Lite
# ============================================================================
# Load Gemini API key
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_llm_path = os.path.join(ROOT, ".env.dev.llm")
load_dotenv(dotenv_llm_path)

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Get the job description from row 8
job_description = df.iloc[3]['description']
print(job_description)

# Create summarization prompt (omit title; description is joined with title column)
prompt = f"""Summarize this job description for semantic search.
Keep: key skills/technologies, responsibilities, domain/industry, seniority.
Drop: marketing/perks/fluff and any job title. 
Return concise sentences.

Job Description:
{job_description}"""

# Generate summary using gemma
response = client.models.generate_content(
    model="gemma-3-27b-it", # it: instruction-tuned
    contents=prompt,
    config=genai.types.GenerateContentConfig(
        max_output_tokens=128,
        temperature=0.3
    )
)

# Extract and print the summary
summary = response.text
print("\n" + "="*80)
print("SUMMARY:")
print("="*80)
print(summary)
print("="*80)




# query = """
#         SELECT job_posting_id, title, description, location 
#         FROM job_posting_raw_data 
#         WHERE description IS NOT NULL 
#         """
# columns = ['job_posting_id', 'title', 'description', 'location']

# job_postings = get_job_postings(query, columns)

# ============================================================================
# Generate embeddings with Gemini
# ============================================================================
# '''
# Method below will need one request per job posting.
# '''


# if not job_postings.empty:
#     # Load Gemini API key
#     BASE_DIR_LLM = os.path.dirname(os.path.abspath(__file__))
#     dotenv_llm_path = os.path.join(BASE_DIR_LLM, '.env.dev.llm')
#     load_dotenv(dotenv_llm_path)
    
#     # Initialize Gemini client
#     client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
#     print("\n" + "="*80)
#     print("Generating embeddings with Gemini...")
#     print("="*80 + "\n")
    
#     # Generate embeddings for each job description
#     embeddings_list = []
#     combination1 = ['description']
#     combination2 = ['title', 'description']
#     combination3 = ['title', 'description', 'location']

#     for idx, row in job_postings.iterrows():
#         try:
#             # Combine title and description
#             job_text = " ".join([row[col] for col in combination2])
            
#             # Generate embedding
#             result = client.models.embed_content(
#                 model="gemini-embedding-001",
#                 contents=job_text,
#                 config=genai.types.EmbedContentConfig(output_dimensionality=768)
#             )
            
#             embedding_values = result.embeddings[0].values
#             embeddings_list.append(embedding_values)
            
#             print(f"Job {idx + 1}: '{row['title']}'")
#             print(f"  Embedding length: {len(embedding_values)}")
#             print(f"  First 10 values: {embedding_values[:10]}")
#             print()
            
#         except Exception as e:
#             print(f"Error generating embedding for job {idx + 1}: {e}")
#             embeddings_list.append(None)
    
#     # Add embeddings to DataFrame
#     job_postings['embedding'] = embeddings_list
    
#     print("\n" + "="*80)
#     print(f"Successfully generated {sum(1 for e in embeddings_list if e is not None)} embeddings")
#     print("="*80)
    
#     # Display DataFrame info
#     print("\nDataFrame shape:", job_postings.shape)
#     print("Columns:", job_postings.columns.tolist())
# else:
#     print("No job postings found to process.")



