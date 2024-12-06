import os
import json
import boto3
import redis
import numpy as np
import pandas as pd
import psycopg2
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv
import re
import string
import nltk

# Load the environment variables from 'secrets.env'
load_dotenv(r"C:\Projects\Similarity\secrets.env") 

# Download NLTK resources if not present
nltk.download("punkt")
nltk.download("stopwords")

# Initialize resources
MODEL_NAME = "thenlper/gte-small"
MODEL = SentenceTransformer(MODEL_NAME)
STOP_WORDS = set(stopwords.words('english'))

# AWS S3 and Redis Configuration
s3_bucket = os.getenv('S3_BUCKET')
redis_host = os.getenv('REDIS_HOST')
redis_port = os.getenv('REDIS_PORT')
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_name = os.getenv('DB_NAME')
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')

# Initialize AWS S3 and Redis clients
s3_client = boto3.client("s3")
redis_cache = redis.StrictRedis(host=redis_host, port=redis_port, decode_responses=True)

def connect_to_postgres():
    """Connect to the PostgreSQL database and retrieve data."""
    try:
        return psycopg2.connect(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password
        )
    except psycopg2.Error as e:
        print(f"Database connection error: {e}")
        return None

def load_dataset_from_postgres():
    """Load dataset from PostgreSQL and return as a DataFrame."""
    conn = connect_to_postgres()
    if conn is None:
        print("Failed to connect to PostgreSQL database.")
        return None

    query = """
        SELECT qa.qna_id, st.id AS subtopic, st.name, qa.question, qa.answer
        FROM articlegeneratedqna AS qa
        JOIN qnasubtopic AS st ON qa.qnasubtopicid = st.id;
    """
    try:
        df = pd.read_sql_query(query, conn)
        conn.close()
        # Concatenate question and answer for embedding generation
        df['QA'] = df['question'] + ' ' + df['answer']
        return df
    except Exception as e:
        print(f"SQL query execution error: {e}")
        return None

def preprocess_text(text):
    """Convert text to lowercase, remove unnecessary characters, and tokenize."""
    text = text.lower()
    text = re.sub(r'[\d' + re.escape(string.punctuation) + '](?<![cC][oO]2)', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in STOP_WORDS]
    return ' '.join(tokens)

def text_embed(text):
    """Generate embedding for preprocessed text using SentenceTransformer."""
    return MODEL.encode(text)

def save_embeddings_to_s3(subtopic, embeddings, max_retries=1):
    """Save embeddings to S3 as a JSON file per subtopic."""
    retries = 0
    while retries < max_retries:
        try:
            # Convert embeddings to list if they are numpy arrays
            for item in embeddings:
                if isinstance(item['QA_embedding'], np.ndarray):
                    item['QA_embedding'] = item['QA_embedding'].tolist()
            
            filename = f"{subtopic}_embeddings.json"
            embeddings_json = json.dumps(embeddings)
            
            # Save to S3
            s3_client.put_object(
                Bucket=s3_bucket,
                Key=filename,
                Body=embeddings_json
            )
        
            # Cache in Redis for efficient access
            cache_key = f"embeddings:subtopic:{subtopic}"
            redis_cache.set(cache_key, embeddings_json)
        
        except Exception as e:
            print(f"Error saving embeddings to S3 or Redis: {e}")
            retries += 1
            if retries >= max_retries:
                print("Max retries reached. Aborting process.")
                break  # Break the loop after max retries

def generate_and_store_embeddings():
    """Generate embeddings for each QnA and store them by subtopic in S3 and Redis."""
    df = load_dataset_from_postgres()
    if df is None:
        print("Data loading failed.")
        return

    df["QA_prep"] = df["QA"].apply(preprocess_text)
    df["QA_embedding"] = df["QA_prep"].apply(text_embed)

    # Group embeddings by subtopic
    grouped = df.groupby("subtopic")
    
    for subtopic, group in grouped:
        embeddings = group[["qna_id", "question", "answer", "QA_embedding"]].to_dict(orient="records")
        save_embeddings_to_s3(subtopic, embeddings)

if __name__ == "__main__":
    generate_and_store_embeddings()
    print("Embedding generation and storage complete.")