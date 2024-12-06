import os
import json
import boto3
import numpy as np
import pandas as pd
import psycopg2
from datetime import datetime, timedelta
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

# AWS S3 Configuration
s3_bucket = os.getenv('S3_BUCKET')
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_name = os.getenv('DB_NAME')
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')

# Initialize AWS S3 client
s3_client = boto3.client("s3")

# Refresh interval (e.g., 7 days)
FULL_REFRESH_INTERVAL = timedelta(days=7)
LAST_REFRESH_KEY = "last_full_refresh"

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

def load_embeddings_from_s3(subtopic):
    """Load existing embeddings from S3 for a subtopic."""
    try:
        response = s3_client.get_object(Bucket=s3_bucket, Key=f"{subtopic}_embeddings.json")
        embeddings_data = json.loads(response['Body'].read().decode('utf-8'))
        return embeddings_data
    except s3_client.exceptions.NoSuchKey:
        print(f"No existing embeddings found for subtopic {subtopic}.")
        return []
    except Exception as e:
        print(f"Error loading embeddings from S3: {e}")
        return []

def save_embeddings_to_s3(subtopic, embeddings):
    """Save embeddings to S3 for a subtopic."""
    try:
        # Convert embeddings to JSON-serializable format
        for item in embeddings:
            if isinstance(item['QA_embedding'], np.ndarray):
                item['QA_embedding'] = item['QA_embedding'].tolist()
        
        embeddings_json = json.dumps(embeddings)
        s3_client.put_object(
            Bucket=s3_bucket,
            Key=f"{subtopic}_embeddings.json",
            Body=embeddings_json
        )
        # Redis caching commented out
        # redis_cache.set(f"embeddings:subtopic:{subtopic}", embeddings_json)
    except Exception as e:
        print(f"Error saving embeddings to S3: {e}")

def merge_embeddings(existing, new_data):
    """Merge new embeddings into existing ones."""
    existing_map = {item['qna_id']: item for item in existing}
    for item in new_data:
        existing_map[item['qna_id']] = item  # Replace or add new data
    return list(existing_map.values())

def perform_full_refresh(df):
    """Perform a full refresh of embeddings for all subtopics."""
    grouped = df.groupby("subtopic")
    for subtopic, group in grouped:
        embeddings = group[["qna_id", "question", "answer", "QA"]].to_dict(orient="records")
        for item in embeddings:
            item["subtopic"] = subtopic  # Add the subtopic ID
            item["QA_embedding"] = text_embed(preprocess_text(item["QA"]))
        save_embeddings_to_s3(subtopic, embeddings)
    #print("Full refresh completed.")

def perform_incremental_update(df):
    """Perform an incremental update of embeddings."""
    grouped = df.groupby("subtopic")
    for subtopic, group in grouped:
        new_data = group[["qna_id", "question", "answer", "QA"]].to_dict(orient="records")
        for item in new_data:
            item["subtopic"] = subtopic  # Add the subtopic ID
            item["QA_embedding"] = text_embed(preprocess_text(item["QA"]))
        
        existing_data = load_embeddings_from_s3(subtopic)
        merged_data = merge_embeddings(existing_data, new_data)
        save_embeddings_to_s3(subtopic, merged_data)
    #print("Incremental update completed.")

def generate_and_store_embeddings():
    """Generate embeddings for QnA data and decide between full refresh or incremental update."""
    df = load_dataset_from_postgres()
    if df is None:
        print("Data loading failed.")
        return
    
    # Redis fetch commented out
    # last_refresh = redis_cache.get(LAST_REFRESH_KEY)
    last_refresh = None
    if last_refresh:
        last_refresh = datetime.fromisoformat(last_refresh)
        if datetime.utcnow() - last_refresh < FULL_REFRESH_INTERVAL:
            perform_incremental_update(df)
        else:
            perform_full_refresh(df)
    else:
        perform_full_refresh(df)

if __name__ == "__main__":
    generate_and_store_embeddings()
    print("Embedding generation and storage process completed.")
