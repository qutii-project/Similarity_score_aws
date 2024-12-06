import os
import json
import numpy as np
import boto3
import psycopg2
# import redis  # Commented out since Redis is not used at the moment
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Load environment variables
load_dotenv(r"C:\Projects\Similarity\secrets.env")

# AWS S3 Configuration
s3_bucket = os.getenv('S3_BUCKET')
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_name = os.getenv('DB_NAME')
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')

# Initialize AWS S3 client
s3_client = boto3.client("s3")

# Commented out Redis initialization
# redis_host = os.getenv('REDIS_HOST')
# redis_port = os.getenv('REDIS_PORT')
# redis_cache = redis.StrictRedis(host=redis_host, port=redis_port, decode_responses=True)

def connect_to_postgres():
    """Establish a connection to the PostgreSQL database."""
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

def get_subtopic_for_qna(selected_qna_id):
    """
    Identify the subtopic ID associated with the selected QnA ID.

    Args:
        selected_qna_id (str): The QnA ID selected by the user.

    Returns:
        str: The subtopic ID associated with the QnA ID.
    """
    try:
        print(f"Fetching subtopic ID for QnA ID: {selected_qna_id}...")

        # Ensure valid input type
        if not isinstance(selected_qna_id, (str, int)):
            raise ValueError("selected_qna_id must be a string or integer.")
        
        # Connect to the database
        conn = connect_to_postgres()
        if conn is None:
            raise ValueError("Database connection failed.")
        
        # Query to fetch the subtopic ID for the given QnA ID
        query = """
            SELECT st.id AS subtopic
            FROM articlegeneratedqna AS qa
            JOIN qnasubtopic AS st ON qa.qnasubtopicid = st.id
            WHERE qa.qna_id = %s;
        """
        
        # Execute the query using a context manager
        with conn.cursor() as cur:
            cur.execute(query, (selected_qna_id,))
            result = cur.fetchone()
        
        conn.close()
        
        # Ensure a subtopic ID is returned
        if result:
            subtopic_id = result[0]
            print(f"Subtopic ID for QnA ID {selected_qna_id}: {subtopic_id}")
            return subtopic_id
        else:
            print(f"No subtopic found for QnA ID: {selected_qna_id}.")
            return None
    except Exception as e:
        print(f"Error fetching subtopic for QnA ID {selected_qna_id}: {e}")
        return None



def fetch_embeddings_from_s3(subtopic_id):
    """
    Fetch embeddings for a specific subtopic from S3.
    
    Args:
        subtopic_id (str): The subtopic ID to fetch embeddings for.
    
    Returns:
        list: Embeddings data as a list of dictionaries.
    """
    try:
        print(f"Fetching embeddings for subtopic ID {subtopic_id} from S3...")
        s3_key = f"{subtopic_id}_embeddings.json"
        response = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
        return json.loads(response['Body'].read().decode('utf-8'))
    except s3_client.exceptions.NoSuchKey:
        print(f"No embeddings found for subtopic ID {subtopic_id}.")
        return []
    except Exception as e:
        print(f"Error fetching embeddings for subtopic ID {subtopic_id}: {e}")
        return []


def predict(selected_qna_id, top_n=5):
    """
    Predict the most similar QnA pairs for a given selected QnA ID.

    Args:
        selected_qna_id (str): The QnA ID selected by the user.
        top_n (int): Number of top similar QnA pairs to return.

    Returns:
        str: JSON-formatted result of the top similar QnA pairs.
    """
    # Identify the subtopic
    subtopic_id = get_subtopic_for_qna(selected_qna_id)
    if not subtopic_id:
        return json.dumps({"error": "Failed to identify subtopic for the selected QnA."})

    # Fetch embeddings
    embeddings_data = fetch_embeddings_from_s3(subtopic_id)
    if not embeddings_data:
        return json.dumps({"error": f"Failed to retrieve embeddings for subtopic {subtopic_id}."})

    # Locate selected QnA embedding
    selected_qna = next((item for item in embeddings_data if item["qna_id"] == selected_qna_id), None)
    if not selected_qna:
        return json.dumps({"error": "Selected QnA ID not found in the subtopic embeddings."})

    selected_embedding = np.array(selected_qna["QA_embedding"])

    # Exclude the selected QnA
    unique_other_embeddings = {
        item["qna_id"]: (item["question"], item["answer"], np.array(item["QA_embedding"]))
        for item in embeddings_data if item["qna_id"] != selected_qna_id
    }

    if not unique_other_embeddings:
        return json.dumps({"error": "No other QnA pairs found in the same subtopic."})

    # Calculate similarity
    all_other_embeddings = np.array([data[-1] for data in unique_other_embeddings.values()])
    similarities = cosine_similarity([selected_embedding], all_other_embeddings)[0]

    # Get top N results
    top_indices = np.argsort(-similarities)[:top_n]
    top_results = [
        {
            "qna_id": qna_id,
            "question": unique_other_embeddings[qna_id][0],
            "answer": unique_other_embeddings[qna_id][1],
            "similarity_score": similarities[idx]
        }
        for idx, qna_id in enumerate(unique_other_embeddings.keys()) if idx in top_indices
    ]

    # Sort results by similarity_score in descending order
    sorted_results = sorted(top_results, key=lambda x: x["similarity_score"], reverse=True)


    return json.dumps(sorted_results[:top_n], indent=4)




if __name__ == "__main__":
    # Example usage
    selected_qna_id = 4165  
    result = predict(selected_qna_id)
    print(result)
