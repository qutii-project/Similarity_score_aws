import os
import json
import numpy as np
import boto3
# import redis  # Commented out since Redis is not used at the moment
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Load environment variables
load_dotenv(r"C:\Projects\Similarity\secrets.env")

# AWS S3 Configuration
s3_bucket = os.getenv('S3_BUCKET')

# Initialize AWS S3 client
s3_client = boto3.client("s3")

# Commented out Redis initialization
# redis_host = os.getenv('REDIS_HOST')
# redis_port = os.getenv('REDIS_PORT')
# redis_cache = redis.StrictRedis(host=redis_host, port=redis_port, decode_responses=True)

def fetch_embeddings_from_s3(subtopic_id=None):
    """Fetch embeddings from S3, optionally filtered by subtopic."""
    try:
        print(f"Fetching embeddings from S3...")
        
        # If subtopic_id is provided, generate the file name for that subtopic
        s3_key = f"{subtopic_id}_embeddings.json" if subtopic_id else "all_embeddings.json"
        
        # Fetch the embeddings file
        response = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
        
        # Read and decode the embeddings data
        embeddings_data = json.loads(response['Body'].read().decode('utf-8'))
        
        return embeddings_data
    except Exception as e:
        print(f"Error fetching embeddings from S3: {e}")
        return None
    
def predict(selected_qna_id, top_n=5):
    """
    Predict the most similar QnA pairs for a given selected QnA ID.
    
    Args:
        selected_qna_id (str): The QnA ID selected by the user.
        top_n (int): Number of top similar QnA pairs to return.
        
    Returns:
        str: JSON-formatted result of the top similar QnA pairs.
    """
    # Step 1: Fetch all embeddings data (for all subtopics or a pre-filtered subset)
    embeddings_data = fetch_embeddings_from_s3()  # Fetch all embeddings
    if not embeddings_data:
        return json.dumps({"error": "Failed to retrieve embeddings."})
    
    # Step 2: Find the selected QnA and its subtopic
    selected_qna = next((item for item in embeddings_data if item["qna_id"] == selected_qna_id), None)
    if not selected_qna:
        return json.dumps({"error": "Selected QnA ID not found."})
    
    selected_subtopic = selected_qna["subtopic"]  # Extract the subtopic ID
    selected_embedding = np.array(selected_qna["QA_embedding"])
    
    # Step 3: Filter embeddings to get only those from the same subtopic
    same_subtopic_embeddings = [
        (item["qna_id"], item["question"], item["answer"], np.array(item["QA_embedding"]))
        for item in embeddings_data if item["subtopic"] == selected_subtopic and item["qna_id"] != selected_qna_id
    ]
    
    if not same_subtopic_embeddings:
        return json.dumps({"error": "No other QnA pairs found in the same subtopic."})
    
    # Step 4: Calculate similarity with the selected QnA
    all_other_embeddings = [embedding[-1] for embedding in same_subtopic_embeddings]  # Extract embeddings only
    similarities = cosine_similarity([selected_embedding], all_other_embeddings)[0]
    
    # Step 5: Get top N results
    top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_n]
    top_results = [
        {
            "qna_id": same_subtopic_embeddings[idx][0],
            "question": same_subtopic_embeddings[idx][1],
            "answer": same_subtopic_embeddings[idx][2],
            "similarity_score": similarities[idx]
        }
        for idx in top_indices
    ]
    
    return json.dumps(top_results)


if __name__ == "__main__":
    # Example usage
    selected_qna_id = "130122"  # Replace with actual QnA ID
    #subtopic = "130122"  # Replace with actual subtopic ID
    result = predict(selected_qna_id)
    print(result)
