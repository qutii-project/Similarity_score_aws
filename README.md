Overview:

The project implements a system that calculates the similarity between Question-and-Answer (QnA) pairs based on pre-generated embeddings. The embeddings are stored in AWS S3, and similarity comparisons are made using these embeddings. The system fetches embeddings for a given QnA's subtopic, computes cosine similarity between the selected QnA and others within the same subtopic, and returns the top N most similar QnA pairs.

Project Structure:

The project consists of the following main components:

generate_embs_final.py: Script to generate and store embeddings for QnA pairs, including pre-processing, embedding generation, and storing them in S3.
similarity_analysis_no_redis.py: Script to analyze the similarity between a selected QnA pair and others within the same subtopic.
requirements.txt: Contains all the dependencies required for the project.
secrets.env: Environment file containing database credentials, S3 keys, etc.

Sample Output: 

The predict function will return a JSON-formatted list of top N similar QnA pairs, sorted by similarity score:

[

    {
        "qna_id": 4161,
        "question": "What are pull measures in the context of mobility behaviour?",
        "answer": "Pull measures are strategies that attract individuals towards more sustainable modes of transportation.",
        "similarity_score": 0.8813700619099212
    },
    
]
