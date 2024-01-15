from flask import jsonify
from face_models import load_face_models, get_face_embedding
from db_operations import save_embedding
import requests
import json

BLOCKCHAIN_API_URL = 'http://103.39.218.177:8088/api/v1/governance/save-data'

# Load face models
mtcnn, resnet = load_face_models()

def feature_extraction(id):
    """
    Extracts facial features from the image.
    """
    try:
        file_path = f"./temporary/{id}/uploaded_videos/closed_mouth_frame.jpg"
        with open(file_path, 'rb') as image_file:
            status, embedding, _ = get_face_embedding(image_file.read(), mtcnn, resnet)
            return embedding if status not in ["no_face", "low_quality", "image_error"] else None
    except IOError as e:
        print(f"Error reading file: {e}")
        return None

def save_to_database(id, img_aligned):
    """
    Saves the aligned image feature to the database.
    """
    if img_aligned is None:
        return False

    img_aligned_list = img_aligned.tolist()
    return save_embedding(id, json.dumps(img_aligned_list))

def save_to_blockchain(id, embedding):
    """
    Saves the feature array to a blockchain.
    """
    if embedding is None:
        return False

    headers = {'Content-Type': 'application/json'}
    data = {'key': id, 'value': json.dumps(embedding.tolist())}
    try:
        response = requests.post(BLOCKCHAIN_API_URL, json=data, headers=headers)
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return False

def embedding_feature_saving(id):
    """
    Main function to handle the feature extraction and saving process.
    """
    embedding = feature_extraction(id)
    if embedding is None:
        return jsonify({"error": "Face extraction failed"}), 400

    db_status = save_to_database(id, embedding)
    blockchain_status = save_to_blockchain(id, embedding)

    return jsonify({"database_status": db_status, "blockchain_status": blockchain_status})


