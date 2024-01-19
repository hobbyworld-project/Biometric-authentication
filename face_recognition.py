import numpy as np
from numpy.linalg import norm
from db_operations import get_embedding, get_all_embeddings
from feature_saving import feature_extraction

# Threshold constants for verification and identification
VERIFICATION_THRESHOLD = 0.7
IDENTIFICATION_THRESHOLD = 0.7
TOP_N_MATCHES = 5  # Number of top similar faces to return

def cosine_similarity(vec_a, vec_b):
    """Calculate the cosine similarity between two vectors."""
    return np.dot(vec_a, vec_b) / (norm(vec_a) * norm(vec_b))

def string_to_array(s):
    """Convert a string representation of an array into a numpy array."""
    return np.fromstring(s.strip('[]'), sep=', ')

def face_verification(id):
    """
    Verify if the face corresponding to the provided id matches the face in the system.
    
    Args:
        id (str): The id of the face to verify.
    
    Returns:
        bool: True if the face is verified, False otherwise.
    """
    target_embedding = string_to_array(get_embedding(id))
    extracted_features = feature_extraction(id)
    
    # Calculate cosine similarity
    similarity = cosine_similarity(target_embedding, extracted_features)
    
    # Compare the similarity with the threshold
    return similarity >= VERIFICATION_THRESHOLD

def face_identification(id):
    """
    Identify the most similar faces in the system for the face corresponding to the provided id.
    
    Args:
        id (str): The id of the face to identify.
    
    Returns:
        list: The ids of the top N most similar faces, or None if no match is found within the threshold.
    """
    extracted_features = feature_extraction(id)
    all_embeddings = get_all_embeddings()
    similarities = []

    # Calculate cosine similarity with each face in the system
    for face_id, emb_str in all_embeddings:
        emb = string_to_array(emb_str)
        similarity = cosine_similarity(extracted_features, emb)
        similarities.append((face_id, similarity))
    
    # Sort based on similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Check if the top similarity is above the identification threshold
    if not similarities or similarities[0][1] < IDENTIFICATION_THRESHOLD:
        return None
    
    # Return the ids of the top N most similar faces
    return [face_id for face_id, _ in similarities[:TOP_N_MATCHES]]
