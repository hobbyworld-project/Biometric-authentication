from flask import Flask, request, jsonify
from face_models import load_face_models, get_face_embedding
from utils import *
from blockchain_api import call_blockchain_api
import os

# Create a Flask app instance
app = Flask(__name__)

# Load the face detection and embedding models
mtcnn, resnet = load_face_models()

@app.route('/upload/<id>', methods=['POST'])
def upload_file(id):
    # Check if the 'file' key is present in the uploaded files
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    # Retrieve the file from the request
    file = request.files['file']

    # Check if a file is selected
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Process the file if it exists
    if file:
        # Save the file locally
        file_path = f"./uploaded_images/{id}_{file.filename}"
        file.save(file_path)

        # Read the image and get its face embedding
        with open(file_path, 'rb') as image_file:
            status, embedding, img_aligned = get_face_embedding(image_file.read(), mtcnn, resnet)
        
        # Save logs and the aligned image
        save_log(id, status)
        save_image(id, img_aligned)

        # Save the face embedding feature
        if save_feature(id, embedding):
            print("success")
        else:
            print("fail")

        # Handle different face detection statuses
        if status in ["no_face", "low_quality", "image_error"]:
            return jsonify({id: f"{status.replace('_', ' ').capitalize()} face"})
        else:
            # Call an external API with the embedding and return the result
            call_blockchain_api(id, embedding.tolist())
            return jsonify({id: embedding.tolist()})

    # Return a generic error if none of the above conditions are met
    return jsonify({"error": "Unknown error occurred"}), 500

# Main block to run the Flask application
if __name__ == '__main__':
    # Create necessary directories if they don't exist
    for directory in ['./uploaded_images', './cropped_faces', './logs']:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
