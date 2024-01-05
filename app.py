from flask import Flask, request, jsonify
from face_models import load_face_models, get_face_embedding
from utils import *
from external_api import call_external_api
import os

app = Flask(__name__)

mtcnn, resnet = load_face_models()

@app.route('/upload/<id>', methods=['POST'])
def upload_file(id):
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        file_path = f"./uploaded_images/{id}_{file.filename}"
        file.save(file_path)

        with open(file_path, 'rb') as image_file:
            status, embedding, img_aligned = get_face_embedding(image_file.read(), mtcnn, resnet)
        
        save_log(id, status)

        save_image(id, img_aligned)

        save_feature(id, embedding)

        if status in ["no_face", "low_quality", "image_error"]:
            return jsonify({id: f"{status.replace('_', ' ').capitalize()} face"})
        else:
            call_external_api(id, embedding.tolist())

            return jsonify({id: embedding.tolist()})

    return jsonify({"error": "Unknown error occurred"}), 500

if __name__ == '__main__':
    for directory in ['./uploaded_images', './cropped_faces', './logs']:
        if not os.path.exists(directory):
            os.makedirs(directory)

    app.run(host='0.0.0.0', port=5000, debug=True)
