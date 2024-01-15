from flask import Flask, request, jsonify
from detect_face_profile import *
from detect_mouth import detect_open_mouth
from datetime import datetime, timedelta
from status_check import update_status_json, check_all_tasks_completed, get_normalized_filename
from feature_saving import embedding_feature_saving
import os
import logging
import json

# Create a Flask app instance
app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

@app.route('/upload/<id>/<task>', methods=['POST'])
def liveness_detection(id, task):
    try:
        if 'file' not in request.files:
            logging.warning("No file part in the request")
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            logging.warning("No selected file")
            return jsonify({"error": "No selected file"}), 400

        if file:
            upload_folder = f'temporary/{id}/uploaded_videos'
            os.makedirs(upload_folder, exist_ok=True)
            normalized_filename = get_normalized_filename(id, task, file.filename)
            file_path = os.path.join(upload_folder, normalized_filename)
            file.save(file_path)
            logging.info(f"File uploaded successfully: {file_path}")

            status_file = f"temporary/{id}/status.json"
            if not os.path.exists(status_file) or datetime.now() - datetime.fromtimestamp(os.path.getmtime(status_file)) > timedelta(minutes=5):
                with open(status_file, 'w') as f:
                    json.dump({"detect_left_face": False, "detect_right_face": False, "detect_open_mouth": False}, f)

            task_mapping = {"1": "detect_left_face", "2": "detect_right_face", "3": "detect_open_mouth"}
            task_name = task_mapping.get(task, "invalid_task")

            if task_name == "invalid_task":
                return jsonify({"error": "Invalid task"}), 400

            result = False
            if task_name == "detect_left_face":
                result = detect_left_face(file_path)
            elif task_name == "detect_right_face":
                result = detect_right_face(file_path)
            elif task_name == "detect_open_mouth":
                result = detect_open_mouth(file_path)

            status = update_status_json(id, task_name, result)

            if check_all_tasks_completed(status):
                saving_result = embedding_feature_saving(id)
                return saving_result, 200
            else:
                return jsonify({task_name: result}), 200
            

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)