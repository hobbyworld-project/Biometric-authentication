# Import the necessary modules and functions
from flask import Flask, request, jsonify
from detect_face_profile import *  # Assumed to include functions like detect_left_face, detect_right_face
from detect_mouth import detect_open_mouth
from datetime import datetime, timedelta
from status_check import update_status_json, check_all_tasks_completed, get_normalized_filename
from feature_saving import embedding_feature_saving
from face_recognition import face_verification, face_identification
import os
import logging
import json

# Initialize a Flask app
app = Flask(__name__)

# Set up logging with the INFO level
logging.basicConfig(level=logging.INFO)

# Route for uploading videos and performing liveness detection
@app.route('/upload/<id>/<task>', methods=['POST'])
def liveness_detection(id, task):
    try:
        # Check if the file part exists in the request
        if 'file' not in request.files:
            logging.warning("No file part in the request")
            return jsonify({"error": "No file part"}), 400

        # Retrieve the file from the request
        file = request.files['file']
        # Check if a file is selected
        if file.filename == '':
            logging.warning("No selected file")
            return jsonify({"error": "No selected file"}), 400

        # Proceed if a file is present
        if file:
            # Define the upload folder path based on the user ID
            upload_folder = f'temporary/{id}/uploaded_videos'
            # Create the directory if it doesn't exist
            os.makedirs(upload_folder, exist_ok=True)
            # Normalize the file name to maintain consistency
            normalized_filename = get_normalized_filename(id, task, file.filename)
            # Construct the full file path
            file_path = os.path.join(upload_folder, normalized_filename)
            # Save the file to the defined path
            file.save(file_path)
            logging.info(f"File uploaded successfully: {file_path}")

            # Define the path for the status file
            status_file = f"temporary/{id}/status.json"
            # Check if the status file needs to be reset (does not exist or is older than 5 minutes)
            if not os.path.exists(status_file) or datetime.now() - datetime.fromtimestamp(os.path.getmtime(status_file)) > timedelta(minutes=5):
                with open(status_file, 'w') as f:
                    # Reset the status file with all tasks marked as not completed
                    json.dump({"detect_left_face": False, "detect_right_face": False, "detect_open_mouth": False}, f)

            # Define a mapping for task IDs to their corresponding function names
            task_mapping = {"1": "detect_left_face", "2": "detect_right_face", "3": "detect_open_mouth"}
            # Retrieve the corresponding task name from the mapping
            task_name = task_mapping.get(task, "invalid_task")

            # Return an error if the task ID is not valid
            if task_name == "invalid_task":
                return jsonify({"error": "Invalid task"}), 400

            result = False
            # Perform the appropriate detection task based on the task name
            if task_name == "detect_left_face":
                result = detect_left_face(file_path)
            elif task_name == "detect_right_face":
                result = detect_right_face(file_path)
            elif task_name == "detect_open_mouth":
                result = detect_open_mouth(file_path)

            # Update the status JSON file with the result of the task
            status = update_status_json(id, task_name, result)

            # Check if all tasks have been completed
            if check_all_tasks_completed(status):
                # If all tasks are completed, save the embedding features and return the result
                saving_result = embedding_feature_saving(id)
                return saving_result, 200
            else:
                # If not all tasks are completed, return the result of the current task
                return jsonify({task_name: result}), 200

    except Exception as e:
        # Log any exceptions that occur and return an internal server error
        logging.error(f"An error occurred: {e}")
        return jsonify({"error": "Internal server error"}), 500

# Route for performing face identification
@app.route('/identification/<id>/<task>', methods=['POST'])
def identification(id, task):
    try:
        # Check if the file part exists in the request
        if 'file' not in request.files:
            logging.warning("No file part in the request")
            return jsonify({"error": "No file part"}), 400

        # Retrieve the file from the request
        file = request.files['file']
        # Check if a file is selected
        if file.filename == '':
            logging.warning("No selected file")
            return jsonify({"error": "No selected file"}), 400

        # Proceed if a file is present
        if file:
            # Define the upload folder path based on the user ID
            upload_folder = f'temporary/{id}/uploaded_videos'
            # Create the directory if it doesn't exist
            os.makedirs(upload_folder, exist_ok=True)
            # Normalize the file name to maintain consistency
            normalized_filename = get_normalized_filename(id, task, file.filename)
            # Construct the full file path
            file_path = os.path.join(upload_folder, normalized_filename)
            # Save the file to the defined path
            file.save(file_path)
            logging.info(f"File uploaded successfully: {file_path}")

            # Define the path for the status file
            status_file = f"temporary/{id}/status.json"
            # Check if the status file needs to be reset (does not exist or is older than 5 minutes)
            if not os.path.exists(status_file) or datetime.now() - datetime.fromtimestamp(os.path.getmtime(status_file)) > timedelta(minutes=5):
                with open(status_file, 'w') as f:
                    # Reset the status file with all tasks marked as not completed
                    json.dump({"detect_left_face": False, "detect_right_face": False, "detect_open_mouth": False}, f)

            # Define a mapping for task IDs to their corresponding function names
            task_mapping = {"1": "detect_left_face", "2": "detect_right_face", "3": "detect_open_mouth"}
            # Retrieve the corresponding task name from the mapping
            task_name = task_mapping.get(task, "invalid_task")

            # Return an error if the task ID is not valid
            if task_name == "invalid_task":
                return jsonify({"error": "Invalid task"}), 400

            result = False
            # Perform the appropriate detection task based on the task name
            if task_name == "detect_left_face":
                result = detect_left_face(file_path)
            elif task_name == "detect_right_face":
                result = detect_right_face(file_path)
            elif task_name == "detect_open_mouth":
                result = detect_open_mouth(file_path)

            # Update the status JSON file with the result of the task
            status = update_status_json(id, task_name, result)

        if check_all_tasks_completed(status):
            # If all tasks are completed, perform face identification
            identification_result = face_identification(id)[0]
            # Return the identification result
            return identification_result, 200
        else:
            # If not all tasks are completed, return the result of the current task
            return jsonify({task_name: result}), 200
            
    except Exception as e:
        # Log any exceptions that occur and return an internal server error
        logging.error(f"An error occurred: {e}")
        return jsonify({"error": "Internal server error"}), 500

# Route for performing face verification
@app.route('/verification/<id>/<task>', methods=['POST'])
def verification(id, task):
    try:
        # Check if the file part exists in the request
        if 'file' not in request.files:
            logging.warning("No file part in the request")
            return jsonify({"error": "No file part"}), 400

        # Retrieve the file from the request
        file = request.files['file']
        # Check if a file is selected
        if file.filename == '':
            logging.warning("No selected file")
            return jsonify({"error": "No selected file"}), 400

        # Proceed if a file is present
        if file:
            # Define the upload folder path based on the user ID
            upload_folder = f'temporary/{id}/uploaded_videos'
            # Create the directory if it doesn't exist
            os.makedirs(upload_folder, exist_ok=True)
            # Normalize the file name to maintain consistency
            normalized_filename = get_normalized_filename(id, task, file.filename)
            # Construct the full file path
            file_path = os.path.join(upload_folder, normalized_filename)
            # Save the file to the defined path
            file.save(file_path)
            logging.info(f"File uploaded successfully: {file_path}")

            # Define the path for the status file
            status_file = f"temporary/{id}/status.json"
            # Check if the status file needs to be reset (does not exist or is older than 5 minutes)
            if not os.path.exists(status_file) or datetime.now() - datetime.fromtimestamp(os.path.getmtime(status_file)) > timedelta(minutes=5):
                with open(status_file, 'w') as f:
                    # Reset the status file with all tasks marked as not completed
                    json.dump({"detect_left_face": False, "detect_right_face": False, "detect_open_mouth": False}, f)

            # Define a mapping for task IDs to their corresponding function names
            task_mapping = {"1": "detect_left_face", "2": "detect_right_face", "3": "detect_open_mouth"}
            # Retrieve the corresponding task name from the mapping
            task_name = task_mapping.get(task, "invalid_task")

            # Return an error if the task ID is not valid
            if task_name == "invalid_task":
                return jsonify({"error": "Invalid task"}), 400

            result = False
            # Perform the appropriate detection task based on the task name
            if task_name == "detect_left_face":
                result = detect_left_face(file_path)
            elif task_name == "detect_right_face":
                result = detect_right_face(file_path)
            elif task_name == "detect_open_mouth":
                result = detect_open_mouth(file_path)

            # Update the status JSON file with the result of the task
            status = update_status_json(id, task_name, result)
            
        if check_all_tasks_completed(status):
            # If all tasks are completed, perform face verification
            verification_result = face_verification(id)
            # Return the verification result
            return verification_result, 200
        else:
            # If not all tasks are completed, return the result of the current task
            return jsonify({task_name: result}), 200
            
    except Exception as e:
        # Log any exceptions that occur and return an internal server error
        logging.error(f"An error occurred: {e}")
        return jsonify({"error": "Internal server error"}), 500

# Start the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
