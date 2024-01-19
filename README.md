# Project Overview
This project implements a sophisticated face recognition system leveraging Flask for the backend, OpenCV and dlib for image processing and facial landmark detection, and FaceNet for generating facial embeddings. The system provides functionalities for liveness detection, face verification, and face identification. It's designed to be robust and efficient, suitable for real-world applications.

## Environment Setup
The project is containerized using Docker for easy setup and deployment. The Dockerfile included in the project sets up the environment, installs all the necessary dependencies, and gets the application running with minimal setup.

**Steps**:
1. Install Docker on your system if it's not already installed.
2. Navigate to the project directory and build the Docker image:
   ```bash
   docker build -t face-recognition-app .
   ```
3. Once the build is complete, run the container:
   ```bash
   docker run -p 5000:5000 face-recognition-app
   ```

## File Structure
- `app.py`: Serves as the entry point of the Flask application. It defines endpoints for uploading videos/images and triggers the face recognition tasks.
- `db_operations.py`: Provides a set of functions for interacting with the SQLite database, including saving and retrieving face embeddings.
- `detect_face_profile.py`: Contains functions for detecting profile faces using Haar cascade classifiers. It's optimized for performance by skipping frames and using grayscale images for detection.
- `detect_mouth.py`: Includes functions to detect whether the mouth is open or closed in a video by analyzing facial landmarks and calculating aspect ratios.
- `dockerfile`: Defines the steps for creating the Docker container, including setting up the environment, installing dependencies, and setting the timezone.
- `face_model.py`: Handles the loading of MTCNN for face detection and InceptionResnetV1 for generating face embeddings. It also includes utilities for image preprocessing.
- `face_recognition.py`: Implements the logic for face verification and identification by comparing cosine similarities of face embeddings.
- `feature_saving.py`: Orchestrates the process of extracting facial features, saving them to the SQLite database, and optionally pushing them to a blockchain.

## Running Instructions
After setting up the environment and starting the Docker container, you can interact with the system through the defined endpoints.

1. **Liveness Detection**: Use the `/upload/<id>/<task>` endpoint to upload videos and perform liveness detection.
2. **Face Verification**: Use the `/verification/<id>/<task>` endpoint to verify the identity of an individual based on a provided face.
3. **Face Identification**: Use the `/identification/<id>/<task>` endpoint to identify an individual from a set of known faces.

Ensure that your requests are properly formatted and that the videos/images are accessible to the application.

## Additional Information
- **Modularity**: The system is designed with modularity in mind, making it easy to extend or modify functionalities.
- **Performance**: Special care has been taken to optimize the performance, including the use of frame skipping in video processing and efficient face detection and recognition algorithms.
- **Blockchain Integration**: The `feature_saving.py` file includes an optional integration with a blockchain for secure and tamper-proof storage of face embeddings.