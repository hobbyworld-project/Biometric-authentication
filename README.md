# Face Recognition System

## Overview
This face recognition system is designed to process and store facial embeddings. Utilizing advanced machine learning models and a robust backend, this system can detect faces, extract embeddings, and perform various database operations. It's ideal for applications in security, user authentication, and personalized user experiences.

## Components

### 1. Flask Application
`app.py` - A Flask-based web server that handles image uploads. It processes these images to extract face embeddings and communicates with an external API and a local SQLite database.

### 2. Utility Functions
Includes various scripts for image processing and database management.

- `utils.py` - Contains helper functions for image processing like logging, image saving, and image orientation correction.
- `db_operations.py` - Manages database interactions, including initializing the database and saving embeddings.

### 3. Model Loading and Face Embedding
- `face_models.py` - Responsible for loading the MTCNN and InceptionResnetV1 models for face detection and embeddings.
- `get_face_embedding.py` - Processes images to detect faces and extract embeddings using the loaded models.

### 4. Blockchain API Communication
- `blockchain_api.py` - Contains functionality to make requests to the blockchain API, sending facial embeddings and related data.

### 5. Database Initialization and Management
- `init_db.py` - Initializes the SQLite database and creates necessary tables.
- `sqlite_operations.py` - Provides functions for database connectivity and performing CRUD operations.

## Setup and Installation

1. **Install Dependencies**: Ensure Python 3.x is installed. Install required libraries using `pip install -r requirements.txt`.

2. **Database Setup**: Run `init_db.py` to initialize the database.

3. **Starting the Server**: Run `app.py` to start the Flask server.

4. **Using the System**: Upload images to the server. The system will process these images, extract embeddings, and perform necessary operations as per the configured functionalities.

## Usage

### Uploading Images
Make a POST request to `/upload/<id>` with an image file. The server will process the image, save logs, embeddings, and communicate with external APIs if configured.

### Database Operations
Use the scripts in `db_operations.py` and `sqlite_operations.py` to interact with the database for storing and retrieving facial embeddings.

## Note
This system is a prototype. Ensure to configure and test thoroughly before deploying in a production environment.