import cv2  # Import OpenCV library for computer vision tasks
import numpy as np  # Import NumPy library for numerical operations

# Constants
SKIP_FRAMES = 3  # Define the number of frames to skip after each processed frame for performance optimization

# Paths to the Haar cascade classifiers for face detection
detect_frontal_face = 'model/haarcascade_frontalface_alt.xml'  # Path to the frontal face cascade classifier
detect_profile_face = 'model/haarcascade_profileface.xml'  # Path to the profile face cascade classifier

# Function to detect a face in an image using the specified Haar cascade
def detect_face(cascade, img):
    # Use the cascade to detect faces in the image. Parameters are set to optimize face detection.
    rects, _, _ = cascade.detectMultiScale3(img, scaleFactor=1.3, minNeighbors=4, minSize=(60, 60),
                                            flags=cv2.CASCADE_SCALE_IMAGE, outputRejectLevels=True)
    # Return True if at least one face is detected, False otherwise
    return len(rects) > 0

# Function to detect a left profile face in a video
def detect_left_face(video_path):
    cap = cv2.VideoCapture(video_path)  # Initialize video capture on the given file
    detect_profile = cv2.CascadeClassifier(detect_profile_face)  # Load the profile face cascade classifier

    frame_count = 0  # Initialize frame counter
    while True:  # Iterate over frames in the video
        ret, frame = cap.read()  # Read the next frame
        if not ret:  # Break the loop if no frame is found
            break

        # Process every (SKIP_FRAMES + 1)-th frame to save computational resources
        if frame_count % (SKIP_FRAMES + 1) == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale for the cascade
            if detect_face(detect_profile, gray):  # Detect face in the grayscale frame
                cap.release()  # Release the video capture object
                return True  # Return True if a face is detected

        frame_count += 1  # Increment the frame counter

    cap.release()  # Release the video capture object
    return False  # Return False if no face is detected in the entire video

# Function to detect a right profile face in a video
def detect_right_face(video_path):
    cap = cv2.VideoCapture(video_path)  # Initialize video capture on the given file
    detect_profile = cv2.CascadeClassifier(detect_profile_face)  # Load the profile face cascade classifier

    frame_count = 0  # Initialize frame counter
    while True:  # Iterate over frames in the video
        ret, frame = cap.read()  # Read the next frame
        if not ret:  # Break the loop if no frame is found
            break

        # Process every (SKIP_FRAMES + 1)-th frame to save computational resources
        if frame_count % (SKIP_FRAMES + 1) == 0:
            gray = cv2.flip(frame, 1)  # Flip the frame horizontally to detect right profile faces
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)  # Convert the flipped frame to grayscale for the cascade
            if detect_face(detect_profile, gray):  # Detect face in the grayscale frame
                cap.release()  # Release the video capture object
                return True  # Return True if a face is detected

        frame_count += 1  # Increment the frame counter

    cap.release()  # Release the video capture object
    return False  # Return False if no face is detected in the entire video
