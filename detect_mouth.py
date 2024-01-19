import dlib  # Import dlib for face detection and landmark prediction
import cv2  # Import OpenCV for image processing
import os  # Import os module for file path operations
from imutils import face_utils  # Import face_utils for facial landmark utilities
from scipy.spatial import distance as dist  # Import distance module for calculating Euclidean distance

# Constants for aspect ratio thresholds and frames to skip
EYE_AR_THRESH = 0.23  # Threshold for the eye aspect ratio below which the eye is considered closed
MOUTH_AR_OPEN_THRESH = 0.3  # Threshold for the mouth aspect ratio above which the mouth is considered open
MOUTH_AR_CLOSED_THRESH = 0.2  # Threshold for the mouth aspect ratio below which the mouth is considered closed
SKIP_FRAMES = 3  # Number of frames to skip after processing one frame for performance optimization

# Path to the dlib facial landmark prediction model
model_path = "model/shape_predictor_68_face_landmarks.dat"  

# Define the FaceDetector class
class FaceDetector():
    def __init__(self):
        # Initialize dlib's face detector and facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model_path)

    # Method to determine if the mouth is open based on the mouth aspect ratio
    def mouth_open(self, gray, rect):
        # Get the indexes for the mouth landmarks
        (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        # Predict facial landmarks and convert to a NumPy array
        shape = self.predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # Extract the mouth coordinates
        mouth = shape[mStart:mEnd]
        # Calculate the mouth aspect ratio
        mar = self.mouth_aspect_ratio(mouth)
        # Return True if the mouth aspect ratio is greater than the open threshold
        return mar > MOUTH_AR_OPEN_THRESH

    # Method to determine if the mouth is closed based on the mouth aspect ratio
    def mouth_closed(self, gray, rect):
        # Get the indexes for the mouth landmarks
        (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        # Predict facial landmarks and convert to a NumPy array
        shape = self.predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # Extract the mouth coordinates
        mouth = shape[mStart:mEnd]
        # Calculate the mouth aspect ratio
        mar = self.mouth_aspect_ratio(mouth)
        # Return True if the mouth aspect ratio is less than or equal to the closed threshold
        return mar <= MOUTH_AR_CLOSED_THRESH

    # Method to calculate the mouth aspect ratio (MAR)
    def mouth_aspect_ratio(self, mouth):
        # Compute the pairwise distances between the specified points
        A = dist.euclidean(mouth[13], mouth[19])
        B = dist.euclidean(mouth[14], mouth[18])
        C = dist.euclidean(mouth[15], mouth[17])
        D = dist.euclidean(mouth[12], mouth[16])
        # Calculate the mouth aspect ratio
        mar = (A + B + C) / (3.0 * D)
        return mar

# Function to detect if the mouth is open in a video
def detect_open_mouth(video_path):
    detector = FaceDetector()  # Create an instance of the FaceDetector class
    cap = cv2.VideoCapture(video_path)  # Initialize video capture for the given file path
    frame_idx = 0  # Initialize frame index

    # Iterate through video frames
    while cap.isOpened():
        ret, frame = cap.read()  # Read a frame
        if not ret:  # Break the loop if no frame is read (end of video)
            break

        # Process every SKIP_FRAMES-th frame to save computational resources
        if frame_idx % SKIP_FRAMES == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale
            rects = detector.detector(gray, 0)  # Detect faces in the grayscale frame

            # Iterate through detected faces
            for rect in rects:
                # Check if the mouth is open for the current face
                if detector.mouth_open(gray, rect):
                    return True  # Return True if the mouth is detected as open

                # Check if the mouth is closed for the current face
                if detector.mouth_closed(gray, rect):
                    # Define the path to save the frame where the mouth is closed
                    save_path = os.path.join(os.path.dirname(video_path), 'closed_mouth_frame.jpg')
                    cv2.imwrite(save_path, frame)  # Save the frame to the specified path

        frame_idx += 1  # Increment the frame index

    cap.release()  # Release the video capture object
    return False  # Return False if no open mouth is detected in the entire video
