import cv2
import numpy as np

# Constants
SKIP_FRAMES = 3  # Skip 'n' frames after each processed frame

# Paths to the cascade classifiers
detect_frontal_face = 'model/haarcascade_frontalface_alt.xml'
detect_profile_face = 'model/haarcascade_profileface.xml'


def detect_face(cascade, img):
    rects, _, _ = cascade.detectMultiScale3(img, scaleFactor=1.3, minNeighbors=4, minSize=(60, 60),
                                            flags=cv2.CASCADE_SCALE_IMAGE, outputRejectLevels=True)
    return len(rects) > 0


def detect_left_face(video_path):
    cap = cv2.VideoCapture(video_path)
    detect_profile = cv2.CascadeClassifier(detect_profile_face)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % (SKIP_FRAMES + 1) == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if detect_face(detect_profile, gray):
                cap.release()
                return True

        frame_count += 1

    cap.release()
    return False


def detect_right_face(video_path):
    cap = cv2.VideoCapture(video_path)
    detect_profile = cv2.CascadeClassifier(detect_profile_face)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % (SKIP_FRAMES + 1) == 0:
            gray = cv2.flip(frame, 1)
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            if detect_face(detect_profile, gray):
                cap.release()
                return True

        frame_count += 1

    cap.release()
    return False
