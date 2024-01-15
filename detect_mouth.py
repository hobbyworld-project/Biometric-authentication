import dlib
import cv2
import os
from imutils import face_utils
from scipy.spatial import distance as dist


EYE_AR_THRESH = 0.23 
MOUTH_AR_OPEN_THRESH = 0.3 
MOUTH_AR_CLOSED_THRESH = 0.2  
SKIP_FRAMES = 3


model_path = "model/shape_predictor_68_face_landmarks.dat"  

class FaceDetector():
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model_path)

    def mouth_open(self, gray, rect):
        (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        shape = self.predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        mouth = shape[mStart:mEnd]
        mar = self.mouth_aspect_ratio(mouth)
        return mar > MOUTH_AR_OPEN_THRESH

    def mouth_closed(self, gray, rect):
        (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        shape = self.predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        mouth = shape[mStart:mEnd]
        mar = self.mouth_aspect_ratio(mouth)
        return mar <= MOUTH_AR_CLOSED_THRESH

    def mouth_aspect_ratio(self, mouth):
        A = dist.euclidean(mouth[13], mouth[19])
        B = dist.euclidean(mouth[14], mouth[18])
        C = dist.euclidean(mouth[15], mouth[17])
        D = dist.euclidean(mouth[12], mouth[16])
        mar = (A + B + C) / (3.0 * D)
        return mar

def detect_open_mouth(video_path):
    detector = FaceDetector()
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % SKIP_FRAMES == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector.detector(gray, 0)

            for rect in rects:
                if detector.mouth_open(gray, rect):
                    return True

                if detector.mouth_closed(gray, rect):
                    save_path = os.path.join(os.path.dirname(video_path), 'closed_mouth_frame.jpg')
                    cv2.imwrite(save_path, frame)

        frame_idx += 1

    cap.release()
    return False
