"""Hand tracking module using MediaPipe Hands.

Responsibilities:
- Initialize webcam
- Detect hand landmarks with optimized inference
- Return smoothed landmark coordinates
- Handle camera errors gracefully
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import config
import os
from utils import LandmarkSmoother


# Hand landmark connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
WRIST = 0

# Inference resolution (lower = faster, results scale back up)
INFERENCE_SIZE = 320


class HandTracker:
    def __init__(self):
        self._ensure_model()

        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=config.MAX_HANDS,
            min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.MIN_TRACKING_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.smoother = LandmarkSmoother(alpha=0.35)

        # Frame skipping: run inference every N frames
        self.inference_skip = 2
        self.frame_counter = 0
        self.last_landmarks = None
        self.frames_since_detection = 0  # staleness guard

        self.cap = None
        self._init_camera()

    def _ensure_model(self):
        model_path = 'hand_landmarker.task'
        if not os.path.exists(model_path):
            print("Downloading hand landmarker model (first time only, ~7MB)...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            import urllib.request
            urllib.request.urlretrieve(url, model_path)
            print("Model downloaded successfully.")

    def _init_camera(self):
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera (index={config.CAMERA_INDEX}). "
                "Check if another app is using it or try a different index."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # reduce input lag

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, False
        frame = cv2.flip(frame, 1)
        return frame, True

    def detect(self, frame):
        self.frame_counter += 1

        # Skip inference on some frames — reuse last result if still fresh
        if self.frame_counter % self.inference_skip != 0:
            if self.last_landmarks is not None:
                self.frames_since_detection += 1
                if self.frames_since_detection > config.LANDMARK_STALE_FRAMES:
                    # Hand has been absent long enough — clear cached landmarks
                    self.last_landmarks = None
                    self.frames_since_detection = 0
                    return None, frame
                self._draw_hand(frame, self.last_landmarks)
            return self.last_landmarks, frame

        # Resize for faster inference
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (INFERENCE_SIZE, int(INFERENCE_SIZE * h / w)))

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)

        landmarks = None

        if result.hand_landmarks:
            hand_lm = result.hand_landmarks[0]
            raw = [(lm.x, lm.y, lm.z) for lm in hand_lm]
            landmarks = self.smoother.update(raw)
            self.last_landmarks = landmarks
            self.frames_since_detection = 0  # reset staleness counter
            self._draw_hand(frame, landmarks)
        else:
            self.smoother.reset()
            self.last_landmarks = None
            self.frames_since_detection = 0

        return landmarks, frame

    def _draw_hand(self, frame, landmarks):
        h, w, _ = frame.shape

        for a, b in HAND_CONNECTIONS:
            pt_a = landmarks[a]
            pt_b = landmarks[b]
            cv2.line(
                frame,
                (int(pt_a[0] * w), int(pt_a[1] * h)),
                (int(pt_b[0] * w), int(pt_b[1] * h)),
                config.COLOR_CYAN, 2,
            )

        for i, lm in enumerate(landmarks):
            cx, cy = int(lm[0] * w), int(lm[1] * h)
            if i in (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP):
                cv2.circle(frame, (cx, cy), 6, config.COLOR_MAGENTA, -1)
            else:
                cv2.circle(frame, (cx, cy), 3, config.COLOR_CYAN, -1)

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.detector.close()
