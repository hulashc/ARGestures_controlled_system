"""Hand tracking — returns up to two hands (left + right) separately.

Public API
----------
tracker.get_frame()  ->  (frame, ok)
tracker.detect(frame) -> (hands, frame)

  hands = {
    'left':  list of 21 (x,y,z) normalised landmarks  |  None
    'right': list of 21 (x,y,z) normalised landmarks  |  None
  }

All coordinates are normalised [0..1] relative to frame size.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import config
import os
from utils import LandmarkSmoother


# ── Landmark indices ──────────────────────────────────────────────────────────
THUMB_TIP  = 4
INDEX_TIP  = 8
MIDDLE_TIP = 12
RING_TIP   = 16
PINKY_TIP  = 20
WRIST      = 0

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# Colours per hand side (BGR)
_COLOR_RIGHT = (0, 230, 255)    # cyan  – right hand
_COLOR_LEFT  = (200, 80, 255)   # violet – left hand

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

        # Independent smoothers per hand
        self._smoothers = {
            'left':  LandmarkSmoother(alpha=0.35),
            'right': LandmarkSmoother(alpha=0.35),
        }

        self.frame_counter   = 0
        self.inference_skip  = 2
        self._last_hands     = {'left': None, 'right': None}
        self._stale          = {'left': 0, 'right': 0}

        self.cap = None
        self._init_camera()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _ensure_model(self):
        path = 'hand_landmarker.task'
        if not os.path.exists(path):
            print('Downloading hand landmarker model (~7 MB, first run only)...')
            import urllib.request
            url = ('https://storage.googleapis.com/mediapipe-models/'
                   'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task')
            urllib.request.urlretrieve(url, path)
            print('Model downloaded.')

    def _init_camera(self):
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        if not self.cap.isOpened():
            raise RuntimeError(
                f'Cannot open camera (index={config.CAMERA_INDEX}).')
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS,          config.TARGET_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, False
        frame = cv2.flip(frame, 1)
        return frame, True

    def detect(self, frame):
        """
        Returns (hands, frame)
          hands = {'left': landmarks|None, 'right': landmarks|None}
        """
        self.frame_counter += 1

        # Frame-skip: reuse last result if still fresh
        if self.frame_counter % self.inference_skip != 0:
            for side in ('left', 'right'):
                if self._last_hands[side] is not None:
                    self._stale[side] += 1
                    if self._stale[side] > config.LANDMARK_STALE_FRAMES:
                        self._last_hands[side] = None
                        self._stale[side] = 0
                    else:
                        self._draw_hand(frame, self._last_hands[side], side)
            return dict(self._last_hands), frame

        # ── Run MediaPipe inference ───────────────────────────────────────────
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (INFERENCE_SIZE, int(INFERENCE_SIZE * h / w)))
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_img)

        # Reset both sides each inference cycle
        detected = {'left': None, 'right': None}

        for i, hand_lm in enumerate(result.hand_landmarks):
            # MediaPipe labels from the model's perspective (mirrored)
            # After cv2.flip(frame,1) "Right" in model = user's right hand
            if i < len(result.handedness):
                label = result.handedness[i][0].category_name.lower()  # 'left'|'right'
            else:
                label = 'right' if i == 0 else 'left'

            raw = [(lm.x, lm.y, lm.z) for lm in hand_lm]
            smoothed = self._smoothers[label].update(raw)
            detected[label] = smoothed

        # Update cache + stale counters
        for side in ('left', 'right'):
            if detected[side] is not None:
                self._last_hands[side] = detected[side]
                self._stale[side] = 0
                self._draw_hand(frame, detected[side], side)
            else:
                self._smoothers[side].reset()
                if self._last_hands[side] is not None:
                    self._stale[side] += 1
                    if self._stale[side] > config.LANDMARK_STALE_FRAMES:
                        self._last_hands[side] = None
                        self._stale[side] = 0

        return dict(self._last_hands), frame

    def _draw_hand(self, frame, landmarks, side='right'):
        h, w, _ = frame.shape
        color = _COLOR_RIGHT if side == 'right' else _COLOR_LEFT
        tip_color = (255, 80, 200) if side == 'right' else (80, 255, 160)

        for a, b in HAND_CONNECTIONS:
            pa = (int(landmarks[a][0]*w), int(landmarks[a][1]*h))
            pb = (int(landmarks[b][0]*w), int(landmarks[b][1]*h))
            cv2.line(frame, pa, pb, color, 2)

        tips = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
        for i, lm in enumerate(landmarks):
            cx, cy = int(lm[0]*w), int(lm[1]*h)
            if i in tips:
                cv2.circle(frame, (cx, cy), 6, tip_color, -1)
            else:
                cv2.circle(frame, (cx, cy), 3, color, -1)

        # Side label near wrist
        wx = int(landmarks[WRIST][0]*w)
        wy = int(landmarks[WRIST][1]*h)
        cv2.putText(frame, side[0].upper(), (wx-6, wy+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.detector.close()
