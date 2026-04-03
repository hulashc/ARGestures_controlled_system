"""Gesture engine — finger state detection and gesture recognition."""

from hand_tracking import THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP, WRIST
import config


class GestureEngine:
    def __init__(self):
        self.finger_pairs = {
            "index":  (INDEX_TIP, 6),
            "middle": (MIDDLE_TIP, 10),
            "ring":   (RING_TIP, 14),
            "pinky":  (PINKY_TIP, 18),
        }
        self.thumb_tip = THUMB_TIP
        self.thumb_ip = 3
        self.finger_mcps = {
            "index":  5,
            "middle": 9,
            "ring":   13,
            "pinky":  17,
        }

    def get_finger_states(self, landmarks):
        states = {}

        for name, (tip_idx, pip_idx) in self.finger_pairs.items():
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]
            mcp = landmarks[self.finger_mcps[name]]

            # Finger is UP if tip is above both PIP and MCP
            # Add small margin to avoid false triggers
            margin = 0.02
            if tip[1] < pip[1] - margin and tip[1] < mcp[1] - margin:
                states[name] = "up"
            else:
                states[name] = "down"

        # Thumb: check distance from index MCP — reliable across hand orientations
        thumb_tip = landmarks[self.thumb_tip]
        index_mcp = landmarks[5]
        thumb_to_index = ((thumb_tip[0] - index_mcp[0])**2 + (thumb_tip[1] - index_mcp[1])**2)**0.5
        states["thumb"] = "up" if thumb_to_index > 0.14 else "down"

        return states

    def count_fingers_up(self, states):
        return sum(1 for state in states.values() if state == "up")

    def get_fingertip_pixels(self, landmarks, frame_width, frame_height):
        tips = {
            "thumb":  landmarks[THUMB_TIP],
            "index":  landmarks[INDEX_TIP],
            "middle": landmarks[MIDDLE_TIP],
            "ring":   landmarks[RING_TIP],
            "pinky":  landmarks[PINKY_TIP],
        }
        return {
            name: (int(lm[0] * frame_width), int(lm[1] * frame_height))
            for name, lm in tips.items()
        }

    def pinch_distance(self, landmarks):
        thumb = landmarks[THUMB_TIP]
        index = landmarks[INDEX_TIP]
        return ((thumb[0] - index[0])**2 + (thumb[1] - index[1])**2)**0.5

    def detect_gesture(self, landmarks):
        states = self.get_finger_states(landmarks)
        pinch = self.pinch_distance(landmarks)
        fingers_up = self.count_fingers_up(states)

        # Order matters: most specific first

        # Fist: all fingers down
        if fingers_up == 0:
            return "fist"

        # Pinch: thumb+index touching, all other fingers down
        if pinch < 0.06 and states["middle"] == "down" and states["ring"] == "down" and states["pinky"] == "down":
            return "pinch"

        # Point: only index up
        if fingers_up == 1 and states["index"] == "up":
            return "point"

        # Open: all 5 up
        if fingers_up == 5:
            return "open"

        return "none"


class SwipeTracker:
    def __init__(self, threshold=0.12):
        self.threshold = threshold
        self.history = []
        self.max_history = 8
        self.last_swipe = "none"

    def update(self, landmarks):
        index_tip = landmarks[INDEX_TIP]
        self.history.append(index_tip[0])

        if len(self.history) > self.max_history:
            self.history.pop(0)

        if len(self.history) < self.max_history:
            return "none"

        start_x = self.history[0]
        end_x = self.history[-1]
        delta = end_x - start_x

        if abs(delta) > self.threshold:
            self.last_swipe = "swipe_right" if delta > 0 else "swipe_left"
            self.history = []
            return self.last_swipe

        return "none"

    def reset(self):
        self.history = []
        self.last_swipe = "none"
