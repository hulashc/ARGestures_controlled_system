"""Gesture engine — single-hand and two-hand gesture recognition.

GestureEngine   : detect gesture from one hand's 21-landmark list
TwoHandEngine   : detect interactions that require both hands
SwipeTracker    : swipe left / right from index-tip history
"""

import math
from hand_tracking import (THUMB_TIP, INDEX_TIP, MIDDLE_TIP,
                            RING_TIP, PINKY_TIP, WRIST)
import config


# ── Single-hand gesture ───────────────────────────────────────────────────────

class GestureEngine:
    """Detects a gesture from one hand's 21-landmark list."""

    _FINGER_PAIRS = {
        'index':  (INDEX_TIP,  6),
        'middle': (MIDDLE_TIP, 10),
        'ring':   (RING_TIP,   14),
        'pinky':  (PINKY_TIP,  18),
    }
    _FINGER_MCPS = {'index': 5, 'middle': 9, 'ring': 13, 'pinky': 17}

    def get_finger_states(self, lm):
        states = {}
        for name, (tip, pip) in self._FINGER_PAIRS.items():
            mcp    = self._FINGER_MCPS[name]
            margin = 0.02
            states[name] = (
                'up' if (lm[tip][1] < lm[pip][1] - margin
                         and lm[tip][1] < lm[mcp][1] - margin)
                else 'down'
            )
        thumb_to_imcp = math.hypot(lm[THUMB_TIP][0] - lm[5][0],
                                   lm[THUMB_TIP][1] - lm[5][1])
        states['thumb'] = 'up' if thumb_to_imcp > 0.14 else 'down'
        return states

    def count_fingers_up(self, states):
        return sum(1 for s in states.values() if s == 'up')

    def pinch_distance(self, lm):
        """Normalised distance between thumb tip and index tip."""
        return math.hypot(lm[THUMB_TIP][0] - lm[INDEX_TIP][0],
                          lm[THUMB_TIP][1] - lm[INDEX_TIP][1])

    def detect_gesture(self, lm):
        """Return gesture string for a single hand's landmarks."""
        states     = self.get_finger_states(lm)
        pinch      = self.pinch_distance(lm)
        fingers_up = self.count_fingers_up(states)

        if fingers_up == 0:
            return 'fist'
        if (pinch < config.PINCH_THRESHOLD
                and states['middle'] == 'down'
                and states['ring']   == 'down'
                and states['pinky']  == 'down'):
            return 'pinch'
        if fingers_up == 1 and states['index'] == 'up':
            return 'point'
        if fingers_up == 5:
            return 'open'
        return 'none'


# ── Two-hand interaction engine ───────────────────────────────────────────────

class TwoHandEngine:
    """
    Detects interactions requiring both hands simultaneously.

    Call update(left_lm, right_lm) every frame.
    Returns a dict:
      {
        'scale_delta':  float    # >0 = spread (scale up), <0 = pinch (scale down)
        'rotate_delta': float    # radians, CW positive
        'translate_xy': (dx,dy)  # normalised, both-fist grab-and-move
        'active':       bool     # True when both hands are present
      }
    """

    def __init__(self):
        self._prev_dist  = None
        self._prev_angle = None
        self._prev_mid   = None
        self._engine     = GestureEngine()

    def update(self, left_lm, right_lm):
        result = {
            'scale_delta':  0.0,
            'rotate_delta': 0.0,
            'translate_xy': (0.0, 0.0),
            'active':       False,
        }

        if left_lm is None or right_lm is None:
            self._prev_dist  = None
            self._prev_angle = None
            self._prev_mid   = None
            return result

        result['active'] = True

        # Anchor points: index-fingertip of each hand
        lx, ly = left_lm[INDEX_TIP][0],  left_lm[INDEX_TIP][1]
        rx, ry = right_lm[INDEX_TIP][0], right_lm[INDEX_TIP][1]

        dist  = math.hypot(rx - lx, ry - ly)
        angle = math.atan2(ry - ly, rx - lx)
        mid   = ((lx + rx) * 0.5, (ly + ry) * 0.5)

        if self._prev_dist is not None:
            # Scale: change in span between index tips
            d_dist = dist - self._prev_dist
            result['scale_delta'] = d_dist * config.TWO_HAND_SCALE_SENS

            # Rotate: change in angle of the line connecting both tips
            d_angle = angle - self._prev_angle
            if d_angle >  math.pi: d_angle -= 2 * math.pi
            if d_angle < -math.pi: d_angle += 2 * math.pi
            result['rotate_delta'] = d_angle * config.TWO_HAND_ROTATE_SENS

            # Translate: movement of midpoint when both hands are fists
            lg = self._engine.detect_gesture(left_lm)
            rg = self._engine.detect_gesture(right_lm)
            if lg == 'fist' and rg == 'fist':
                dx = mid[0] - self._prev_mid[0]
                dy = mid[1] - self._prev_mid[1]
                result['translate_xy'] = (dx, dy)

        self._prev_dist  = dist
        self._prev_angle = angle
        self._prev_mid   = mid
        return result


# ── Swipe tracker ─────────────────────────────────────────────────────────────

class SwipeTracker:
    def __init__(self, threshold=0.12):
        self.threshold   = threshold
        self.history     = []
        self.max_history = 8
        self.last_swipe  = 'none'

    def update(self, landmarks):
        self.history.append(landmarks[INDEX_TIP][0])
        if len(self.history) > self.max_history:
            self.history.pop(0)
        if len(self.history) < self.max_history:
            return 'none'
        delta = self.history[-1] - self.history[0]
        if abs(delta) > self.threshold:
            result = 'swipe_right' if delta > 0 else 'swipe_left'
            self.history    = []
            self.last_swipe = result
            return result
        return 'none'

    def reset(self):
        self.history    = []
        self.last_swipe = 'none'
