"""Touch-based interaction system — cursor position + pinch detection.

Note: hover detection is handled in main.py via renderer.get_tile_rects()
for accurate tile-aligned hit-testing. This module provides cursor and
pinch helpers used by the main loop.
"""

from hand_tracking import INDEX_TIP, THUMB_TIP
import config


class TouchEngine:
    def __init__(self):
        self.hover_index = -1
        self.hovered_item_name = ""

    def get_cursor(self, landmarks, frame_w, frame_h):
        """Get cursor position from index fingertip."""
        tip = landmarks[INDEX_TIP]
        return int(tip[0] * frame_w), int(tip[1] * frame_h)

    def is_pinching(self, landmarks):
        """Check if thumb and index are within PINCH_THRESHOLD (tap gesture)."""
        thumb = landmarks[THUMB_TIP]
        index = landmarks[INDEX_TIP]
        dist = ((thumb[0] - index[0])**2 + (thumb[1] - index[1])**2)**0.5
        return dist < config.PINCH_THRESHOLD
