"""Touch-based interaction system — cursor hover + pinch to select."""

from hand_tracking import INDEX_TIP


class TouchEngine:
    def __init__(self):
        self.hover_index = -1
        self.hovered_item_name = ""

    def get_cursor(self, landmarks, frame_w, frame_h):
        """Get cursor position from index fingertip."""
        tip = landmarks[INDEX_TIP]
        return int(tip[0] * frame_w), int(tip[1] * frame_h)

    def detect_hover(self, cursor_x, cursor_y, menu_items, panel_x, panel_y, item_h, header_h):
        """Check which menu item the cursor is hovering over.

        Returns index of hovered item, or -1 if none.
        """
        for i in range(len(menu_items)):
            y_top = panel_y + header_h + 8 + i * item_h
            y_bottom = y_top + item_h - 3
            x_left = panel_x + 4
            x_right = panel_x + 220 - 4  # panel_w - 4

            if x_left <= cursor_x <= x_right and y_top <= cursor_y <= y_bottom:
                self.hover_index = i
                self.hovered_item_name = menu_items[i].name
                return i

        self.hover_index = -1
        self.hovered_item_name = ""
        return -1

    def is_pinching(self, landmarks, threshold=0.06):
        """Check if thumb and index are close (tap gesture)."""
        thumb = landmarks[4]
        index = landmarks[INDEX_TIP]
        dist = ((thumb[0] - index[0])**2 + (thumb[1] - index[1])**2)**0.5
        return dist < threshold
