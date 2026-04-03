"""Windows 8 tile-style AR menu renderer."""

import cv2
import numpy as np
import config
from utils import ParticleSystem


TILE_COLS = 2
TILE_ROWS = 3
TILE_W = 160
TILE_H = 100
TILE_GAP = 10
HEADER_H = 40


class UIRenderer:
    def __init__(self):
        self.frame_count = 0
        self.particles = ParticleSystem()
        self.tile_rects = []  # stored each frame for hover detection

    def render(self, frame, menu, gesture_info):
        self.frame_count += 1
        h, w, _ = frame.shape
        self.tile_rects = []

        overlay = frame.copy()
        cv2.addWeighted(overlay, 0.90, np.zeros_like(frame), 0.10, 0, frame)

        cursor = gesture_info.get("cursor")
        if cursor:
            self._draw_cursor(frame, cursor, gesture_info.get("pinch_progress", 0))

        self._draw_tile_menu(frame, w, h, menu, gesture_info)
        self._draw_status(frame, w, gesture_info)

        self.particles.update_and_draw(frame)

    def _draw_tile_menu(self, frame, w, h, menu, gesture_info):
        info = menu.render_info()
        items = info["items"]
        depth = info["depth"]
        breadcrumb = info["breadcrumb"]
        trans_type = info["transition_type"]
        trans_timer = info["transition_timer"]

        if not items:
            return

        # Transition slide
        trans_offset = 0
        if trans_type == "enter" and trans_timer > 0:
            trans_offset = int(30 * (trans_timer / 15.0))
        elif trans_type == "exit" and trans_timer > 0:
            trans_offset = int(-30 * (trans_timer / 15.0))

        # Grid layout
        cols = TILE_COLS
        rows = min(TILE_ROWS, (len(items) + cols - 1) // cols)
        grid_w = cols * TILE_W + (cols - 1) * TILE_GAP
        grid_h = rows * TILE_H + (rows - 1) * TILE_GAP

        # Center the grid
        panel_x = (w - grid_w) // 2 + trans_offset
        panel_y = 60

        # Header bar
        header_y = panel_y - HEADER_H
        cv2.rectangle(frame, (0, header_y), (w, panel_y), (5, 5, 5), -1)
        cv2.line(frame, (0, panel_y), (w, panel_y), (50, 50, 50), 1)

        if depth > 0:
            cv2.putText(frame, "<  " + breadcrumb, (20, panel_y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_MAGENTA, 1)
        else:
            cv2.putText(frame, breadcrumb, (20, panel_y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_CYAN, 1)

        # Depth dots
        for d in range(depth + 1):
            cv2.circle(frame, (w - 30 - d * 12, panel_y - 15), 4,
                       config.COLOR_CYAN if d == depth else (60, 60, 60), -1)

        # Draw tiles
        self.tile_rects = []
        for i, item in enumerate(items):
            col = i % cols
            row = i // cols
            tx = panel_x + col * (TILE_W + TILE_GAP)
            ty = panel_y + row * (TILE_H + TILE_GAP)

            is_hovered = (gesture_info.get("hovered_idx") == i)
            is_activated = item.activated

            if is_activated:
                flash = int(255 * (info["activation_timer"] / 30.0))
                color = (flash, 0, flash)
            elif is_hovered:
                color = (0, 200, 200)
            else:
                # Alternating tile colors
                tile_colors = [
                    (0, 120, 180),    # blue
                    (0, 150, 80),     # green
                    (180, 80, 0),     # orange
                    (120, 0, 150),    # purple
                    (0, 100, 150),    # teal
                    (150, 50, 50),    # red
                ]
                color = tile_colors[i % len(tile_colors)]

            # Tile background
            cv2.rectangle(frame, (tx, ty), (tx + TILE_W, ty + TILE_H), color, -1)

            # Tile border on hover
            if is_hovered:
                cv2.rectangle(frame, (tx, ty), (tx + TILE_W, ty + TILE_H), (255, 255, 255), 2)

            # Item name (centered)
            text = item.name.upper()
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.55
            thickness = 1
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            text_x = tx + (TILE_W - tw) // 2
            text_y = ty + (TILE_H + th) // 2 - 5
            cv2.putText(frame, text, (text_x, text_y), font, scale, (255, 255, 255), thickness)

            # Submenu arrow
            if item.is_submenu:
                arrow = ">"
                (aw, ah), _ = cv2.getTextSize(arrow, font, 0.5, 1)
                cv2.putText(frame, arrow, (tx + TILE_W - aw - 10, ty + 20),
                            font, 0.5, (200, 200, 200), 1)

            # Description on hover
            if is_hovered and hasattr(item, "desc") and item.desc:
                desc_y = panel_y + grid_h + 20
                cv2.putText(frame, item.desc, (panel_x, desc_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, config.COLOR_WHITE, 1)

            # Store rect for hover detection
            self.tile_rects.append((tx, ty, tx + TILE_W, ty + TILE_H))

    def _draw_cursor(self, frame, cursor, pinch_progress):
        x, y = int(cursor[0]), int(cursor[1])

        # Outer ring
        cv2.circle(frame, (x, y), 16, config.COLOR_CYAN, 2)

        # Pinch progress arc
        if pinch_progress > 0.05:
            angle = int(360 * pinch_progress)
            cv2.ellipse(frame, (x, y), (16, 16), 0, -90, -90 + angle, config.COLOR_MAGENTA, 3)

        # Center dot
        cv2.circle(frame, (x, y), 4, config.COLOR_WHITE, -1)

    def _draw_status(self, frame, w, gesture_info):
        hand = gesture_info.get("hand_detected", False)
        status_color = config.COLOR_GREEN if hand else config.COLOR_RED
        status_text = "TRACKING" if hand else "SEARCHING"

        cv2.putText(frame, status_text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)

        if gesture_info.get("fps"):
            cv2.putText(frame, f"{gesture_info['fps']:.0f} fps", (w - 75, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, config.COLOR_WHITE, 1)

        gesture = gesture_info.get("gesture", "none").upper()
        gesture_colors = {
            "PINCH": config.COLOR_MAGENTA,
            "FIST": config.COLOR_RED,
            "POINT": config.COLOR_GREEN,
            "OPEN": config.COLOR_CYAN,
        }
        g_color = gesture_colors.get(gesture, (80, 80, 80))
        h, _, _ = frame.shape
        cv2.putText(frame, gesture, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, g_color, 1)

        if gesture_info.get("action"):
            cv2.putText(frame, gesture_info["action"], (10, h - 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, config.COLOR_GREEN, 1)

    def get_tile_rects(self):
        return self.tile_rects

    def emit_particles(self, x, y, color, count=25):
        self.particles.emit(x, y, color, count)
