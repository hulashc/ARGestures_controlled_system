"""Touch-based AR tile menu system.

Controls:
    Open palm + move  → Cursor moves
    Hover tile + pinch → Select / enter submenu
    Swipe left        → Go back
    Swipe right       → Next item

Run: python main.py
Press 'q' to quit.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import time
import numpy as np
from hand_tracking import HandTracker
from gesture_engine import GestureEngine, SwipeTracker
from utils import CursorSmoother
from menu_state import MenuState
from ui_renderer import UIRenderer
import config


def play_beep(freq=800, duration=0.08, volume=0.12):
    try:
        import sounddevice as sd
        t = np.linspace(0, duration, int(44100 * duration), False)
        tone = volume * np.sin(2 * np.pi * freq * t)
        sd.play(tone, 44100, blocking=False)
    except Exception:
        pass


def main():
    print("=== AR Tile Menu System ===")
    print("Initializing...")

    tracker = HandTracker()
    engine = GestureEngine()
    swipe_tracker = SwipeTracker(threshold=0.08)
    cursor_smooth = CursorSmoother(alpha=0.35)
    menu = MenuState()
    renderer = UIRenderer()

    print("Ready.")
    print("  Open palm + move → Move cursor")
    print("  Hover tile + pinch → Select")
    print("  Swipe left       → Go back")
    print("  Swipe right      → Next item")
    print("Press 'q' to quit.\n")

    action_flash = ""
    action_timer = 0
    prev_time = time.time()
    fps = 0

    pinch_was_active = False
    hovered_idx = -1

    try:
        while True:
            frame, ok = tracker.get_frame()
            if not ok or frame is None:
                continue

            h, w, _ = frame.shape

            curr_time = time.time()
            fps = 0.95 * fps + 0.05 / (curr_time - prev_time + 0.0001)
            prev_time = curr_time

            landmarks, frame = tracker.detect(frame)

            cursor = None
            is_pinching = False
            raw_gesture = "none"
            hovered_idx = -1

            if landmarks:
                raw_gesture = engine.detect_gesture(landmarks)
                states = engine.get_finger_states(landmarks)

                # Cursor
                tip = landmarks[8]  # index tip
                cursor = cursor_smooth.update(int(tip[0] * w), int(tip[1] * h))

                # Pinch
                thumb = landmarks[4]
                index = landmarks[8]
                pinch_dist = ((thumb[0] - index[0])**2 + (thumb[1] - index[1])**2)**0.5
                is_pinching = pinch_dist < 0.06

                # Hover detection using tile rects from last render
                tile_rects = renderer.get_tile_rects()
                for i, (x1, y1, x2, y2) in enumerate(tile_rects):
                    if x1 <= cursor[0] <= x2 and y1 <= cursor[1] <= y2:
                        hovered_idx = i
                        break

                # Pinch tap on hovered tile
                if is_pinching and not pinch_was_active and hovered_idx >= 0:
                    menu.selected_index = hovered_idx
                    menu.activate()
                    item = menu.selected_item
                    if item and item.is_submenu:
                        action_flash = f"ENTER: {item.name}"
                        play_beep(1000, 0.1, 0.1)
                    elif item:
                        action_flash = f"ACTIVATE: {item.name}"
                        renderer.emit_particles(cursor[0], cursor[1], config.COLOR_MAGENTA, 25)
                        play_beep(1200, 0.15, 0.12)
                    action_timer = 30

                # Swipe — always track index finger position
                swipe = swipe_tracker.update(landmarks)
                if swipe == "swipe_left":
                    if not menu.is_root:
                        menu.go_back()
                        action_flash = "BACK"
                        play_beep(400, 0.08, 0.08)
                    else:
                        menu.prev_item()
                        action_flash = "PREV"
                        play_beep(600, 0.05, 0.08)
                    action_timer = 15
                elif swipe == "swipe_right":
                    menu.next_item()
                    action_flash = "NEXT"
                    play_beep(600, 0.05, 0.08)
                    action_timer = 15

                pinch_was_active = is_pinching
            else:
                cursor_smooth.reset()
                swipe_tracker.reset()
                pinch_was_active = False

            if action_timer > 0:
                action_timer -= 1
            else:
                action_flash = ""

            gesture_info = {
                "fps": fps,
                "hand_detected": landmarks is not None,
                "gesture": raw_gesture,
                "action": action_flash,
                "cursor": cursor,
                "pinch_progress": 1.0 if is_pinching else 0.0,
                "hovered_idx": hovered_idx,
            }

            renderer.render(frame, menu, gesture_info)

            cv2.imshow("AR Tile Menu System", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nQuitting...")
                break

    except KeyboardInterrupt:
        print("\nInterrupted. Shutting down...")
    finally:
        tracker.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
