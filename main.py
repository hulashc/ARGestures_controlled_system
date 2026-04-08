"""AR Gesture Controlled System — main loop.

Controls (MENU mode):
    Open palm + move   → Move cursor
    Hover tile + pinch → Select / enter submenu
    Swipe left         → Go back
    Swipe right        → Next item

Controls (FORGE 3-D modeller):
    OPEN  palm          → Orbit scene
    POINT (1 finger)    → Translate selected part
    PINCH               → Select part / toolbar action
    FIST  (hold 0.4 s)  → Confirm attach / detach / delete
    Swipe left          → Back to menu

Press 'q' to quit.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import time
import numpy as np
from hand_tracking  import HandTracker, INDEX_TIP, THUMB_TIP
from gesture_engine import GestureEngine, SwipeTracker
from utils          import CursorSmoother
from menu_state     import MenuState
from ui_renderer    import UIRenderer
import config

# App registry — lazy-loaded on first launch
APP_REGISTRY = {}


def _load_app(app_id):
    if app_id == "modeller_3d":
        from apps.modeller_3d import Modeller3D
        return Modeller3D()
    return None


def play_beep(freq=800, duration=0.08, volume=0.12):
    try:
        import sounddevice as sd
        t = np.linspace(0, duration, int(44100 * duration), False)
        tone = volume * np.sin(2 * np.pi * freq * t)
        sd.play(tone, 44100, blocking=False)
    except Exception:
        pass


def main():
    print("=== AR Gesture Controlled System ===")
    print("Initializing...")

    tracker       = HandTracker()
    engine        = GestureEngine()
    swipe_tracker = SwipeTracker(threshold=config.SWIPE_THRESHOLD)
    cursor_smooth = CursorSmoother(alpha=config.CURSOR_SMOOTHER_ALPHA)
    menu          = MenuState()
    renderer      = UIRenderer()

    # Active app instance (None = menu mode)
    active_app    = None
    active_app_id = None

    action_flash  = ""
    action_timer  = 0
    prev_time     = time.time()
    fps           = 0

    pinch_was_active = False
    fist_was_active  = False
    hovered_idx      = -1

    # For just_pinch / just_fist edge detection inside app
    _prev_pinch = False
    _prev_fist  = False

    print("Ready.")
    print("  Open palm + move   → Move cursor")
    print("  Hover + pinch      → Select")
    print("  Swipe left         → Go back / exit app")
    print("Press 'q' to quit.\n")

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

            cursor     = None
            is_pinching = False
            is_fisting  = False
            raw_gesture = "none"
            hovered_idx = -1
            swipe_result = "none"

            if landmarks:
                raw_gesture = engine.detect_gesture(landmarks)

                tip   = landmarks[INDEX_TIP]
                cursor = cursor_smooth.update(
                    int(tip[0] * w), int(tip[1] * h))

                thumb = landmarks[THUMB_TIP]
                index = landmarks[INDEX_TIP]
                pinch_dist = (
                    (thumb[0]-index[0])**2 + (thumb[1]-index[1])**2
                ) ** 0.5
                is_pinching = pinch_dist < config.PINCH_THRESHOLD
                is_fisting  = (raw_gesture == "fist")

                swipe_result = swipe_tracker.update(landmarks) or "none"

                # ── MENU mode ────────────────────────────────────────
                if active_app is None:
                    tile_rects = renderer.get_tile_rects()
                    for i, (x1, y1, x2, y2) in enumerate(tile_rects):
                        if x1 <= cursor[0] <= x2 and y1 <= cursor[1] <= y2:
                            hovered_idx = i
                            break

                    if is_pinching and not pinch_was_active and hovered_idx >= 0:
                        menu.selected_index = hovered_idx
                        app_id = menu.activate()
                        if app_id:
                            # Launch app
                            active_app_id = app_id
                            active_app    = _load_app(app_id)
                            action_flash  = f"LAUNCHING: {app_id.upper()}"
                            play_beep(1400, 0.15, 0.12)
                        else:
                            item = menu.selected_item
                            if item and item.is_submenu:
                                action_flash = f"ENTER: {item.name}"
                                play_beep(1000, 0.1, 0.1)
                            elif item:
                                action_flash = f"ACTIVATE: {item.name}"
                                renderer.emit_particles(
                                    cursor[0], cursor[1],
                                    config.COLOR_MAGENTA, 25)
                                play_beep(1200, 0.15, 0.12)
                        action_timer = 30

                    if swipe_result == "swipe_left":
                        if not menu.is_root:
                            menu.go_back()
                            action_flash = "BACK"
                            play_beep(400, 0.08, 0.08)
                        else:
                            menu.prev_item()
                            action_flash = "PREV"
                            play_beep(600, 0.05, 0.08)
                        action_timer = 15
                    elif swipe_result == "swipe_right":
                        menu.next_item()
                        action_flash = "NEXT"
                        play_beep(600, 0.05, 0.08)
                        action_timer = 15

                pinch_was_active = is_pinching

            else:
                cursor_smooth.reset()
                swipe_tracker.reset()
                pinch_was_active = False

            # ── APP mode ───────────────────────────────────────────
            if active_app is not None:
                just_pinch = is_pinching and not _prev_pinch
                just_fist  = is_fisting  and not _prev_fist

                app_gesture_info = {
                    "fps":          fps,
                    "hand_detected": landmarks is not None,
                    "gesture":      raw_gesture,
                    "cursor":       cursor,
                    "pinch_progress": 1.0 if is_pinching else 0.0,
                    "just_pinched": just_pinch,
                    "just_fisted":  just_fist,
                    "swipe":        swipe_result,
                }

                still_open = active_app.update_and_render(frame, app_gesture_info)
                if not still_open:
                    active_app    = None
                    active_app_id = None
                    menu._clear_activation()
                    play_beep(400, 0.10, 0.10)

                _prev_pinch = is_pinching
                _prev_fist  = is_fisting

            # ── MENU render ─────────────────────────────────────────
            if active_app is None:
                if action_timer > 0:
                    action_timer -= 1
                else:
                    action_flash = ""

                gesture_info = {
                    "fps":          fps,
                    "hand_detected": landmarks is not None,
                    "gesture":      raw_gesture,
                    "action":       action_flash,
                    "cursor":       cursor,
                    "pinch_progress": 1.0 if is_pinching else 0.0,
                    "hovered_idx":  hovered_idx,
                    "depth":        menu.depth,
                }

                renderer.render(frame, menu, gesture_info)

            cv2.imshow("AR Gesture Controlled System", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nQuitting...")
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        tracker.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
