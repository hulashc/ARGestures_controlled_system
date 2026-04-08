"""Stark OS — main loop.

Boots directly into the FORGE holographic modeller.
Left hand = violet  |  Right hand = cyan.

Press 'q' to quit.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import time
import math
import numpy as np

from hand_tracking  import HandTracker, INDEX_TIP, THUMB_TIP, WRIST
from gesture_engine import GestureEngine, TwoHandEngine, SwipeTracker
from utils          import CursorSmoother
from menu_state     import MenuState
from ui_renderer    import UIRenderer
from apps.modeller_3d import Modeller3D
import config


def play_beep(freq=800, duration=0.08, volume=0.12):
    try:
        import sounddevice as sd
        t = np.linspace(0, duration, int(44100*duration), False)
        sd.play(volume * np.sin(2*np.pi*freq*t), 44100, blocking=False)
    except Exception:
        pass


def _pinch_dist(lm):
    return math.hypot(lm[THUMB_TIP][0]-lm[INDEX_TIP][0],
                      lm[THUMB_TIP][1]-lm[INDEX_TIP][1])


def main():
    print('=== STARK OS — DUAL-HAND FORGE ===')
    tracker    = HandTracker()
    engine     = GestureEngine()
    two_engine = TwoHandEngine()
    swipe_r    = SwipeTracker(threshold=config.SWIPE_THRESHOLD)
    swipe_l    = SwipeTracker(threshold=config.SWIPE_THRESHOLD)
    cursor_r   = CursorSmoother(alpha=config.CURSOR_SMOOTHER_ALPHA)
    cursor_l   = CursorSmoother(alpha=config.CURSOR_SMOOTHER_ALPHA)
    menu       = MenuState()
    renderer   = UIRenderer()

    # Boot straight into the Forge
    active_app    = Modeller3D()
    active_app_id = 'modeller_3d'

    prev_time = time.time()
    fps       = 0.0

    # Edge-detection state
    _pp_r = False   # previous pinch right
    _pp_l = False   # previous pinch left
    _pf_l = False   # previous fist left

    print('Ready — show hands to the camera.')
    print('Right hand cyan  |  Left hand violet')
    print('Press q to quit.\n')

    try:
        while True:
            frame, ok = tracker.get_frame()
            if not ok or frame is None:
                continue

            h, w, _ = frame.shape
            now  = time.time()
            fps  = 0.95*fps + 0.05/(now-prev_time+1e-6)
            prev_time = now

            hands, frame = tracker.detect(frame)
            lm_r = hands.get('right')
            lm_l = hands.get('left')

            # ── Per-hand gesture + cursor ─────────────────────────────────────
            r_gest, l_gest = 'none', 'none'
            r_cur,  l_cur  = None, None
            r_pinch_d, l_pinch_d = 1.0, 1.0

            if lm_r:
                r_gest    = engine.detect_gesture(lm_r)
                tip       = lm_r[INDEX_TIP]
                r_cur     = cursor_r.update(int(tip[0]*w), int(tip[1]*h))
                r_pinch_d = _pinch_dist(lm_r)
                swipe_r.update(lm_r)
            else:
                cursor_r.reset(); swipe_r.reset()

            if lm_l:
                l_gest    = engine.detect_gesture(lm_l)
                tip       = lm_l[INDEX_TIP]
                l_cur     = cursor_l.update(int(tip[0]*w), int(tip[1]*h))
                l_pinch_d = _pinch_dist(lm_l)
                swipe_l.update(lm_l)
            else:
                cursor_l.reset(); swipe_l.reset()

            is_pinch_r = r_pinch_d < config.PINCH_THRESHOLD
            is_pinch_l = l_pinch_d < config.PINCH_THRESHOLD
            is_fist_l  = (l_gest == 'fist')

            just_pinch_r = is_pinch_r and not _pp_r
            just_pinch_l = is_pinch_l and not _pp_l
            just_fist_l  = is_fist_l  and not _pf_l

            # Swipe: check right hand
            swipe_result = 'none'
            if lm_r:
                swipe_result = swipe_r.update(lm_r) or 'none'

            # ── Two-hand engine ───────────────────────────────────────────────
            two_result = two_engine.update(lm_l, lm_r)

            # ── App dispatch ──────────────────────────────────────────────────
            if active_app is not None:
                app_gi = {
                    'left_lm':         lm_l,
                    'right_lm':        lm_r,
                    'left_gesture':    l_gest,
                    'right_gesture':   r_gest,
                    'left_cursor':     l_cur,
                    'right_cursor':    r_cur,
                    'left_pinch':      1.0 if is_pinch_l else 0.0,
                    'right_pinch':     1.0 if is_pinch_r else 0.0,
                    'just_pinch_left':  just_pinch_l,
                    'just_pinch_right': just_pinch_r,
                    'just_fist_left':   just_fist_l,
                    'two_hand':        two_result,
                    'swipe_right':     swipe_result,
                    'fps':             fps,
                }
                still_open = active_app.update_and_render(frame, app_gi)
                if not still_open:
                    # Re-launch immediately (FORGE is the only app)
                    active_app = Modeller3D()
                    play_beep(600, 0.10, 0.10)

            # ── Menu fallback (shown on top of FORGE briefly) — not used now ─
            # kept for future multi-app support

            _pp_r = is_pinch_r
            _pp_l = is_pinch_l
            _pf_l = is_fist_l

            cv2.imshow('STARK FORGE', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('\nQuitting...')
                break

    except KeyboardInterrupt:
        print('\nInterrupted.')
    finally:
        tracker.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
