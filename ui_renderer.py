"""JARVIS-style Iron Man holographic UI renderer.

Design language:
  - Arc reactor blue/cyan palette with magenta accents
  - Hexagonal panel geometry
  - Animated scan lines and rotating rings
  - Glowing text with layered draw passes
  - Radial menu layout for submenu depth
  - Full alpha-composited overlay pipeline
"""

import cv2
import numpy as np
import math
import time
import config
from utils import ParticleSystem


# ── Colour palette (BGR) ────────────────────────────────────────────────────
C_ARC       = (255, 210, 50)    # arc-reactor gold-white core
C_CYAN      = (230, 220, 20)    # primary holographic cyan
C_CYAN_DIM  = (100, 90,  8)     # dimmed cyan for inactive
C_MAGENTA   = (200, 40, 200)    # pinch / accent
C_BLUE      = (200, 120, 10)    # deep blue tint
C_WHITE     = (255, 255, 255)
C_DARK      = (8,   12,  16)    # near-black bg tint
C_GREEN     = (50,  230, 80)
C_RED       = (40,  40,  220)
C_ORANGE    = (30,  140, 255)

# Per-item accent colours (BGR) — one per root menu item
ITEM_ACCENTS = [
    (230, 220, 20),   # cyan
    (50,  230, 80),   # green
    (30,  140, 255),  # orange
    (200, 40, 200),   # magenta
    (255, 180, 20),   # sky-blue
    (40,  200, 160),  # teal-gold
]

_FONT  = cv2.FONT_HERSHEY_DUPLEX
_FONTM = cv2.FONT_HERSHEY_SIMPLEX


# ── Low-level drawing helpers ───────────────────────────────────────────────

def _alpha_rect(frame, x1, y1, x2, y2, color, alpha):
    """Fill a rectangle with per-pixel alpha blend."""
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    bg = np.full_like(roi, color, dtype=np.uint8)
    cv2.addWeighted(bg, alpha, roi, 1.0 - alpha, 0, roi)
    frame[y1:y2, x1:x2] = roi


def _glow_circle(frame, cx, cy, r, color, thickness=1, layers=3):
    """Draw a circle with an outward glow bloom."""
    for i in range(layers, 0, -1):
        a = 0.12 * i / layers
        dim = tuple(max(0, int(c * a)) for c in color)
        cv2.circle(frame, (cx, cy), r + i * 2, dim, thickness + i)
    cv2.circle(frame, (cx, cy), r, color, thickness)


def _glow_line(frame, p1, p2, color, thickness=1, layers=2):
    for i in range(layers, 0, -1):
        a = 0.15 * i / layers
        dim = tuple(max(0, int(c * a)) for c in color)
        cv2.line(frame, p1, p2, dim, thickness + i * 2)
    cv2.line(frame, p1, p2, color, thickness)


def _glow_text(frame, text, x, y, font, scale, color, thickness=1):
    """Render text with a soft bloom behind it."""
    dim = tuple(max(0, c // 4) for c in color)
    cv2.putText(frame, text, (x - 1, y + 1), font, scale, dim, thickness + 2)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness)


def _hex_points(cx, cy, r, rotation=0.0):
    """Return 6 integer (x,y) vertices of a regular hexagon."""
    pts = []
    for i in range(6):
        angle = math.radians(60 * i + rotation)
        pts.append((int(cx + r * math.cos(angle)),
                    int(cy + r * math.sin(angle))))
    return pts


def _draw_hex(frame, cx, cy, r, color, thickness=1, rotation=0.0):
    pts = _hex_points(cx, cy, r, rotation)
    for i in range(6):
        p1 = pts[i]
        p2 = pts[(i + 1) % 6]
        cv2.line(frame, p1, p2, color, thickness)


def _fill_hex(frame, cx, cy, r, color, alpha=0.5, rotation=0.0):
    pts = np.array(_hex_points(cx, cy, r, rotation), dtype=np.int32)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ── Main renderer ────────────────────────────────────────────────────────────

class UIRenderer:
    def __init__(self):
        self.t0 = time.time()
        self.frame_count = 0
        self.particles = ParticleSystem()
        self.tile_rects = []

        # Per-tile hover animation state (0..1 float)
        self._hover_anim   = {}   # idx -> float (0=idle, 1=full hover)
        # Per-tile activation flash state
        self._flash_anim   = {}   # idx -> float (0=done, 1=peak)
        # Transition slide interpolation
        self._trans_alpha  = 0.0  # 0=hidden, 1=visible
        self._trans_dir    = 1    # 1=forward, -1=back
        # Cursor trail
        self._cursor_trail = []   # list of (x,y)
        self._trail_max    = 18
        # Scan line vertical offset
        self._scan_y       = 0
        # Arc reactor spin angle
        self._arc_angle    = 0.0
        # Waveform noise for ambient decoration
        self._wave_phase   = 0.0

    # ── Public API ──────────────────────────────────────────────────────────

    def render(self, frame, menu, gesture_info):
        self.frame_count += 1
        h, w, _ = frame.shape
        t = time.time() - self.t0
        self.tile_rects = []

        # 1. Ambient dark vignette
        self._draw_vignette(frame, w, h)

        # 2. Global scan-line sweep
        self._draw_scanlines(frame, w, h, t)

        # 3. Corner HUD brackets
        self._draw_corner_brackets(frame, w, h, t)

        # 4. Ambient arc rings (always spinning)
        self._draw_ambient_rings(frame, w, h, t)

        # 5. Menu panels
        info = menu.render_info()
        menu.update(hand_present=gesture_info.get("hand_detected", False))
        self._draw_menu(frame, w, h, info, gesture_info, t)

        # 6. Cursor (always on top of menu)
        cursor = gesture_info.get("cursor")
        if cursor:
            self._update_trail(cursor)
            self._draw_cursor_trail(frame, t)
            self._draw_cursor(frame, cursor, gesture_info.get("pinch_progress", 0), t)

        # 7. HUD overlay (status, gesture label, fps)
        self._draw_hud(frame, w, h, gesture_info, t)

        # 8. Particles
        self.particles.update_and_draw(frame)

    def get_tile_rects(self):
        return self.tile_rects

    def emit_particles(self, x, y, color, count=30):
        # Emit a burst of holographic sparks
        self.particles.emit(x, y, C_CYAN, count // 2)
        self.particles.emit(x, y, C_MAGENTA, count // 2)

    # ── Vignette ────────────────────────────────────────────────────────────

    def _draw_vignette(self, frame, w, h):
        """Subtle dark border vignette to focus attention to center."""
        vig = np.zeros((h, w), dtype=np.uint8)
        cx, cy = w // 2, h // 2
        r_outer = int(math.hypot(cx, cy))
        for r in range(r_outer, 0, -max(1, r_outer // 24)):
            alpha_v = int(80 * (1.0 - r / r_outer) ** 2)
            cv2.circle(vig, (cx, cy), r, alpha_v, max(1, r_outer // 24))
        vignette_bgr = cv2.merge([vig, vig, vig])
        frame[:] = cv2.subtract(frame, vignette_bgr)

    # ── Scan lines ──────────────────────────────────────────────────────────

    def _draw_scanlines(self, frame, w, h, t):
        """Moving horizontal sweep line with a faint grid of static lines."""
        # Static faint horizontal lines
        step = 6
        for y in range(0, h, step):
            frame[y, :] = (frame[y, :] * 0.88).astype(np.uint8)

        # Moving bright sweep band
        sweep_y = int((t * 90) % h)
        band = 28
        for dy in range(-band, band):
            sy = sweep_y + dy
            if 0 <= sy < h:
                alpha = 1.0 - abs(dy) / band
                bright = tuple(int(c * alpha * 0.18) for c in C_CYAN)
                frame[sy] = np.clip(frame[sy].astype(np.int32) + bright, 0, 255).astype(np.uint8)

    # ── Corner HUD brackets ──────────────────────────────────────────────────

    def _draw_corner_brackets(self, frame, w, h, t):
        """Iron Man HUD corner brackets with subtle pulse."""
        pulse = 0.7 + 0.3 * math.sin(t * 1.8)
        c = tuple(int(ch * pulse) for ch in C_CYAN)
        L = 40   # bracket arm length
        T = 2    # thickness
        pad = 12
        corners = [
            ((pad, pad),         ( 1,  1)),
            ((w - pad, pad),     (-1,  1)),
            ((pad, h - pad),     ( 1, -1)),
            ((w - pad, h - pad), (-1, -1)),
        ]
        for (ox, oy), (dx, dy) in corners:
            cv2.line(frame, (ox, oy), (ox + dx * L, oy), c, T)
            cv2.line(frame, (ox, oy), (ox, oy + dy * L), c, T)
            # inner tick
            cv2.line(frame, (ox + dx * 6, oy + dy * 6),
                     (ox + dx * 14, oy + dy * 6), c, 1)
            cv2.line(frame, (ox + dx * 6, oy + dy * 6),
                     (ox + dx * 6, oy + dy * 14), c, 1)

    # ── Ambient rings ───────────────────────────────────────────────────────

    def _draw_ambient_rings(self, frame, w, h, t):
        """Slow-spinning decorative rings in the background."""
        cx, cy = w // 2, h // 2
        # Outer slow ring
        angle1 = t * 12  # degrees/sec
        r1 = min(w, h) // 2 - 20
        for seg in range(24):
            a_start = angle1 + seg * 15
            a_end   = a_start + 8
            dim = C_CYAN_DIM
            cv2.ellipse(frame, (cx, cy), (r1, r1), 0,
                        a_start, a_end, dim, 1)

        # Inner faster ring
        angle2 = -t * 22
        r2 = r1 - 30
        for seg in range(12):
            a_start = angle2 + seg * 30
            a_end   = a_start + 14
            cv2.ellipse(frame, (cx, cy), (r2, r2), 0,
                        a_start, a_end, C_CYAN_DIM, 1)

        # Very subtle hex grid overlay in center
        for row in range(-2, 3):
            for col in range(-2, 3):
                hx = cx + col * 52 + (row % 2) * 26
                hy = cy + row * 44
                _draw_hex(frame, hx, hy, 22,
                          tuple(int(c * 0.06) for c in C_CYAN), 1, 30)

    # ── Menu panels ─────────────────────────────────────────────────────────

    def _draw_menu(self, frame, w, h, info, gesture_info, t):
        items    = info["items"]
        depth    = info["depth"]
        trans_t  = info["transition_timer"]
        trans_type = info["transition_type"]
        hovered  = gesture_info.get("hovered_idx", -1)

        if not items:
            return

        # Transition progress (0 = fully visible, counts down from 20)
        TRANS_FRAMES = 20
        trans_p = trans_t / TRANS_FRAMES if trans_t > 0 else 0.0
        if trans_type == "enter":
            slide_x = int(w * 0.08 * trans_p)        # slide in from right
            alpha_p = 1.0 - trans_p * 0.6
        elif trans_type == "exit":
            slide_x = int(-w * 0.08 * trans_p)       # slide in from left
            alpha_p = 1.0 - trans_p * 0.6
        else:
            slide_x = 0
            alpha_p = 1.0

        n = len(items)
        cols = 3 if n > 4 else (2 if n > 1 else 1)
        rows = math.ceil(n / cols)

        HEX_R  = 64       # hex circumradius
        HEX_PADDING = 18  # gap between hexagons
        col_step = int(HEX_R * 1.75) + HEX_PADDING
        row_step = int(HEX_R * 1.52) + HEX_PADDING

        grid_w = cols * col_step - HEX_PADDING
        grid_h = rows * row_step - HEX_PADDING

        origin_x = (w - grid_w) // 2 + slide_x
        origin_y = (h - grid_h) // 2 + 10

        self.tile_rects = []

        for i, item in enumerate(items):
            col = i % cols
            row = i // cols
            cx = origin_x + col * col_step + HEX_R
            cy = origin_y + row * row_step + HEX_R

            # Hover animation
            prev_h = self._hover_anim.get(i, 0.0)
            target_h = 1.0 if hovered == i else 0.0
            new_h = prev_h + (target_h - prev_h) * 0.18
            self._hover_anim[i] = new_h

            # Activation flash
            if item.activated:
                self._flash_anim[i] = 1.0
            prev_f = self._flash_anim.get(i, 0.0)
            new_f = max(0.0, prev_f - 0.04)
            self._flash_anim[i] = new_f

            self._draw_hex_panel(
                frame, cx, cy, HEX_R, i, item,
                new_h, new_f, alpha_p, t, depth
            )

            # Hit rect (bounding box of hexagon)
            self.tile_rects.append((
                cx - HEX_R, cy - HEX_R,
                cx + HEX_R, cy + HEX_R
            ))

        # Breadcrumb path + depth indicator
        self._draw_breadcrumb(frame, w, h, info, t)

        # Hovered item description strip
        if hovered >= 0 and hovered < len(items):
            item = items[hovered]
            if item.desc:
                self._draw_desc_strip(frame, w, h, item, t)

    def _draw_hex_panel(self, frame, cx, cy, r, idx, item,
                        hover_t, flash_t, alpha, t, depth):
        """Draw one holographic hexagonal menu tile."""
        accent = ITEM_ACCENTS[idx % len(ITEM_ACCENTS)]
        spin   = t * (8 + idx * 3) * (1 if idx % 2 == 0 else -1)

        # ── Layer 1: filled hex background ──────────────────────────────────
        base_alpha = 0.10 + 0.18 * hover_t + 0.35 * flash_t
        fill_color = accent if flash_t < 0.05 else C_WHITE
        _fill_hex(frame, cx, cy, r - 2, fill_color,
                  alpha=base_alpha * alpha, rotation=30)

        # ── Layer 2: outer hex border (spinning dashed effect) ───────────────
        border_alpha = 0.5 + 0.5 * hover_t
        border_color = tuple(int(c * border_alpha * alpha) for c in accent)
        _draw_hex(frame, cx, cy, r, border_color, 1, 30)

        # On hover: second bright hex inside
        if hover_t > 0.05:
            inner_c = tuple(int(c * hover_t * alpha) for c in accent)
            _draw_hex(frame, cx, cy, r - 5, inner_c, 1, 30)
            # Glow corners
            for pt in _hex_points(cx, cy, r, 30):
                glow_r = int(3 + 2 * hover_t)
                _glow_circle(frame, pt[0], pt[1], glow_r, accent, 1, 2)

        # On flash: ring burst
        if flash_t > 0.05:
            burst_r = int(r + (r * 0.5 * (1.0 - flash_t)))
            burst_c = tuple(int(c * flash_t) for c in C_WHITE)
            cv2.circle(frame, (cx, cy), burst_r, burst_c, 2)
            cv2.circle(frame, (cx, cy), burst_r + 5,
                       tuple(int(c * flash_t * 0.4) for c in C_CYAN), 1)

        # ── Layer 3: spinning inner decorative ring ──────────────────────────
        ring_r = r - 18
        segs = 8
        for s in range(segs):
            a_s = spin + s * (360 / segs)
            a_e = a_s + 360 / segs * 0.55
            ring_c = tuple(int(c * 0.35 * alpha) for c in accent)
            cv2.ellipse(frame, (cx, cy), (ring_r, ring_r),
                        0, a_s, a_e, ring_c, 1)

        # ── Layer 4: centre dot / arc reactor node ───────────────────────────
        node_r = 5 + int(3 * hover_t)
        node_c = tuple(int(c * (0.6 + 0.4 * hover_t) * alpha) for c in accent)
        cv2.circle(frame, (cx, cy), node_r, node_c, -1)
        if hover_t > 0.1:
            _glow_circle(frame, cx, cy, node_r, accent, 1, 2)

        # ── Layer 5: label text ──────────────────────────────────────────────
        label = item.name.upper()
        scale = 0.52
        thick = 1
        (tw, th), _ = cv2.getTextSize(label, _FONT, scale, thick)
        tx = cx - tw // 2
        ty = cy + th // 2 + 18
        text_c = tuple(int(c * (0.55 + 0.45 * hover_t) * alpha) for c in accent)
        _glow_text(frame, label, tx, ty, _FONT, scale, text_c, thick)

        # Submenu indicator: small arrow above label
        if item.is_submenu:
            arr_cx = cx
            arr_y  = cy - r + 14
            arr_pts = np.array([
                [arr_cx - 6, arr_y + 6],
                [arr_cx + 6, arr_y + 6],
                [arr_cx,     arr_y - 2],
            ], dtype=np.int32)
            cv2.fillPoly(frame, [arr_pts],
                         tuple(int(c * 0.7 * alpha) for c in accent))

    def _draw_breadcrumb(self, frame, w, h, info, t):
        """Path indicator at the top with animated underline."""
        breadcrumb = info["breadcrumb"]
        depth      = info["depth"]

        # Top bar
        bar_h = 38
        _alpha_rect(frame, 0, 0, w, bar_h, C_DARK, 0.72)
        _glow_line(frame, (0, bar_h), (w, bar_h), C_CYAN_DIM, 1)

        # Breadcrumb text
        parts = breadcrumb.split(" / ")
        x = 18
        for pi, part in enumerate(parts):
            is_last = (pi == len(parts) - 1)
            c = C_CYAN if is_last else tuple(int(ch * 0.45) for ch in C_CYAN)
            scale = 0.52 if is_last else 0.42
            _glow_text(frame, part, x, 24, _FONTM, scale, c, 1)
            (pw, _), _ = cv2.getTextSize(part, _FONTM, scale, 1)
            x += pw
            if not is_last:
                x += 4
                _glow_text(frame, "  /", x, 24, _FONTM, 0.38,
                           tuple(int(ch * 0.3) for ch in C_CYAN), 1)
                x += 20

        # Animated underline under active breadcrumb item
        pulse = 0.5 + 0.5 * math.sin(t * 3.5)
        und_w = min(120, w - 18)
        und_c = tuple(int(c * pulse) for c in C_CYAN)
        _glow_line(frame, (18, bar_h - 2), (18 + und_w, bar_h - 2), und_c, 1)

        # Depth dots (right side)
        for d in range(depth + 1):
            dot_x = w - 20 - d * 14
            dot_y = bar_h // 2
            if d == depth:
                _glow_circle(frame, dot_x, dot_y, 4, C_CYAN, -1, 2)
            else:
                cv2.circle(frame, (dot_x, dot_y), 3, C_CYAN_DIM, -1)

    def _draw_desc_strip(self, frame, w, h, item, t):
        """Semi-transparent strip at bottom with hovered item description."""
        strip_h = 36
        y1 = h - strip_h
        _alpha_rect(frame, 0, y1, w, h, C_DARK, 0.75)
        _glow_line(frame, (0, y1), (w, y1), C_CYAN_DIM, 1)

        desc = item.desc
        (tw, _), _ = cv2.getTextSize(desc, _FONTM, 0.46, 1)
        tx = (w - tw) // 2
        pulse = 0.75 + 0.25 * math.sin(t * 2.2)
        c = tuple(int(ch * pulse) for ch in C_CYAN)
        _glow_text(frame, desc, tx, y1 + 22, _FONTM, 0.46, c, 1)

        # Small icon marker
        marker = ">> " if item.is_submenu else ">> ACTION: "
        c2 = tuple(int(ch * 0.55) for ch in C_MAGENTA)
        _glow_text(frame, marker, 18, y1 + 22, _FONTM, 0.42, c2, 1)

    # ── Cursor ───────────────────────────────────────────────────────────────

    def _update_trail(self, cursor):
        self._cursor_trail.append((int(cursor[0]), int(cursor[1])))
        if len(self._cursor_trail) > self._trail_max:
            self._cursor_trail.pop(0)

    def _draw_cursor_trail(self, frame, t):
        n = len(self._cursor_trail)
        for i in range(1, n):
            alpha = (i / n) ** 1.8
            c = tuple(int(ch * alpha * 0.6) for ch in C_CYAN)
            thickness = max(1, int(3 * alpha))
            cv2.line(frame, self._cursor_trail[i - 1],
                     self._cursor_trail[i], c, thickness)

    def _draw_cursor(self, frame, cursor, pinch_progress, t):
        x, y = int(cursor[0]), int(cursor[1])
        spin  = t * 120   # degrees/sec

        # ── Trailing glow dot ────────────────────────────────────────────────
        _glow_circle(frame, x, y, 3, C_WHITE, -1, 2)

        # ── Inner spinning hex ───────────────────────────────────────────────
        _draw_hex(frame, x, y, 12, C_CYAN, 1, spin % 360)

        # ── Outer ring (dashed segments) ─────────────────────────────────────
        outer_r = 22 + int(4 * math.sin(t * 4))
        for seg in range(8):
            a_s = spin * 0.7 + seg * 45
            a_e = a_s + 22
            cv2.ellipse(frame, (x, y), (outer_r, outer_r),
                        0, a_s, a_e, C_CYAN, 2)

        # ── Counter-rotating outer arc ───────────────────────────────────────
        for seg in range(4):
            a_s = -spin * 0.4 + seg * 90
            a_e = a_s + 30
            cv2.ellipse(frame, (x, y), (outer_r + 8, outer_r + 8),
                        0, a_s, a_e,
                        tuple(int(c * 0.5) for c in C_MAGENTA), 1)

        # ── Pinch progress arc ───────────────────────────────────────────────
        if pinch_progress > 0.05:
            sweep = int(360 * pinch_progress)
            _glow_line(frame, (x, y - outer_r - 12), (x, y - outer_r - 4),
                       C_MAGENTA, 1)
            cv2.ellipse(frame, (x, y), (outer_r + 4, outer_r + 4),
                        0, -90, -90 + sweep, C_MAGENTA, 2)

        # ── Crosshair ticks ──────────────────────────────────────────────────
        tick_len = 6
        gap      = outer_r + 4
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            p1 = (x + dx * gap, y + dy * gap)
            p2 = (x + dx * (gap + tick_len), y + dy * (gap + tick_len))
            _glow_line(frame, p1, p2, C_CYAN, 1, 1)

    # ── HUD overlay ──────────────────────────────────────────────────────────

    def _draw_hud(self, frame, w, h, gesture_info, t):
        """Bottom-left and bottom-right HUD readouts."""
        hand     = gesture_info.get("hand_detected", False)
        gesture  = gesture_info.get("gesture", "none").upper()
        action   = gesture_info.get("action", "")
        fps_val  = gesture_info.get("fps", 0)

        # ── Status indicator (top-right) ─────────────────────────────────────
        pulse = 0.6 + 0.4 * math.sin(t * 3.0)
        if hand:
            sc = tuple(int(c * pulse) for c in C_GREEN)
            st = "TRACKING"
        else:
            sc = tuple(int(c * pulse) for c in C_RED)
            st = "SEARCHING..."
        _glow_text(frame, st, w - 160, 24, _FONTM, 0.46, sc, 1)

        # Small blinking dot
        if hand:
            dot_r = 4 + int(2 * math.sin(t * 6))
            _glow_circle(frame, w - 170, 19, dot_r, sc, -1, 2)

        # ── FPS (top-right corner) ────────────────────────────────────────────
        fps_c = tuple(int(c * 0.55) for c in C_CYAN)
        _glow_text(frame, f"{fps_val:.0f} FPS", w - 78, h - 14,
                   _FONTM, 0.38, fps_c, 1)

        # ── Gesture label (bottom-left) ───────────────────────────────────────
        gc = {
            "PINCH":  C_MAGENTA,
            "FIST":   C_RED,
            "POINT":  C_GREEN,
            "OPEN":   C_CYAN,
        }.get(gesture, tuple(int(c * 0.4) for c in C_CYAN))
        _glow_text(frame, gesture, 18, h - 46, _FONTM, 0.50, gc, 1)

        # ── Action flash ─────────────────────────────────────────────────────
        if action:
            pulse_a = 0.7 + 0.3 * math.sin(t * 8)
            ac = tuple(int(c * pulse_a) for c in C_ORANGE)
            (aw, _), _ = cv2.getTextSize(action, _FONT, 0.60, 1)
            ax = (w - aw) // 2
            _glow_text(frame, action, ax, h - 52, _FONT, 0.60, ac, 1)
            # Flanking decorators
            _glow_line(frame, (ax - 20, h - 54), (ax - 8, h - 54), ac, 1)
            _glow_line(frame, (ax + aw + 8, h - 54), (ax + aw + 20, h - 54), ac, 1)

        # ── Minimal data stream (right edge) ─────────────────────────────────
        data_lines = [
            f"DEPTH: {gesture_info.get('depth', 0)}",
            f"HAND : {'Y' if hand else 'N'}",
            f"PINCH: {gesture_info.get('pinch_progress', 0):.2f}",
        ]
        for li, line in enumerate(data_lines):
            lc = tuple(int(c * 0.28) for c in C_CYAN)
            _glow_text(frame, line, w - 128, 58 + li * 18, _FONTM, 0.34, lc, 1)
