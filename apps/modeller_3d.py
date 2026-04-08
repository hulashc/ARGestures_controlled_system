"""STARK FORGE — Iron Man holographic 3-D modeller (dual-hand redesign).

Two-hand gesture map
--------------------
  RIGHT hand only
    OPEN   + move          -> orbit scene  (cyan trails)
    POINT  + move          -> translate selected part (XY)
    PINCH  (tap)           -> select nearest part  /  fire toolbar button

  LEFT hand only
    OPEN   + move          -> orbit scene  (violet trails)
    POINT  + move          -> translate selected part along Z (depth)
    FIST   (hold 0.4 s)    -> confirm pending action

  BOTH hands
    Spread / close tips    -> uniform scale selected part (+ children)
    Twist line angle        -> spin selected part on Z axis
    Both FIST + drag mid   -> translate entire scene

  RIGHT swipe left         -> reset / re-launch Forge

UI highlights
  - Full-screen dark holographic canvas with JARVIS scan-line sweep
  - Corner bracket HUD
  - Dual cursor rings (cyan / violet) with pinch-progress arc
  - Two-hand connector line + midpoint beacon
  - Hand-role badge strip (L / R + gesture)
  - Collapsible info panel (top-right)
  - Bottom toolbar (12 tools, hover highlight per hand)
  - Left-side material swatch column
  - Notification banner (centre, animated)
  - Legend strip (bottom-left, faint)
"""

import cv2
import numpy as np
import math
import time

# ── Palette (BGR) ──────────────────────────────────────────────────────────────
C_BG        = (  6,  10,  14)   # near-black canvas
C_CYAN      = (  0, 230, 255)   # right hand / primary accent
C_CYAN_DIM  = (  0,  55,  60)   # inactive cyan
C_VIOLET    = (200,  80, 255)   # left hand accent
C_VIOLET_DIM= ( 45,  18,  58)   # inactive violet
C_MAGENTA   = (200,  40, 200)   # pinch confirm / attachment
C_GREEN     = ( 50, 230,  80)   # success / open-palm
C_ORANGE    = ( 30, 140, 255)   # warnings
C_WHITE     = (255, 255, 255)
C_RED       = ( 40,  40, 220)   # delete / error
C_GOLD      = ( 30, 200, 255)   # notifications
C_TEAL      = ( 20, 200, 180)   # two-hand connector
C_DARK      = (  8,  12,  16)   # panel backgrounds

MATERIALS = [
    ((  0, 230, 255), 0.07, 'TITANIUM'),
    (( 50, 230,  80), 0.07, 'CARBON'),
    ((200,  40, 200), 0.09, 'PLASMA'),
    (( 30, 140, 255), 0.08, 'COPPER'),
    ((255, 180,  20), 0.08, 'GOLD'),
    (( 20, 200, 180), 0.07, 'TEAL'),
]

_FONT  = cv2.FONT_HERSHEY_DUPLEX
_FONTS = cv2.FONT_HERSHEY_SIMPLEX


# ── Drawing helpers ────────────────────────────────────────────────────────────

def _gt(frame, text, x, y, font, scale, color, thick=1):
    """Glow text: dark shadow + bright foreground."""
    shadow = tuple(max(0, c // 5) for c in color)
    cv2.putText(frame, text, (x - 1, y + 1), font, scale, shadow, thick + 2)
    cv2.putText(frame, text, (x,     y),     font, scale, color,  thick)


def _gl(frame, p1, p2, color, thick=1, layers=2):
    """Glow line: bloom layers behind a bright core."""
    for i in range(layers, 0, -1):
        a   = 0.12 * i / layers
        dim = tuple(max(0, int(c * a)) for c in color)
        cv2.line(frame, p1, p2, dim, thick + i * 2)
    cv2.line(frame, p1, p2, color, thick)


def _ar(frame, x1, y1, x2, y2, color, alpha):
    """Alpha-blended rectangle fill."""
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    bg = np.full_like(roi, color, dtype=np.uint8)
    cv2.addWeighted(bg, alpha, roi, 1 - alpha, 0, roi)
    frame[y1:y2, x1:x2] = roi


def _hex_pts(cx, cy, r, rot=0.0):
    return [
        (int(cx + r * math.cos(math.radians(60 * i + rot))),
         int(cy + r * math.sin(math.radians(60 * i + rot))))
        for i in range(6)
    ]


def _draw_hex(frame, cx, cy, r, color, thick=1, rot=0.0):
    pts = _hex_pts(cx, cy, r, rot)
    for i in range(6):
        cv2.line(frame, pts[i], pts[(i + 1) % 6], color, thick)


def _fill_hex(frame, cx, cy, r, color, alpha=0.5, rot=0.0):
    pts  = np.array(_hex_pts(cx, cy, r, rot), np.int32)
    over = frame.copy()
    cv2.fillPoly(over, [pts], color)
    cv2.addWeighted(over, alpha, frame, 1 - alpha, 0, frame)


# ── 3-D math ───────────────────────────────────────────────────────────────────

def _rx(pts, a):
    c, s = math.cos(a), math.sin(a)
    R = np.array([[1,0,0],[0,c,-s],[0,s,c]], np.float32)
    return (R @ pts.T).T

def _ry(pts, a):
    c, s = math.cos(a), math.sin(a)
    R = np.array([[c,0,s],[0,1,0],[-s,0,c]], np.float32)
    return (R @ pts.T).T

def _rz(pts, a):
    c, s = math.cos(a), math.sin(a)
    R = np.array([[c,-s,0],[s,c,0],[0,0,1]], np.float32)
    return (R @ pts.T).T

def _proj(pts3d, cx, cy, fov=520, z_off=4.5):
    out = []
    for x, y, z in pts3d:
        zz = max(0.01, z + z_off)
        out.append((int(cx + fov * x / zz), int(cy - fov * y / zz)))
    return out


# ── Primitives ─────────────────────────────────────────────────────────────────

def _sphere(r=0.5, lat=9, lon=14):
    v, e = [], []
    for i in range(lat + 1):
        phi = math.pi * i / lat
        for j in range(lon):
            th = 2 * math.pi * j / lon
            v.append([r * math.sin(phi) * math.cos(th),
                      r * math.cos(phi),
                      r * math.sin(phi) * math.sin(th)])
    for i in range(lat):
        for j in range(lon):
            a = i*lon+j; b = i*lon+(j+1)%lon; c2 = (i+1)*lon+j
            e += [(a, b), (a, c2)]
    return np.array(v, np.float32), e

def _cube(s=0.5):
    h = s
    v = np.array([[-h,-h,-h],[h,-h,-h],[h,h,-h],[-h,h,-h],
                  [-h,-h, h],[h,-h, h],[h,h, h],[-h,h, h]], np.float32)
    e = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
         (0,4),(1,5),(2,6),(3,7)]
    return v, e

def _cylinder(r=0.4, h=0.8, segs=14):
    v, e = [], []
    top, bot = [], []
    for i in range(segs):
        a = 2 * math.pi * i / segs
        x, z = r * math.cos(a), r * math.sin(a)
        v.append([x,  h/2, z]); top.append(len(v) - 1)
        v.append([x, -h/2, z]); bot.append(len(v) - 1)
    for i in range(segs):
        ni = (i + 1) % segs
        e += [(top[i], top[ni]), (bot[i], bot[ni]), (top[i], bot[i])]
    return np.array(v, np.float32), e

def _torus(R=0.52, r=0.18, maj=16, mn=9):
    v, e = [], []
    for i in range(maj):
        phi = 2 * math.pi * i / maj
        for j in range(mn):
            th = 2 * math.pi * j / mn
            x = (R + r * math.cos(th)) * math.cos(phi)
            y = r * math.sin(th)
            z = (R + r * math.cos(th)) * math.sin(phi)
            v.append([x, y, z])
    for i in range(maj):
        ni = (i + 1) % maj
        for j in range(mn):
            nj = (j + 1) % mn
            a = i*mn+j; b = i*mn+nj; c2 = ni*mn+j
            e += [(a, b), (a, c2)]
    return np.array(v, np.float32), e

def _cone(r=0.4, h=0.8, segs=14):
    v = [[0, h/2, 0]]
    e = []
    base = []
    for i in range(segs):
        a = 2 * math.pi * i / segs
        v.append([r * math.cos(a), -h/2, r * math.sin(a)])
        base.append(len(v) - 1)
    for i in range(segs):
        ni = (i + 1) % segs
        e += [(0, base[i]), (base[i], base[ni])]
    return np.array(v, np.float32), e

def _diamond(s=0.45):
    """Octahedron accent primitive."""
    v = np.array([[0,s,0],[s,0,0],[0,0,s],[-s,0,0],[0,0,-s],[0,-s,0]], np.float32)
    e = [(0,1),(0,2),(0,3),(0,4),(5,1),(5,2),(5,3),(5,4),
         (1,2),(2,3),(3,4),(4,1)]
    return v, e

PRIMITIVES = {
    'SPHERE':   _sphere,
    'CUBE':     _cube,
    'CYLINDER': _cylinder,
    'TORUS':    _torus,
    'CONE':     _cone,
    'DIAMOND':  _diamond,
}


# ── Toolbar definition ─────────────────────────────────────────────────────────

TOOLBAR = [
    {'id': 'new_sphere',   'label': 'SPHERE',  'color': C_CYAN},
    {'id': 'new_cube',     'label': 'CUBE',    'color': C_CYAN},
    {'id': 'new_cylinder', 'label': 'CYL',     'color': C_CYAN},
    {'id': 'new_torus',    'label': 'TORUS',   'color': C_CYAN},
    {'id': 'new_cone',     'label': 'CONE',    'color': C_CYAN},
    {'id': 'new_diamond',  'label': 'DIAMOND', 'color': C_CYAN},
    {'id': 'clone',        'label': 'CLONE',   'color': C_TEAL},
    {'id': 'attach',       'label': 'ATTACH',  'color': C_TEAL},
    {'id': 'detach',       'label': 'DETACH',  'color': C_TEAL},
    {'id': 'delete',       'label': 'DELETE',  'color': C_RED},
    {'id': 'material',     'label': 'MAT',     'color': C_GOLD},
    {'id': 'reset_view',   'label': 'RESET',   'color': C_ORANGE},
]


# ── Material swatch sidebar ────────────────────────────────────────────────────

MAT_SWATCH_W  = 58
MAT_SWATCH_H  = 34
MAT_SWATCH_PAD = 5


# ── Part ───────────────────────────────────────────────────────────────────────

class Part:
    _ctr = 0

    def __init__(self, kind='CUBE', pos=None, scale=0.55, material=0):
        Part._ctr += 1
        self.pid         = Part._ctr
        self.kind        = kind
        self.pos         = np.array(pos or [0., 0., 0.], np.float32)
        self.rot         = np.array([0., 0., 0.], np.float32)  # euler XYZ
        self.scale       = float(scale)
        self.material    = material % len(MATERIALS)
        self.selected    = False
        self.attached_to = None
        self.children    = []
        self.spawn_t     = time.time()
        verts, edges     = PRIMITIVES[kind]()
        self.base_verts  = verts
        self.edges       = edges

    def world_verts(self, srx, sry, ambient_spin=0.0):
        v = self.base_verts.copy() * self.scale
        v = _ry(v, self.rot[1] + ambient_spin)
        v = _rx(v, self.rot[0])
        v = _rz(v, self.rot[2])
        v = v + self.pos
        v = _rx(v, srx)
        v = _ry(v, sry)
        return v

    @property
    def mat_color(self): return MATERIALS[self.material][0]
    @property
    def mat_name(self):  return MATERIALS[self.material][2]


# ── Modeller3D ─────────────────────────────────────────────────────────────────

class Modeller3D:
    def __init__(self):
        self.t0    = time.time()
        self.parts: list[Part] = []
        self.selected_pid = None

        # Scene rotation state
        self.srx = 0.28
        self.sry = 0.45

        # Per-hand tracking state
        self._orbit_prev = {'left': None, 'right': None}
        self._trans_prev = {'left': None, 'right': None}

        # Toolbar hit rects: (x1,y1,x2,y2,tool_id)
        self._tb_rects   = []
        self._tb_hover_r = -1
        self._tb_hover_l = -1

        # Material swatch hit rects: (x1,y1,x2,y2, mat_idx)
        self._mat_rects  = []

        # Fist-confirm state
        self._fist_t  = 0.0
        self._pending = None

        # Notification
        self._notif       = ''
        self._notif_timer = 0

        # Ambient slow spin for unselected parts
        self._amb_spin = 0.0

        # Info panel collapsed state
        self._info_collapsed = False

        # Boot with a starter cube
        self._add_part('CUBE', [0., 0., 0.])

    # ── Public entry point ─────────────────────────────────────────────────────

    def update_and_render(self, frame, gi):
        """
        Called every frame. Returns True to stay open, False to exit.
        gi keys: left_lm, right_lm, left_gesture, right_gesture,
                 left_cursor, right_cursor, left_pinch, right_pinch,
                 just_pinch_left, just_pinch_right, just_fist_left,
                 two_hand, swipe_right, fps
        """
        h, w, _ = frame.shape
        t = time.time() - self.t0
        self._amb_spin = t * 0.28

        rg   = gi.get('right_gesture', 'none')
        lg   = gi.get('left_gesture',  'none')
        rc   = gi.get('right_cursor')
        lc   = gi.get('left_cursor')
        rp   = gi.get('right_pinch',  0.)
        lp   = gi.get('left_pinch',   0.)
        jp_r = gi.get('just_pinch_right', False)
        jp_l = gi.get('just_pinch_left',  False)
        jf_l = gi.get('just_fist_left',   False)
        two  = gi.get('two_hand', {})
        swipe = gi.get('swipe_right', 'none')

        # Right swipe-left -> reset Forge
        if swipe == 'swipe_left':
            self._notify('RESETTING FORGE...')
            return False

        # ── Two-hand interactions (highest priority) ───────────────────────────
        if two.get('active'):
            sel = self._get_selected()
            if sel:
                sd = two.get('scale_delta', 0.)
                if abs(sd) > 0.001:
                    sel.scale = max(0.05, min(3.5, sel.scale + sd * 0.35))
                    for cpid in sel.children:
                        cp = self._get_part(cpid)
                        if cp:
                            cp.scale = max(0.05, min(3.5, cp.scale + sd * 0.35))

                rd = two.get('rotate_delta', 0.)
                if abs(rd) > 0.002:
                    sel.rot[2] += rd * 0.5

            tx, ty = two.get('translate_xy', (0., 0.))
            if abs(tx) > 0.001 or abs(ty) > 0.001:
                for p in self.parts:
                    p.pos[0] += tx * 2.5
                    p.pos[1] -= ty * 2.5

        # ── Toolbar hover detection ────────────────────────────────────────────
        self._tb_hover_r = -1
        self._tb_hover_l = -1
        if rc:
            for i, (x1, y1, x2, y2, _) in enumerate(self._tb_rects):
                if x1 <= rc[0] <= x2 and y1 <= rc[1] <= y2:
                    self._tb_hover_r = i
                    break
        if lc:
            for i, (x1, y1, x2, y2, _) in enumerate(self._tb_rects):
                if x1 <= lc[0] <= x2 and y1 <= lc[1] <= y2:
                    self._tb_hover_l = i
                    break

        # ── Material swatch click ──────────────────────────────────────────────
        if jp_r and rc:
            for (mx1, my1, mx2, my2, mi) in self._mat_rects:
                if mx1 <= rc[0] <= mx2 and my1 <= rc[1] <= my2:
                    sel = self._get_selected()
                    if sel:
                        sel.material = mi
                        self._notify(f'MATERIAL: {sel.mat_name}')
                    jp_r = False
                    break

        # ── Toolbar pinch ──────────────────────────────────────────────────────
        if jp_r and self._tb_hover_r >= 0:
            self._dispatch(self._tb_rects[self._tb_hover_r][4])
        elif jp_l and self._tb_hover_l >= 0:
            self._dispatch(self._tb_rects[self._tb_hover_l][4])

        # ── Part select (right pinch, not on toolbar) ──────────────────────────
        elif jp_r and rc and self._tb_hover_r < 0:
            self._try_select(rc, w, h)

        # ── LEFT hand: Z-depth translate ───────────────────────────────────────
        if lg == 'point' and lc and self.selected_pid is not None and not two.get('active'):
            prev = self._trans_prev['left']
            if prev is not None:
                dy = (lc[1] - prev[1]) / h
                sel = self._get_selected()
                if sel:
                    sel.pos[2] -= dy * 2.5
                    for cpid in sel.children:
                        cp = self._get_part(cpid)
                        if cp: cp.pos[2] -= dy * 2.5
            self._trans_prev['left'] = lc
        else:
            self._trans_prev['left'] = None

        # ── RIGHT hand: XY translate ───────────────────────────────────────────
        if rg == 'point' and rc and self.selected_pid is not None and not two.get('active'):
            prev = self._trans_prev['right']
            if prev is not None:
                dx = (rc[0] - prev[0]) / w
                dy = (rc[1] - prev[1]) / h
                sel = self._get_selected()
                if sel:
                    sel.pos[0] += dx * 3.0
                    sel.pos[1] -= dy * 3.0
                    for cpid in sel.children:
                        cp = self._get_part(cpid)
                        if cp:
                            cp.pos[0] += dx * 3.0
                            cp.pos[1] -= dy * 3.0
            self._trans_prev['right'] = rc
        else:
            self._trans_prev['right'] = None

        # ── RIGHT hand: orbit (open palm) ──────────────────────────────────────
        if rg == 'open' and rc and not two.get('active'):
            prev = self._orbit_prev['right']
            if prev is not None:
                self.sry += (rc[0] - prev[0]) / w * 3.2
                self.srx += (rc[1] - prev[1]) / h * 3.2
                self.srx  = max(-1.4, min(1.4, self.srx))
            self._orbit_prev['right'] = rc
        else:
            self._orbit_prev['right'] = None

        # ── LEFT hand: orbit (open palm) ──────────────────────────────────────
        if lg == 'open' and lc and not two.get('active'):
            prev = self._orbit_prev['left']
            if prev is not None:
                self.sry += (lc[0] - prev[0]) / w * 3.2
                self.srx += (lc[1] - prev[1]) / h * 3.2
                self.srx  = max(-1.4, min(1.4, self.srx))
            self._orbit_prev['left'] = lc
        else:
            self._orbit_prev['left'] = None

        # ── LEFT fist hold: confirm pending ───────────────────────────────────
        if lg == 'fist' and self._pending:
            self._fist_t += 1 / 30.
            if self._fist_t >= 0.4:
                self._execute(self._pending)
                self._pending = None
                self._fist_t  = 0.
        else:
            if lg != 'fist':
                self._fist_t = 0.

        # Notification timer
        if self._notif_timer > 0:
            self._notif_timer -= 1
        else:
            self._notif = ''

        # ── Render pipeline ────────────────────────────────────────────────────
        self._render_bg(frame, w, h, t)
        self._render_grid(frame, w, h)
        self._render_parts(frame, w, h, t)
        self._render_two_hand_bridge(frame, w, h, two, lc, rc, t)
        self._render_toolbar(frame, w, h, t)
        self._render_mat_swatches(frame, w, h, t)
        self._render_info_panel(frame, w, h, gi, t)
        self._render_hand_badges(frame, w, h, lc, rc, lg, rg)
        self._render_legend(frame, w, h)
        self._render_notification(frame, w, h, t)
        self._render_fist_bar(frame, w, h)
        self._render_corner_brackets(frame, w, h, t)
        if rc: self._render_cursor(frame, rc, rp, t, C_CYAN)
        if lc: self._render_cursor(frame, lc, lp, t, C_VIOLET)
        return True

    # ── Toolbar / action dispatch ──────────────────────────────────────────────

    def _dispatch(self, action):
        off = np.array([np.random.uniform(-.5, .5),
                        np.random.uniform(-.3, .3),
                        np.random.uniform(-.3, .3)], np.float32)
        kind_map = {
            'new_sphere':   'SPHERE',
            'new_cube':     'CUBE',
            'new_cylinder': 'CYLINDER',
            'new_torus':    'TORUS',
            'new_cone':     'CONE',
            'new_diamond':  'DIAMOND',
        }
        if action in kind_map:
            p = self._add_part(kind_map[action], off)
            self._notify(f'{kind_map[action]} SPAWNED  P{p.pid:02d}')
        elif action == 'clone':    self._clone()
        elif action == 'attach':   self._pending = 'attach';  self._notify('L-FIST TO CONFIRM ATTACH')
        elif action == 'detach':   self._pending = 'detach';  self._notify('L-FIST TO CONFIRM DETACH')
        elif action == 'delete':   self._pending = 'delete';  self._notify('L-FIST TO CONFIRM DELETE')
        elif action == 'material': self._cycle_mat()
        elif action == 'reset_view':
            self.srx, self.sry = 0.28, 0.45
            self._notify('VIEW RESET')

    def _execute(self, action):
        if action == 'attach':  self._do_attach()
        elif action == 'detach': self._do_detach()
        elif action == 'delete': self._do_delete()

    # ── Part operations ────────────────────────────────────────────────────────

    def _add_part(self, kind, pos, mat=None):
        m = mat if mat is not None else len(self.parts) % len(MATERIALS)
        p = Part(kind, list(pos), scale=0.55, material=m)
        self.parts.append(p)
        self.selected_pid = p.pid
        for pp in self.parts: pp.selected = (pp.pid == p.pid)
        return p

    def _get_part(self, pid):
        for p in self.parts:
            if p.pid == pid: return p
        return None

    def _get_selected(self):
        return self._get_part(self.selected_pid)

    def _try_select(self, cursor, w, h):
        cx, cy = w // 2, h // 2 + 30
        best_d, best_pid = 80 * 80, None
        for p in self.parts:
            v   = p.world_verts(self.srx, self.sry)
            cen = v.mean(axis=0)
            pp  = _proj([cen], cx, cy)[0]
            d   = (cursor[0] - pp[0]) ** 2 + (cursor[1] - pp[1]) ** 2
            if d < best_d: best_d, best_pid = d, p.pid
        for p in self.parts: p.selected = False
        if best_pid:
            self.selected_pid = best_pid
            self._get_part(best_pid).selected = True
            self._notify(f'SELECTED  P{best_pid:02d}')
        else:
            self.selected_pid = None

    def _clone(self):
        s = self._get_selected()
        if not s: self._notify('NO PART SELECTED'); return
        p = self._add_part(s.kind, s.pos + np.array([.22, .15, 0], np.float32), s.material)
        p.rot[:] = s.rot
        self._notify(f'CLONED -> P{p.pid:02d}')

    def _do_attach(self):
        sel = self._get_selected()
        if not sel or len(self.parts) < 2: self._notify('NEED 2+ PARTS'); return
        best_d, best_p = 1e9, None
        for p in self.parts:
            if p.pid == sel.pid: continue
            d = float(np.linalg.norm(sel.pos - p.pos))
            if d < best_d: best_d, best_p = d, p
        if best_p:
            sel.attached_to = best_p.pid
            if sel.pid not in best_p.children: best_p.children.append(sel.pid)
            self._notify(f'P{sel.pid:02d} ATTACHED -> P{best_p.pid:02d}')

    def _do_detach(self):
        sel = self._get_selected()
        if not sel or not sel.attached_to: self._notify('NOT ATTACHED'); return
        parent = self._get_part(sel.attached_to)
        if parent and sel.pid in parent.children: parent.children.remove(sel.pid)
        sel.attached_to = None
        self._notify(f'P{sel.pid:02d} DETACHED')

    def _do_delete(self):
        sel = self._get_selected()
        if not sel: self._notify('NO PART SELECTED'); return
        if sel.attached_to:
            parent = self._get_part(sel.attached_to)
            if parent and sel.pid in parent.children: parent.children.remove(sel.pid)
        for cpid in sel.children:
            cp = self._get_part(cpid)
            if cp: cp.attached_to = None
        pid = sel.pid
        self.parts = [p for p in self.parts if p.pid != pid]
        self.selected_pid = self.parts[-1].pid if self.parts else None
        self._notify(f'P{pid:02d} DELETED')

    def _cycle_mat(self):
        sel = self._get_selected()
        if not sel: self._notify('NO PART SELECTED'); return
        sel.material = (sel.material + 1) % len(MATERIALS)
        self._notify(f'MATERIAL: {sel.mat_name}')

    def _notify(self, msg, frames=100):
        self._notif       = msg
        self._notif_timer = frames

    # ══════════════════════════════════════════════════════════════════════════
    # RENDER PIPELINE
    # ══════════════════════════════════════════════════════════════════════════

    def _render_bg(self, frame, w, h, t):
        """Dark holographic canvas: dim the live feed + scan-line sweep."""
        dark = np.full_like(frame, C_BG, np.uint8)
        cv2.addWeighted(dark, 0.32, frame, 0.68, 0, frame)

        # Horizontal scan-line grid (faint)
        for y in range(0, h, 5):
            frame[y] = (frame[y] * 0.90).astype(np.uint8)

        # Moving bright sweep band
        sy = int((t * 72) % h)
        for dy in range(-22, 22):
            y = sy + dy
            if 0 <= y < h:
                a   = 1.0 - abs(dy) / 22
                add = tuple(int(c * a * 0.10) for c in C_CYAN)
                frame[y] = np.clip(frame[y].astype(np.int32) + add, 0, 255)

        # Title bar
        _ar(frame, 0, 0, w, 44, C_DARK, 0.85)
        cv2.line(frame, (0, 44), (w, 44),
                 tuple(int(c * .40) for c in C_CYAN), 1)
        pulse = .55 + .45 * math.sin(t * 1.8)
        _gt(frame,
            'STARK  FORGE   //   HOLOGRAPHIC MODELLER   //   DUAL-HAND',
            14, 28, _FONTS, 0.46,
            tuple(int(c * pulse) for c in C_CYAN), 1)
        # Hand colour legend in title bar
        _gt(frame, 'L=VIOLET', w - 158, 28, _FONTS, 0.36,
            tuple(int(c * .55) for c in C_VIOLET), 1)
        _gt(frame, 'R=CYAN',   w - 72,  28, _FONTS, 0.36,
            tuple(int(c * .55) for c in C_CYAN), 1)

    def _render_grid(self, frame, w, h):
        cx, cy = w // 2, h // 2 + 55
        gc  = tuple(int(c * .12) for c in C_CYAN)
        ext, step = 5, 0.42
        for gx in range(-ext, ext + 1):
            pts = [np.array([[gx * step, -.75, z * step]]) for z in (-ext, ext)]
            ps  = [_proj([_ry(_rx(p, self.srx), self.sry)[0]], cx, cy)[0] for p in pts]
            cv2.line(frame, ps[0], ps[1], gc, 1)
        for gz in range(-ext, ext + 1):
            pts = [np.array([[x * step, -.75, gz * step]]) for x in (-ext, ext)]
            ps  = [_proj([_ry(_rx(p, self.srx), self.sry)[0]], cx, cy)[0] for p in pts]
            cv2.line(frame, ps[0], ps[1], gc, 1)

    def _render_parts(self, frame, w, h, t):
        cx, cy = w // 2, h // 2 + 30

        def zkey(p):
            return -p.world_verts(self.srx, self.sry)[:, 2].mean()

        for part in sorted(self.parts, key=zkey):
            age   = t - (part.spawn_t - self.t0)
            entry = min(1.0, age * 5.0)
            spin  = self._amb_spin * .22 if not part.selected else 0.
            v3d   = part.world_verts(self.srx, self.sry, spin)
            if entry < 1.: v3d = v3d * entry
            prj   = _proj(v3d, cx, cy)
            mc    = part.mat_color
            issel = (part.pid == self.selected_pid)

            for (a, b) in part.edges:
                if a >= len(prj) or b >= len(prj): continue
                p1, p2 = prj[a], prj[b]
                if not (0 <= p1[0] < w and 0 <= p1[1] < h): continue
                if not (0 <= p2[0] < w and 0 <= p2[1] < h): continue
                bright = 1.0 if issel else 0.55
                c = tuple(int(ch * bright) for ch in mc)
                _gl(frame, p1, p2, c,
                    2 if issel else 1,
                    layers=2 if issel else 1)

            # Centroid label
            cen3 = v3d.mean(axis=0)
            cp   = _proj([cen3], cx, cy)[0]
            lbl  = f'[P{part.pid:02d}]' if issel else f'P{part.pid:02d}'
            _gt(frame, lbl, cp[0] - 16, cp[1] - 8, _FONTS, 0.36,
                tuple(int(c * .75) for c in mc), 1)

            # Attachment bond line
            if part.attached_to:
                par = self._get_part(part.attached_to)
                if par:
                    pv  = par.world_verts(self.srx, self.sry)
                    pcp = _proj([pv.mean(axis=0)], cx, cy)[0]
                    bc  = tuple(int(c * .45) for c in C_MAGENTA)
                    cv2.line(frame, cp, pcp, bc, 1)
                    mid = ((cp[0] + pcp[0]) // 2, (cp[1] + pcp[1]) // 2)
                    cv2.circle(frame, mid, 3, bc, -1)

            # Selection ring
            if issel:
                rr    = int(42 * entry)
                spin2 = t * 190
                for seg in range(6):
                    a_s = spin2 + seg * 60
                    a_e = a_s + 26
                    cv2.ellipse(frame, cp, (rr, rr), 0, a_s, a_e, C_CYAN, 2)
                # Inner counter-spin
                for seg in range(4):
                    a_s = -spin2 * .6 + seg * 90
                    a_e = a_s + 20
                    cv2.ellipse(frame, cp, (rr + 6, rr + 6), 0, a_s, a_e,
                                tuple(int(c * .4) for c in C_VIOLET), 1)
                # Scale / rotation readout
                _gt(frame,
                    f'S:{part.scale:.2f}  Rz:{math.degrees(part.rot[2]):.0f}\xb0',
                    cp[0] + rr + 8, cp[1], _FONTS, 0.32,
                    tuple(int(c * .52) for c in C_CYAN), 1)

    def _render_two_hand_bridge(self, frame, w, h, two, lc, rc, t):
        """Visualise the two-hand connector when both cursors are live."""
        if not (two.get('active') and lc and rc):
            return
        # Bridge line
        _gl(frame, lc, rc, tuple(int(c * .45) for c in C_TEAL), 1, 1)
        # Midpoint beacon
        mid = ((lc[0] + rc[0]) // 2, (lc[1] + rc[1]) // 2)
        r_beacon = 7 + int(3 * math.sin(t * 6))
        cv2.circle(frame, mid, r_beacon, C_TEAL, 1)
        cv2.circle(frame, mid, 3, C_TEAL, -1)
        # Span readout
        dist = math.hypot(rc[0] - lc[0], rc[1] - lc[1])
        _gt(frame, f'SPAN {dist:.0f}px', mid[0] - 32, mid[1] - 14,
            _FONTS, 0.34, C_TEAL, 1)
        # Scale / rotate hints
        sel = self._get_selected()
        if sel:
            _gt(frame, f'SCALE {sel.scale:.2f}x',
                mid[0] - 32, mid[1] + 16, _FONTS, 0.30,
                tuple(int(c * .6) for c in C_TEAL), 1)

    def _render_toolbar(self, frame, w, h, t):
        """Bottom toolbar — 12 tool buttons, colour-coded per function group."""
        n     = len(TOOLBAR)
        btn_w = max(52, (w - 20) // n)
        btn_h = 44
        y1    = h - btn_h - 8
        y2    = h - 8

        # Background bar
        _ar(frame, 0, y1 - 10, w, h, C_DARK, 0.86)
        cv2.line(frame, (0, y1 - 10), (w, y1 - 10),
                 tuple(int(c * .32) for c in C_CYAN), 1)

        self._tb_rects = []
        for i, tool in enumerate(TOOLBAR):
            x1 = 10 + i * btn_w
            x2 = x1 + btn_w - 4
            if x2 > w - 4: break

            hr   = (self._tb_hover_r == i)
            hl   = (self._tb_hover_l == i)
            base = tool['color']
            pulse = .5 + .5 * math.sin(t * 3.2 + i * .9)

            # Button fill
            fill_a = .28 if (hr or hl) else .07
            _ar(frame, x1, y1, x2, y2, base, fill_a)

            # Border
            bc = tuple(int(c * (pulse if (hr or hl) else .30)) for c in base)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bc, 1)

            # Active hand indicator dot (top-left corner of button)
            if hr:
                cv2.circle(frame, (x1 + 5, y1 + 5), 3, C_CYAN, -1)
            if hl:
                cv2.circle(frame, (x1 + 5, y1 + 5), 3, C_VIOLET, -1)

            # Label
            lbl = tool['label']
            (tw, th), _ = cv2.getTextSize(lbl, _FONTS, 0.33, 1)
            tx = x1 + (btn_w - tw) // 2
            ty = y1 + (btn_h + th) // 2 - 2
            tc = tuple(int(c * (1.0 if (hr or hl) else .55)) for c in base)
            _gt(frame, lbl, tx, ty, _FONTS, 0.33, tc, 1)

            # Fist-confirm progress bar (bottom of button)
            if self._pending == tool['id'] and self._fist_t > 0:
                prog = min(1., self._fist_t / 0.4)
                cv2.rectangle(frame, (x1, y2 - 3),
                              (x1 + int((x2 - x1) * prog), y2),
                              C_MAGENTA, -1)

            self._tb_rects.append((x1, y1, x2, y2, tool['id']))

    def _render_mat_swatches(self, frame, w, h, t):
        """Left-side material swatch column — click with right pinch to apply."""
        sw  = MAT_SWATCH_W
        sh  = MAT_SWATCH_H
        pad = MAT_SWATCH_PAD
        x1  = 6
        x2  = x1 + sw
        sel = self._get_selected()

        self._mat_rects = []
        for mi, (color, _, name) in enumerate(MATERIALS):
            y1 = 54 + mi * (sh + pad)
            y2 = y1 + sh

            active = sel and sel.material == mi
            _ar(frame, x1, y1, x2, y2, color,
                .40 if active else .12)
            border_c = tuple(int(c * (1.0 if active else .35)) for c in color)
            cv2.rectangle(frame, (x1, y1), (x2, y2), border_c, 1)
            if active:
                # Animated left edge bar
                pulse = .6 + .4 * math.sin(t * 4)
                cv2.rectangle(frame, (x1, y1), (x1 + 3, y2),
                              tuple(int(c * pulse) for c in color), -1)

            # Material name
            _gt(frame, name, x1 + 6, y1 + sh - 8, _FONTS, 0.26,
                tuple(int(c * (.85 if active else .45)) for c in color), 1)

            self._mat_rects.append((x1, y1, x2, y2, mi))

    def _render_info_panel(self, frame, w, h, gi, t):
        """Top-right info panel — part properties + scene state + FPS."""
        sel   = self._get_selected()
        px    = w - 190
        py    = 52
        pan_h = 192

        _ar(frame, px - 10, py - 6, w - 4, py + pan_h, C_DARK, 0.80)
        cv2.rectangle(frame, (px - 10, py - 6), (w - 4, py + pan_h),
                      tuple(int(c * .24) for c in C_CYAN), 1)

        lines = [
            ('PARTS',   str(len(self.parts))),
            ('SEL',     f'P{sel.pid:02d}' if sel else 'NONE'),
            ('KIND',    sel.kind if sel else '---'),
            ('MAT',     sel.mat_name if sel else '---'),
            ('SCALE',   f'{sel.scale:.2f}' if sel else '---'),
            ('Rz',      f'{math.degrees(sel.rot[2]):.0f}\xb0' if sel else '---'),
            ('ATTACH',  f'P{sel.attached_to:02d}' if sel and sel.attached_to else 'FREE'),
            ('ORBIT X', f'{math.degrees(self.srx):.0f}\xb0'),
            ('ORBIT Y', f'{math.degrees(self.sry):.0f}\xb0'),
            ('FPS',     f"{gi.get('fps', 0):.0f}"),
        ]
        for i, (key, val) in enumerate(lines):
            ky = py + i * 19
            _gt(frame, f'{key:<8}', px,      ky, _FONTS, 0.33,
                tuple(int(c * .38) for c in C_CYAN), 1)
            _gt(frame, val,          px + 72, ky, _FONTS, 0.33,
                tuple(int(c * .72) for c in C_CYAN), 1)

    def _render_hand_badges(self, frame, w, h, lc, rc, lg, rg):
        """Small floating badges near each cursor showing hand role + gesture."""
        badge_data = [
            (lc, 'L', lg, C_VIOLET),
            (rc, 'R', rg, C_CYAN),
        ]
        for cur, side, gest, col in badge_data:
            if cur is None:
                continue
            bx = max(4, min(w - 90, cur[0] - 44))
            by = max(18, min(h - 10, cur[1] + 42))
            label = f'{side}  {gest.upper()}'
            _gt(frame, label, bx, by, _FONTS, 0.34,
                tuple(int(c * .68) for c in col), 1)

    def _render_legend(self, frame, w, h):
        """Faint control legend above the toolbar."""
        tb_top = h - 44 - 8 - 10
        legend = [
            'R OPEN=ORBIT   R POINT=MOVE XY   L POINT=MOVE Z',
            'R PINCH=SELECT/TOOLBAR   L FIST=CONFIRM   BOTH=SCALE/SPIN/DRAG',
        ]
        for i, line in enumerate(legend):
            cv2.putText(frame, line,
                        (MAT_SWATCH_W + 16, tb_top - 18 + i * 13),
                        _FONTS, 0.26,
                        tuple(int(c * .20) for c in C_CYAN), 1)

    def _render_notification(self, frame, w, h, t):
        """Centre-screen notification banner with pulsing gold text."""
        if not self._notif:
            return
        pulse = .65 + .35 * math.sin(t * 7)
        nc    = tuple(int(c * pulse) for c in C_GOLD)
        (nw, _), _ = cv2.getTextSize(self._notif, _FONTS, 0.52, 1)
        nx = (w - nw) // 2
        ny = h - 100
        # Semi-transparent pill behind text
        _ar(frame, nx - 12, ny - 18, nx + nw + 12, ny + 8, C_DARK, 0.72)
        _gt(frame, self._notif, nx, ny, _FONTS, 0.52, nc, 1)

    def _render_fist_bar(self, frame, w, h):
        """Left-fist hold progress bar and pending action label."""
        if self._fist_t <= 0:
            return
        prog = min(1., self._fist_t / 0.4)
        bw   = int(w // 4 * prog)
        cv2.rectangle(frame, (14, 52), (14 + bw, 60), C_MAGENTA, -1)
        _gt(frame,
            f'CONFIRM: {self._pending.upper() if self._pending else ""}',
            18, 76, _FONTS, 0.36, C_MAGENTA, 1)

    def _render_corner_brackets(self, frame, w, h, t):
        """Iron Man HUD corner brackets."""
        pulse = .65 + .35 * math.sin(t * 1.6)
        c     = tuple(int(ch * pulse) for ch in C_CYAN)
        L, T, pad = 38, 2, 14
        corners = [
            ((pad, pad),         ( 1,  1)),
            ((w - pad, pad),     (-1,  1)),
            ((pad, h - pad),     ( 1, -1)),
            ((w - pad, h - pad), (-1, -1)),
        ]
        for (ox, oy), (dx, dy) in corners:
            cv2.line(frame, (ox, oy), (ox + dx * L, oy), c, T)
            cv2.line(frame, (ox, oy), (ox, oy + dy * L), c, T)
            cv2.line(frame, (ox + dx * 5, oy + dy * 5),
                            (ox + dx * 12, oy + dy * 5), c, 1)
            cv2.line(frame, (ox + dx * 5, oy + dy * 5),
                            (ox + dx * 5, oy + dy * 12), c, 1)

    def _render_cursor(self, frame, cursor, pinch_p, t, color):
        """JARVIS-style spinning cursor ring for one hand."""
        x, y  = int(cursor[0]), int(cursor[1])
        spin  = t * 130
        r_out = 20 + int(4 * math.sin(t * 4))

        # Centre dot
        cv2.circle(frame, (x, y), 3, C_WHITE, -1)

        # Inner spinning hex
        pts = []
        for i in range(6):
            ang = math.radians(60 * i + spin % 360)
            pts.append((int(x + 10 * math.cos(ang)),
                         int(y + 10 * math.sin(ang))))
        for i in range(6):
            cv2.line(frame, pts[i], pts[(i + 1) % 6], color, 1)

        # Outer dashed ring segments
        for seg in range(8):
            a_s = spin * .65 + seg * 45
            a_e = a_s + 18
            cv2.ellipse(frame, (x, y), (r_out, r_out), 0, a_s, a_e, color, 2)

        # Counter-rotating accent ring
        opp = tuple(int(c * .5) for c in
                    (C_VIOLET if color == C_CYAN else C_CYAN))
        for seg in range(4):
            a_s = -spin * .45 + seg * 90
            a_e = a_s + 22
            cv2.ellipse(frame, (x, y), (r_out + 7, r_out + 7),
                        0, a_s, a_e, opp, 1)

        # Pinch progress arc
        if pinch_p > 0.05:
            cv2.ellipse(frame, (x, y), (r_out + 4, r_out + 4),
                        0, -90, -90 + int(360 * pinch_p), C_MAGENTA, 2)

        # Crosshair ticks
        gap = r_out + 5
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            cv2.line(frame,
                     (x + dx * gap,     y + dy * gap),
                     (x + dx * (gap+6), y + dy * (gap+6)),
                     color, 1)
