"""Stark Forge — Iron Man holographic 3-D modeller.

Two-hand gesture map
--------------------
  RIGHT hand only
    OPEN  + move   -> orbit scene
    POINT + move   -> translate selected part (XY)
    PINCH (tap)    -> select nearest part / toolbar button

  LEFT hand only
    OPEN  + move   -> orbit scene (same as right — either hand orbits)
    POINT + move   -> translate selected part in Z (depth)
    FIST  (hold)   -> confirm destructive action

  BOTH hands
    Spread / close index tips  -> scale selected part
    Twist (rotate line)        -> spin selected part on Z axis
    Both FIST + drag midpoint  -> translate whole scene

  RIGHT hand swipe left  -> exit Forge back to menu
"""

import cv2
import numpy as np
import math
import time

# ── Palette (BGR) ──────────────────────────────────────────────────────────────
C_CYAN      = (230, 220,  20)
C_CYAN_DIM  = ( 55,  50,   5)
C_MAGENTA   = (200,  40, 200)
C_GREEN     = ( 50, 230,  80)
C_ORANGE    = ( 30, 140, 255)
C_WHITE     = (255, 255, 255)
C_RED       = ( 40,  40, 220)
C_GOLD      = ( 30, 200, 255)
C_VIOLET    = (200,  80, 255)   # left-hand colour
C_DARK      = (  6,  10,  14)
C_TEAL      = ( 20, 200, 180)

MATERIALS = [
    ((230, 220,  20), 0.07, 'TITANIUM'),
    (( 50, 230,  80), 0.07, 'CARBON'),
    ((200,  40, 200), 0.09, 'PLASMA'),
    (( 30, 140, 255), 0.08, 'COPPER'),
    ((255, 180,  20), 0.08, 'GOLD'),
    (( 20, 200, 180), 0.07, 'TEAL'),
]

_FONT  = cv2.FONT_HERSHEY_DUPLEX
_FONTM = cv2.FONT_HERSHEY_SIMPLEX


# ── Drawing helpers ────────────────────────────────────────────────────────────

def _gt(frame, text, x, y, font, scale, color, thick=1):
    dim = tuple(max(0, c//5) for c in color)
    cv2.putText(frame, text, (x-1, y+1), font, scale, dim, thick+2)
    cv2.putText(frame, text, (x, y),     font, scale, color, thick)

def _gl(frame, p1, p2, color, thick=1, layers=2):
    for i in range(layers, 0, -1):
        a   = 0.12 * i / layers
        dim = tuple(max(0, int(c*a)) for c in color)
        cv2.line(frame, p1, p2, dim, thick + i*2)
    cv2.line(frame, p1, p2, color, thick)

def _ar(frame, x1, y1, x2, y2, color, alpha):
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0: return
    bg = np.full_like(roi, color, dtype=np.uint8)
    cv2.addWeighted(bg, alpha, roi, 1-alpha, 0, roi)
    frame[y1:y2, x1:x2] = roi

def _hex_pts(cx, cy, r, rot=0.0):
    return [(int(cx + r*math.cos(math.radians(60*i+rot))),
             int(cy + r*math.sin(math.radians(60*i+rot)))) for i in range(6)]

def _draw_hex(frame, cx, cy, r, color, thick=1, rot=0.0):
    pts = _hex_pts(cx, cy, r, rot)
    for i in range(6): cv2.line(frame, pts[i], pts[(i+1)%6], color, thick)

def _fill_hex(frame, cx, cy, r, color, alpha=0.5, rot=0.0):
    pts  = np.array(_hex_pts(cx, cy, r, rot), np.int32)
    over = frame.copy()
    cv2.fillPoly(over, [pts], color)
    cv2.addWeighted(over, alpha, frame, 1-alpha, 0, frame)


# ── 3-D math ───────────────────────────────────────────────────────────────────

def _rx(pts, a):
    c,s = math.cos(a), math.sin(a)
    R = np.array([[1,0,0],[0,c,-s],[0,s,c]], np.float32)
    return (R @ pts.T).T

def _ry(pts, a):
    c,s = math.cos(a), math.sin(a)
    R = np.array([[c,0,s],[0,1,0],[-s,0,c]], np.float32)
    return (R @ pts.T).T

def _rz(pts, a):
    c,s = math.cos(a), math.sin(a)
    R = np.array([[c,-s,0],[s,c,0],[0,0,1]], np.float32)
    return (R @ pts.T).T

def _proj(pts3d, cx, cy, fov=520, z_off=4.5):
    out = []
    for x,y,z in pts3d:
        zz = max(0.01, z + z_off)
        out.append((int(cx + fov*x/zz), int(cy - fov*y/zz)))
    return out


# ── Primitives ─────────────────────────────────────────────────────────────────

def _sphere(r=0.5, lat=9, lon=14):
    v, e = [], []
    for i in range(lat+1):
        phi = math.pi*i/lat
        for j in range(lon):
            th = 2*math.pi*j/lon
            v.append([r*math.sin(phi)*math.cos(th),
                      r*math.cos(phi),
                      r*math.sin(phi)*math.sin(th)])
    for i in range(lat):
        for j in range(lon):
            a = i*lon+j; b = i*lon+(j+1)%lon; c2 = (i+1)*lon+j
            e += [(a,b),(a,c2)]
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
        a = 2*math.pi*i/segs
        x,z = r*math.cos(a), r*math.sin(a)
        v.append([x, h/2, z]); top.append(len(v)-1)
        v.append([x,-h/2, z]); bot.append(len(v)-1)
    for i in range(segs):
        ni = (i+1)%segs
        e += [(top[i],top[ni]),(bot[i],bot[ni]),(top[i],bot[i])]
    return np.array(v, np.float32), e

def _torus(R=0.52, r=0.18, maj=16, mn=9):
    v, e = [], []
    for i in range(maj):
        phi = 2*math.pi*i/maj
        for j in range(mn):
            th = 2*math.pi*j/mn
            x = (R+r*math.cos(th))*math.cos(phi)
            y = r*math.sin(th)
            z = (R+r*math.cos(th))*math.sin(phi)
            v.append([x,y,z])
    for i in range(maj):
        ni = (i+1)%maj
        for j in range(mn):
            nj = (j+1)%mn
            a=i*mn+j; b=i*mn+nj; c2=ni*mn+j
            e += [(a,b),(a,c2)]
    return np.array(v, np.float32), e

def _cone(r=0.4, h=0.8, segs=14):
    v, e = [[[0, h/2, 0]]], []
    base = []
    for i in range(segs):
        a = 2*math.pi*i/segs
        v.append([r*math.cos(a),-h/2,r*math.sin(a)])
        base.append(len(v)-1)
    for i in range(segs):
        ni=(i+1)%segs
        e += [(0,base[i]),(base[i],base[ni])]
    return np.array(v, np.float32), e

def _diamond(s=0.45):
    """Octahedron — looks great as an accent primitive."""
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


# ── Part ───────────────────────────────────────────────────────────────────────

class Part:
    _ctr = 0

    def __init__(self, kind='CUBE', pos=None, scale=0.55, material=0):
        Part._ctr += 1
        self.pid      = Part._ctr
        self.kind     = kind
        self.pos      = np.array(pos or [0.,0.,0.], np.float32)
        self.rot      = np.array([0.,0.,0.], np.float32)  # euler XYZ
        self.scale    = float(scale)
        self.material = material % len(MATERIALS)
        self.selected = False
        self.attached_to = None
        self.children    = []
        self.spawn_t     = time.time()
        verts, edges = PRIMITIVES[kind]()
        self.base_verts = verts
        self.edges      = edges

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


# ── Toolbar ────────────────────────────────────────────────────────────────────

TOOLBAR = [
    {'id': 'new_sphere',   'label': 'SPHERE'},
    {'id': 'new_cube',     'label': 'CUBE'},
    {'id': 'new_cylinder', 'label': 'CYL'},
    {'id': 'new_torus',    'label': 'TORUS'},
    {'id': 'new_cone',     'label': 'CONE'},
    {'id': 'new_diamond',  'label': 'DIAMOND'},
    {'id': 'clone',        'label': 'CLONE'},
    {'id': 'attach',       'label': 'ATTACH'},
    {'id': 'detach',       'label': 'DETACH'},
    {'id': 'delete',       'label': 'DELETE'},
    {'id': 'material',     'label': 'MAT'},
    {'id': 'reset_view',   'label': 'RESET'},
]


# ── Modeller ───────────────────────────────────────────────────────────────────

class Modeller3D:
    def __init__(self):
        self.t0    = time.time()
        self.parts: list[Part] = []
        self.selected_pid = None

        # Scene rotation
        self.srx = 0.28
        self.sry = 0.45

        # Per-hand orbit state
        self._orbit_prev  = {'left': None, 'right': None}
        self._trans_prev  = {'left': None, 'right': None}

        # Toolbar
        self._tb_rects  = []
        self._tb_hover_r = -1   # right-hand hover index
        self._tb_hover_l = -1   # left-hand hover index

        # Fist confirm
        self._fist_t    = 0.0
        self._pending   = None

        # Pinch edge detection (per hand)
        self._prev_pinch = {'left': False, 'right': False}

        # Notification
        self._notif       = ''
        self._notif_timer = 0

        # Ambient slow spin for unselected parts
        self._amb_spin = 0.0

        # Spawn starter cube
        self._add_part('CUBE', [0., 0., 0.])

    # ── Public entry point ─────────────────────────────────────────────────────

    def update_and_render(self, frame, gesture_info):
        """
        Called every frame.  Returns True to stay open, False to exit.
        gesture_info keys:
          left_lm, right_lm     : landmark lists (or None)
          left_gesture           : str
          right_gesture          : str
          left_cursor            : (x,y) px or None
          right_cursor           : (x,y) px or None
          left_pinch             : float 0-1
          right_pinch            : float 0-1
          just_pinch_left        : bool
          just_pinch_right       : bool
          just_fist_left         : bool
          two_hand               : dict from TwoHandEngine.update()
          swipe_right            : str
          fps                    : float
        """
        h, w, _ = frame.shape
        t = time.time() - self.t0
        self._amb_spin = t * 0.28

        rg   = gesture_info.get('right_gesture', 'none')
        lg   = gesture_info.get('left_gesture',  'none')
        rc   = gesture_info.get('right_cursor')
        lc   = gesture_info.get('left_cursor')
        rp   = gesture_info.get('right_pinch', 0.)
        lp   = gesture_info.get('left_pinch',  0.)
        jp_r = gesture_info.get('just_pinch_right', False)
        jp_l = gesture_info.get('just_pinch_left',  False)
        jf_l = gesture_info.get('just_fist_left',   False)
        two  = gesture_info.get('two_hand', {})
        swipe = gesture_info.get('swipe_right', 'none')

        # Exit
        if swipe == 'swipe_left':
            self._notify('EXITING FORGE')
            return False

        # ── Two-hand interactions (highest priority) ───────────────────────────
        if two.get('active'):
            sel = self._get_selected()
            if sel:
                # Scale: two-hand spread / close
                sd = two.get('scale_delta', 0.)
                if abs(sd) > 0.001:
                    sel.scale = max(0.05, min(3.0, sel.scale + sd * 0.35))
                    for cpid in sel.children:
                        cp = self._get_part(cpid)
                        if cp:
                            cp.scale = max(0.05, min(3.0, cp.scale + sd * 0.35))

                # Rotate Z: two-hand twist
                rd = two.get('rotate_delta', 0.)
                if abs(rd) > 0.002:
                    sel.rot[2] += rd * 0.5

            # Both-fist drag: translate whole scene offset
            tx, ty = two.get('translate_xy', (0., 0.))
            if abs(tx) > 0.001 or abs(ty) > 0.001:
                for p in self.parts:
                    p.pos[0] += tx * 2.5
                    p.pos[1] -= ty * 2.5

        # ── Toolbar hover ──────────────────────────────────────────────────────
        self._tb_hover_r = -1
        self._tb_hover_l = -1
        if rc:
            for i, (x1,y1,x2,y2,tid) in enumerate(self._tb_rects):
                if x1 <= rc[0] <= x2 and y1 <= rc[1] <= y2:
                    self._tb_hover_r = i; break
        if lc:
            for i, (x1,y1,x2,y2,tid) in enumerate(self._tb_rects):
                if x1 <= lc[0] <= x2 and y1 <= lc[1] <= y2:
                    self._tb_hover_l = i; break

        # ── Toolbar pinch (right or left) ──────────────────────────────────────
        if jp_r and self._tb_hover_r >= 0:
            self._dispatch(self._tb_rects[self._tb_hover_r][4])
        elif jp_l and self._tb_hover_l >= 0:
            self._dispatch(self._tb_rects[self._tb_hover_l][4])

        # ── Part select (right pinch, not on toolbar) ──────────────────────────
        elif jp_r and rc and self._tb_hover_r < 0:
            self._try_select(rc, w, h)

        # ── LEFT hand: Z-depth translate of selected part ─────────────────────
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

        # ── RIGHT hand: XY translate of selected part ─────────────────────────
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
                self.sry += (rc[0]-prev[0])/w * 3.2
                self.srx += (rc[1]-prev[1])/h * 3.2
                self.srx  = max(-1.4, min(1.4, self.srx))
            self._orbit_prev['right'] = rc
        else:
            self._orbit_prev['right'] = None

        # ── LEFT hand: orbit also (open palm, single-hand use)
        if lg == 'open' and lc and not two.get('active'):
            prev = self._orbit_prev['left']
            if prev is not None:
                self.sry += (lc[0]-prev[0])/w * 3.2
                self.srx += (lc[1]-prev[1])/h * 3.2
                self.srx  = max(-1.4, min(1.4, self.srx))
            self._orbit_prev['left'] = lc
        else:
            self._orbit_prev['left'] = None

        # ── LEFT fist hold: confirm pending ───────────────────────────────────
        if lg == 'fist' and self._pending:
            self._fist_t += 1/30.
            if self._fist_t >= 0.4:
                self._execute(self._pending)
                self._pending = None
                self._fist_t  = 0.
        else:
            if lg != 'fist': self._fist_t = 0.

        # Notification timer
        if self._notif_timer > 0:
            self._notif_timer -= 1
        else:
            self._notif = ''

        # ── Render ─────────────────────────────────────────────────────────────
        self._bg(frame, w, h, t)
        self._draw_grid(frame, w, h)
        self._draw_parts(frame, w, h, t)
        self._draw_hand_indicators(frame, w, h, two, lc, rc, lg, rg, t)
        self._draw_toolbar(frame, w, h, t)
        self._draw_hud(frame, w, h, gesture_info, t)
        if rc: self._draw_cursor(frame, rc, rp, t, C_CYAN)
        if lc: self._draw_cursor(frame, lc, lp, t, C_VIOLET)
        return True

    # ── Toolbar dispatch ───────────────────────────────────────────────────────

    def _dispatch(self, action):
        off = np.array([np.random.uniform(-.5,.5),
                        np.random.uniform(-.3,.3),
                        np.random.uniform(-.3,.3)], np.float32)
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
        elif action == 'attach':   self._pending = 'attach';  self._notify('LEFT FIST TO CONFIRM ATTACH')
        elif action == 'detach':   self._pending = 'detach';  self._notify('LEFT FIST TO CONFIRM DETACH')
        elif action == 'delete':   self._pending = 'delete';  self._notify('LEFT FIST TO CONFIRM DELETE')
        elif action == 'material': self._cycle_mat()
        elif action == 'reset_view':
            self.srx, self.sry = 0.28, 0.45
            self._notify('VIEW RESET')

    def _execute(self, action):
        if action == 'attach': self._do_attach()
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
        cx, cy = w//2, h//2+30
        best_d, best_pid = 80*80, None
        for p in self.parts:
            v = p.world_verts(self.srx, self.sry)
            cen = v.mean(axis=0)
            pp  = _proj([cen], cx, cy)[0]
            d   = (cursor[0]-pp[0])**2 + (cursor[1]-pp[1])**2
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
        p = self._add_part(s.kind, s.pos + np.array([.22,.15,0], np.float32), s.material)
        p.rot[:] = s.rot
        self._notify(f'CLONED -> P{p.pid:02d}')

    def _do_attach(self):
        sel = self._get_selected()
        if not sel or len(self.parts) < 2: self._notify('NEED 2 PARTS'); return
        best_d, best_p = 1e9, None
        for p in self.parts:
            if p.pid == sel.pid: continue
            d = np.linalg.norm(sel.pos - p.pos)
            if d < best_d: best_d, best_p = d, p
        if best_p:
            sel.attached_to = best_p.pid
            if sel.pid not in best_p.children: best_p.children.append(sel.pid)
            self._notify(f'P{sel.pid:02d} ATTACHED TO P{best_p.pid:02d}')

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

    def _notify(self, msg):
        self._notif = msg
        self._notif_timer = 90

    # ── Rendering ──────────────────────────────────────────────────────────────

    def _bg(self, frame, w, h, t):
        dark = np.full_like(frame, (6,10,14), np.uint8)
        cv2.addWeighted(dark, 0.30, frame, 0.70, 0, frame)
        # sweep line
        sy = int((t*75) % h)
        for dy in range(-20, 20):
            y = sy+dy
            if 0 <= y < h:
                a = 1.0 - abs(dy)/20
                add = tuple(int(c*a*0.09) for c in C_CYAN)
                frame[y] = np.clip(frame[y].astype(np.int32)+add, 0, 255)
        # title bar
        _ar(frame, 0, 0, w, 40, C_DARK, 0.82)
        cv2.line(frame, (0,40), (w,40), tuple(int(c*.45) for c in C_CYAN), 1)
        pulse = .6 + .4*math.sin(t*2.)
        _gt(frame, 'STARK FORGE  //  HOLOGRAPHIC MODELLER  //  DUAL-HAND MODE',
            14, 26, _FONTM, 0.48, tuple(int(c*pulse) for c in C_CYAN), 1)

    def _draw_grid(self, frame, w, h):
        cx, cy = w//2, h//2+60
        gc = tuple(int(c*.15) for c in C_CYAN)
        ext, step = 5, 0.42
        for gx in range(-ext, ext+1):
            pts = [np.array([[gx*step, -.75, z*step]]) for z in (-ext, ext)]
            ps  = [_proj([_ry(_rx(p, self.srx), self.sry)[0]], cx, cy)[0] for p in pts]
            cv2.line(frame, ps[0], ps[1], gc, 1)
        for gz in range(-ext, ext+1):
            pts = [np.array([[x*step, -.75, gz*step]]) for x in (-ext, ext)]
            ps  = [_proj([_ry(_rx(p, self.srx), self.sry)[0]], cx, cy)[0] for p in pts]
            cv2.line(frame, ps[0], ps[1], gc, 1)

    def _draw_parts(self, frame, w, h, t):
        cx, cy = w//2, h//2+30

        def zkey(p):
            return -p.world_verts(self.srx, self.sry)[:,2].mean()

        for part in sorted(self.parts, key=zkey):
            age   = t - (part.spawn_t - self.t0)
            entry = min(1.0, age * 5.0)
            spin  = self._amb_spin * .22 if not part.selected else 0.
            v3d   = part.world_verts(self.srx, self.sry, spin)
            if entry < 1.: v3d = v3d * entry
            prj   = _proj(v3d, cx, cy)
            mc    = part.mat_color
            issel = (part.pid == self.selected_pid)

            for (a,b) in part.edges:
                if a >= len(prj) or b >= len(prj): continue
                p1, p2 = prj[a], prj[b]
                if not (0<=p1[0]<w and 0<=p1[1]<h): continue
                if not (0<=p2[0]<w and 0<=p2[1]<h): continue
                c = tuple(int(ch*(1. if issel else .6)) for ch in mc)
                _gl(frame, p1, p2, c, 2 if issel else 1,
                    layers=2 if issel else 1)

            # centroid label
            cen3 = v3d.mean(axis=0)
            cp   = _proj([cen3], cx, cy)[0]
            lbl  = f'[P{part.pid:02d}]' if issel else f'P{part.pid:02d}'
            _gt(frame, lbl, cp[0]-16, cp[1]-8, _FONTM, 0.36,
                tuple(int(c*.8) for c in mc), 1)

            # attachment bond
            if part.attached_to:
                par = self._get_part(part.attached_to)
                if par:
                    pv  = par.world_verts(self.srx, self.sry)
                    pcp = _proj([pv.mean(axis=0)], cx, cy)[0]
                    lc2 = tuple(int(c*.5) for c in C_MAGENTA)
                    cv2.line(frame, cp, pcp, lc2, 1)
                    mid = ((cp[0]+pcp[0])//2, (cp[1]+pcp[1])//2)
                    cv2.circle(frame, mid, 3, lc2, -1)

            # selection ring
            if issel:
                rr   = int(42*entry)
                spin2 = t*190
                for seg in range(6):
                    a_s = spin2+seg*60; a_e = a_s+26
                    cv2.ellipse(frame, cp, (rr,rr), 0, a_s, a_e, C_CYAN, 2)
                # scale/rot readout
                _gt(frame, f'S:{part.scale:.2f}  Rz:{math.degrees(part.rot[2]):.0f}d',
                    cp[0]+rr+6, cp[1], _FONTM, 0.32,
                    tuple(int(c*.55) for c in C_CYAN), 1)

    def _draw_hand_indicators(self, frame, w, h, two, lc, rc, lg, rg, t):
        """Visual bridge between the two hands when both are active."""
        if two.get('active') and lc and rc:
            # Line connecting both index tips
            _gl(frame, lc, rc, tuple(int(c*.4) for c in C_TEAL), 1, 1)
            # Midpoint dot
            mid = ((lc[0]+rc[0])//2, (lc[1]+rc[1])//2)
            cv2.circle(frame, mid, 5, C_TEAL, -1)
            # Scale bar label
            dist = math.hypot(rc[0]-lc[0], rc[1]-lc[1])
            _gt(frame, f'SPAN: {dist:.0f}px', mid[0]-30, mid[1]-14,
                _FONTM, 0.35, C_TEAL, 1)

        # Hand role badges near wrists (drawn under cursor so rendered first)
        badge_data = [
            (lc, lg, 'L  ' + lg.upper(), C_VIOLET),
            (rc, rg, 'R  ' + rg.upper(), C_CYAN),
        ]
        for cur, gest, label, col in badge_data:
            if cur:
                bx, by = cur[0]-40, cur[1]+38
                bx = max(4, min(w-80, bx))
                by = max(20, min(h-10, by))
                _gt(frame, label, bx, by, _FONTM, 0.36,
                    tuple(int(c*.7) for c in col), 1)

    def _draw_toolbar(self, frame, w, h, t):
        n     = len(TOOLBAR)
        btn_w = max(52, (w-16)//n)
        btn_h = 42
        y1    = h - btn_h - 6
        y2    = h - 6
        _ar(frame, 0, y1-8, w, h, C_DARK, 0.84)
        cv2.line(frame, (0,y1-8), (w,y1-8),
                 tuple(int(c*.35) for c in C_CYAN), 1)
        self._tb_rects = []
        for i, tool in enumerate(TOOLBAR):
            x1 = 8 + i*btn_w
            x2 = x1 + btn_w - 3
            if x2 > w-4: break
            hr = (self._tb_hover_r == i)
            hl = (self._tb_hover_l == i)
            pulse = .5 + .5*math.sin(t*3.+i*.8)
            base  = C_GOLD if hr else (C_VIOLET if hl else C_CYAN)
            _ar(frame, x1, y1, x2, y2, base, .22 if (hr or hl) else .07)
            bc = tuple(int(c*(pulse if (hr or hl) else .35)) for c in base)
            cv2.rectangle(frame, (x1,y1), (x2,y2), bc, 1)
            lbl = tool['label']
            (tw,th),_ = cv2.getTextSize(lbl, _FONTM, 0.34, 1)
            tx = x1 + (btn_w-tw)//2
            ty = y1 + (btn_h+th)//2 - 2
            tc = tuple(int(c*(1. if (hr or hl) else .6)) for c in base)
            _gt(frame, lbl, tx, ty, _FONTM, 0.34, tc, 1)
            # fist confirm bar
            if self._pending == tool['id'] and self._fist_t > 0:
                prog = min(1., self._fist_t/0.4)
                cv2.rectangle(frame, (x1,y2-3),
                              (x1+int((x2-x1)*prog),y2), C_MAGENTA, -1)
            self._tb_rects.append((x1,y1,x2,y2,tool['id']))

    def _draw_hud(self, frame, w, h, gi, t):
        sel = self._get_selected()
        px, py = w-185, 48
        ph = 175
        _ar(frame, px-8, py-8, w-4, py+ph, C_DARK, 0.78)
        cv2.rectangle(frame, (px-8,py-8), (w-4,py+ph),
                      tuple(int(c*.28) for c in C_CYAN), 1)
        lines = [
            f"PARTS : {len(self.parts)}",
            f"SEL   : {'P'+str(sel.pid).zfill(2) if sel else 'NONE'}",
            f"KIND  : {sel.kind if sel else '---'}",
            f"MAT   : {sel.mat_name if sel else '---'}",
            f"SCALE : {sel.scale:.2f}" if sel else "SCALE : ---",
            f"ATTACH: {'P'+str(sel.attached_to).zfill(2) if sel and sel.attached_to else 'FREE'}",
            f"RX    : {math.degrees(self.srx):.0f}d",
            f"RY    : {math.degrees(self.sry):.0f}d",
            f"FPS   : {gi.get('fps',0):.0f}",
        ]
        for i, line in enumerate(lines):
            _gt(frame, line, px, py+i*19, _FONTM, 0.36,
                tuple(int(c*.52) for c in C_CYAN), 1)

        # Notification
        if self._notif:
            pulse = .7+.3*math.sin(t*6)
            nc = tuple(int(c*pulse) for c in C_GOLD)
            (nw,_),_ = cv2.getTextSize(self._notif, _FONTM, 0.50, 1)
            _gt(frame, self._notif, (w-nw)//2, h-72, _FONTM, 0.50, nc, 1)

        # Fist hold bar
        if self._fist_t > 0:
            prog = min(1., self._fist_t/0.4)
            bw   = int(w//4 * prog)
            cv2.rectangle(frame, (14, 68), (14+bw, 76), C_MAGENTA, -1)
            _gt(frame, f'CONFIRM: {self._pending.upper() if self._pending else ""}',
                18, 90, _FONTM, 0.38, C_MAGENTA, 1)

        # Legend (bottom-left above toolbar)
        legend = [
            'R OPEN   = ORBIT',
            'R POINT  = MOVE XY',
            'L POINT  = MOVE Z',
            'R PINCH  = SELECT / TOOLBAR',
            'L FIST   = CONFIRM',
            'BOTH     = SCALE / SPIN / DRAG',
            'R SWIPE< = EXIT',
        ]
        for i, line in enumerate(legend):
            cv2.putText(frame, line, (14, 90+i*15),
                        _FONTM, 0.28, tuple(int(c*.22) for c in C_CYAN), 1)

    def _draw_cursor(self, frame, cursor, pinch_p, t, color):
        x, y  = int(cursor[0]), int(cursor[1])
        spin  = t*130
        r_out = 18 + int(3*math.sin(t*4))
        cv2.circle(frame, (x,y), 3, C_WHITE, -1)
        # spinning hex
        pts = []
        for i in range(6):
            ang = math.radians(60*i + spin%360)
            pts.append((int(x+9*math.cos(ang)), int(y+9*math.sin(ang))))
        for i in range(6): cv2.line(frame, pts[i], pts[(i+1)%6], color, 1)
        # outer ring segments
        for seg in range(8):
            a_s = spin*.65+seg*45; a_e = a_s+16
            cv2.ellipse(frame, (x,y), (r_out,r_out), 0, a_s, a_e, color, 2)
        # pinch fill arc
        if pinch_p > 0.05:
            cv2.ellipse(frame, (x,y), (r_out+5,r_out+5),
                        0, -90, -90+int(360*pinch_p), C_MAGENTA, 2)
