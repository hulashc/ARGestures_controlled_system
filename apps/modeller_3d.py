"""Iron Man Holographic 3D Modeller
============================================================
A gesture-driven 3D part-assembly workbench rendered entirely
with OpenCV + NumPy — no OpenGL dependency required.

Gesture controls
----------------
  OPEN  palm + move     -> Orbit / rotate the whole scene
  POINT (1 finger) move -> Translate selected part
  PINCH on wireframe    -> Select / pick that part
  FIST  (hold 0.4s)     -> Confirm ATTACH / DETACH toggle
  Swipe LEFT            -> Go back to main menu
  Toolbar (top strip)   -> New Sphere / New Cube / New Cylinder /
                           Clone / Attach / Detach / Delete /
                           Material / Reset View

Parts live in 3-D model space (X right, Y up, Z toward viewer).
A simple orthographic projection is used with a perspective
divide factor so near parts look larger.
"""

import cv2
import numpy as np
import math
import time

# ── Colour palette (BGR) ────────────────────────────────────────────────────
C_CYAN    = (230, 220, 20)
C_CYAN_DIM= (60,  55,  6)
C_MAGENTA = (200, 40, 200)
C_GREEN   = (50,  230, 80)
C_ORANGE  = (30,  140, 255)
C_WHITE   = (255, 255, 255)
C_RED     = (40,  40,  220)
C_GOLD    = (30,  200, 255)
C_BLUE    = (200, 130, 20)
C_DARK    = (6,   10,  14)

_FONT  = cv2.FONT_HERSHEY_DUPLEX
_FONTM = cv2.FONT_HERSHEY_SIMPLEX

# Material definitions: (BGR wire colour, fill alpha, name)
MATERIALS = [
    ((230, 220,  20), 0.07, "TITANIUM"),
    (( 50, 230,  80), 0.07, "CARBON"),
    ((200,  40, 200), 0.09, "PLASMA"),
    (( 30, 140, 255), 0.08, "COPPER"),
    ((255, 180,  20), 0.08, "GOLD"),
]


# ── Low-level helpers ───────────────────────────────────────────────────────

def _glow_text(frame, text, x, y, font, scale, color, thick=1):
    dim = tuple(max(0, c // 5) for c in color)
    cv2.putText(frame, text, (x-1, y+1), font, scale, dim, thick+2)
    cv2.putText(frame, text, (x, y), font, scale, color, thick)


def _glow_line(frame, p1, p2, color, thick=1, layers=2):
    for i in range(layers, 0, -1):
        a = 0.12 * i / layers
        dim = tuple(max(0, int(c*a)) for c in color)
        cv2.line(frame, p1, p2, dim, thick+i*2)
    cv2.line(frame, p1, p2, color, thick)


def _alpha_rect(frame, x1, y1, x2, y2, color, alpha):
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    bg = np.full_like(roi, color, dtype=np.uint8)
    cv2.addWeighted(bg, alpha, roi, 1-alpha, 0, roi)
    frame[y1:y2, x1:x2] = roi


def _rot_x(pts, ang):
    c, s = math.cos(ang), math.sin(ang)
    R = np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float32)
    return (R @ pts.T).T

def _rot_y(pts, ang):
    c, s = math.cos(ang), math.sin(ang)
    R = np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float32)
    return (R @ pts.T).T

def _rot_z(pts, ang):
    c, s = math.cos(ang), math.sin(ang)
    R = np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)
    return (R @ pts.T).T


def _perspective_proj(pts3d, cx, cy, fov=500, z_off=4.0):
    """Project Nx3 world points -> Nx2 screen pixels."""
    out = []
    for x, y, z in pts3d:
        zz = z + z_off
        if zz < 0.01:
            zz = 0.01
        sx = int(cx + fov * x / zz)
        sy = int(cy - fov * y / zz)
        out.append((sx, sy))
    return out


# ── Primitive geometry factories ────────────────────────────────────────────

def _sphere_verts(r=0.5, lat=8, lon=12):
    verts, edges = [], []
    for i in range(lat+1):
        phi = math.pi * i / lat
        for j in range(lon):
            theta = 2*math.pi * j / lon
            x = r * math.sin(phi) * math.cos(theta)
            y = r * math.cos(phi)
            z = r * math.sin(phi) * math.sin(theta)
            verts.append([x, y, z])
    for i in range(lat):
        for j in range(lon):
            a = i*lon + j
            b = i*lon + (j+1)%lon
            c = (i+1)*lon + j
            edges.append((a,b))
            edges.append((a,c))
    return np.array(verts, dtype=np.float32), edges


def _cube_verts(s=0.5):
    h = s
    verts = np.array([
        [-h,-h,-h],[h,-h,-h],[h,h,-h],[-h,h,-h],
        [-h,-h, h],[h,-h, h],[h,h, h],[-h,h, h],
    ], dtype=np.float32)
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7),
    ]
    return verts, edges


def _cylinder_verts(r=0.4, h=0.8, segs=12):
    verts, edges = [], []
    top_ids, bot_ids = [], []
    for i in range(segs):
        a = 2*math.pi * i / segs
        x, z = r*math.cos(a), r*math.sin(a)
        verts.append([x,  h/2, z]); top_ids.append(len(verts)-1)
        verts.append([x, -h/2, z]); bot_ids.append(len(verts)-1)
    for i in range(segs):
        ni = (i+1) % segs
        edges.append((top_ids[i], top_ids[ni]))
        edges.append((bot_ids[i], bot_ids[ni]))
        edges.append((top_ids[i], bot_ids[i]))
    return np.array(verts, dtype=np.float32), edges


def _torus_verts(R=0.5, r=0.18, majsegs=14, minsegs=8):
    verts, edges = [], []
    for i in range(majsegs):
        phi = 2*math.pi*i/majsegs
        for j in range(minsegs):
            theta = 2*math.pi*j/minsegs
            x = (R + r*math.cos(theta)) * math.cos(phi)
            y = r * math.sin(theta)
            z = (R + r*math.cos(theta)) * math.sin(phi)
            verts.append([x,y,z])
    for i in range(majsegs):
        ni = (i+1) % majsegs
        for j in range(minsegs):
            nj = (j+1) % minsegs
            a = i*minsegs+j
            b = i*minsegs+nj
            c = ni*minsegs+j
            edges.append((a,b))
            edges.append((a,c))
    return np.array(verts, dtype=np.float32), edges


def _cone_verts(r=0.4, h=0.8, segs=12):
    verts, edges = [], []
    apex = [0, h/2, 0]
    verts.append(apex)
    base_ids = []
    for i in range(segs):
        a = 2*math.pi*i/segs
        verts.append([r*math.cos(a), -h/2, r*math.sin(a)])
        base_ids.append(len(verts)-1)
    for i in range(segs):
        ni = (i+1)%segs
        edges.append((0, base_ids[i]))
        edges.append((base_ids[i], base_ids[ni]))
    return np.array(verts, dtype=np.float32), edges


PRIMITIVE_FACTORIES = {
    "SPHERE":   _sphere_verts,
    "CUBE":     _cube_verts,
    "CYLINDER": _cylinder_verts,
    "TORUS":    _torus_verts,
    "CONE":     _cone_verts,
}


# ── Part class ───────────────────────────────────────────────────────────────

class Part:
    _id_counter = 0

    def __init__(self, kind="CUBE", pos=None, scale=1.0, material=0):
        Part._id_counter += 1
        self.pid      = Part._id_counter
        self.kind     = kind
        self.pos      = np.array(pos or [0.0, 0.0, 0.0], dtype=np.float32)
        self.rot      = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # euler xyz
        self.scale    = scale
        self.material = material % len(MATERIALS)
        self.selected = False
        self.attached_to = None   # pid of parent
        self.children    = []     # pids of attached children
        self.spawn_t     = time.time()  # for entry animation
        # build geometry
        verts, edges = PRIMITIVE_FACTORIES[kind]()
        self.base_verts = verts
        self.edges      = edges

    def world_verts(self, scene_rot_x, scene_rot_y, extra_spin=0.0):
        """Return world-space vertices after self rot, scale, translate."""
        v = self.base_verts.copy() * self.scale
        # local spin for ambient effect
        v = _rot_y(v, self.rot[1] + extra_spin)
        v = _rot_x(v, self.rot[0])
        v = _rot_z(v, self.rot[2])
        # translate to part position
        v = v + self.pos
        # apply global scene rotation
        v = _rot_x(v, scene_rot_x)
        v = _rot_y(v, scene_rot_y)
        return v

    @property
    def mat_color(self):
        return MATERIALS[self.material % len(MATERIALS)][0]

    @property
    def mat_name(self):
        return MATERIALS[self.material % len(MATERIALS)][2]


# ── Toolbar definition ───────────────────────────────────────────────────────

TOOLBAR = [
    {"id": "new_sphere",   "label": "SPHERE",  "key": "SH"},
    {"id": "new_cube",     "label": "CUBE",    "key": "CB"},
    {"id": "new_cylinder", "label": "CYL",     "key": "CY"},
    {"id": "new_torus",    "label": "TORUS",   "key": "TR"},
    {"id": "new_cone",     "label": "CONE",    "key": "CN"},
    {"id": "clone",        "label": "CLONE",   "key": "CL"},
    {"id": "attach",       "label": "ATTACH",  "key": "AT"},
    {"id": "detach",       "label": "DETACH",  "key": "DT"},
    {"id": "delete",       "label": "DELETE",  "key": "DL"},
    {"id": "material",     "label": "MAT",     "key": "MT"},
    {"id": "reset_view",   "label": "RESET",   "key": "RV"},
]


# ── Main modeller class ──────────────────────────────────────────────────────

class Modeller3D:
    """Self-contained gesture-driven holographic 3D modeller."""

    def __init__(self):
        self.t0          = time.time()
        self.parts: list[Part] = []
        self.selected_pid = None

        # Scene rotation state
        self.scene_rx    = 0.25   # radians
        self.scene_ry    = 0.40
        self._orbit_prev = None   # last cursor pos while orbiting

        # Translate state
        self._trans_prev = None

        # Toolbar hover
        self._tb_rects   = []     # list of (x1,y1,x2,y2, tool_id)
        self._tb_hover   = -1

        # Fist confirmation
        self._fist_t     = 0.0
        self._pending_action = None   # toolbar id waiting for fist confirm

        # Notification strip
        self._notif      = ""
        self._notif_timer= 0

        # Spawn a default cube so the scene isn't empty
        self._add_part("CUBE", [0.0, 0.0, 0.0])

        # Ambient slow spin
        self._ambient_spin = 0.0

    # ── Public API ─────────────────────────────────────────────────────────

    def update_and_render(self, frame, gesture_info):
        """Called every frame. Returns True if the modeller should stay open."""
        h, w, _ = frame.shape
        t = time.time() - self.t0
        self._ambient_spin = t * 0.3

        gesture  = gesture_info.get("gesture", "none")
        cursor   = gesture_info.get("cursor")
        pinch_p  = gesture_info.get("pinch_progress", 0.0)
        is_pinch = pinch_p > 0.7
        just_pinch = gesture_info.get("just_pinched", False)
        just_fist  = gesture_info.get("just_fisted", False)
        swipe    = gesture_info.get("swipe", "none")

        # Swipe left → close modeller
        if swipe == "swipe_left":
            self._notify("EXITING FORGE")
            return False

        # ── Toolbar hover + pinch select ──────────────────────────────────
        self._tb_hover = -1
        if cursor:
            for i, (x1, y1, x2, y2, tid) in enumerate(self._tb_rects):
                if x1 <= cursor[0] <= x2 and y1 <= cursor[1] <= y2:
                    self._tb_hover = i
                    break

        if just_pinch and self._tb_hover >= 0:
            action = self._tb_rects[self._tb_hover][4]
            self._dispatch_toolbar(action)

        # ── Part selection (pinch on 3-D wireframe) ───────────────────────
        elif just_pinch and cursor and self._tb_hover < 0:
            self._try_select_part(cursor, w, h)

        # ── Orbit (open palm) ──────────────────────────────────────────────
        if gesture == "open" and cursor:
            if self._orbit_prev is not None:
                dx = (cursor[0] - self._orbit_prev[0]) / w
                dy = (cursor[1] - self._orbit_prev[1]) / h
                self.scene_ry += dx * 3.0
                self.scene_rx += dy * 3.0
                self.scene_rx  = max(-1.4, min(1.4, self.scene_rx))
            self._orbit_prev = cursor
            self._trans_prev = None
        else:
            self._orbit_prev = None

        # ── Translate selected part (point gesture) ────────────────────────
        if gesture == "point" and cursor and self.selected_pid is not None:
            if self._trans_prev is not None:
                dx = (cursor[0] - self._trans_prev[0]) / w
                dy = (cursor[1] - self._trans_prev[1]) / h
                part = self._get_part(self.selected_pid)
                if part:
                    part.pos[0] += dx * 3.0
                    part.pos[1] -= dy * 3.0
                    # also move children
                    for cpid in part.children:
                        cp = self._get_part(cpid)
                        if cp:
                            cp.pos[0] += dx * 3.0
                            cp.pos[1] -= dy * 3.0
            self._trans_prev = cursor
        else:
            self._trans_prev = None

        # ── Fist hold to confirm pending action ────────────────────────────
        if gesture == "fist" and self._pending_action:
            self._fist_t += 1/30.0
            if self._fist_t >= 0.4:
                self._execute_action(self._pending_action)
                self._pending_action = None
                self._fist_t = 0.0
        else:
            self._fist_t = 0.0

        # ── Notification timer ─────────────────────────────────────────────
        if self._notif_timer > 0:
            self._notif_timer -= 1
        else:
            self._notif = ""

        # ── Render ────────────────────────────────────────────────────────
        self._draw_background(frame, w, h, t)
        self._draw_grid(frame, w, h)
        self._draw_parts(frame, w, h, t)
        self._draw_toolbar(frame, w, h, t)
        self._draw_hud(frame, w, h, gesture_info, t)
        if cursor:
            self._draw_cursor(frame, cursor, pinch_p, t)

        return True

    # ── Toolbar dispatch ──────────────────────────────────────────────────

    def _dispatch_toolbar(self, action):
        offset = np.array([
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-0.3, 0.3),
            np.random.uniform(-0.3, 0.3)
        ], dtype=np.float32)

        if action == "new_sphere":
            self._add_part("SPHERE", offset); self._notify("SPHERE SPAWNED")
        elif action == "new_cube":
            self._add_part("CUBE", offset);   self._notify("CUBE SPAWNED")
        elif action == "new_cylinder":
            self._add_part("CYLINDER", offset); self._notify("CYLINDER SPAWNED")
        elif action == "new_torus":
            self._add_part("TORUS", offset);  self._notify("TORUS SPAWNED")
        elif action == "new_cone":
            self._add_part("CONE", offset);   self._notify("CONE SPAWNED")
        elif action == "clone":
            self._clone_selected()
        elif action == "attach":
            self._pending_action = "attach"
            self._notify("MAKE FIST TO CONFIRM ATTACH")
        elif action == "detach":
            self._pending_action = "detach"
            self._notify("MAKE FIST TO CONFIRM DETACH")
        elif action == "delete":
            self._pending_action = "delete"
            self._notify("MAKE FIST TO CONFIRM DELETE")
        elif action == "material":
            self._cycle_material()
        elif action == "reset_view":
            self.scene_rx = 0.25
            self.scene_ry = 0.40
            self._notify("VIEW RESET")

    def _execute_action(self, action):
        if action == "attach":
            self._do_attach()
        elif action == "detach":
            self._do_detach()
        elif action == "delete":
            self._do_delete()

    # ── Part operations ───────────────────────────────────────────────────

    def _add_part(self, kind, pos, material=None):
        mat = material if material is not None else len(self.parts) % len(MATERIALS)
        p = Part(kind, list(pos), scale=0.55, material=mat)
        self.parts.append(p)
        self.selected_pid = p.pid
        return p

    def _get_part(self, pid):
        for p in self.parts:
            if p.pid == pid:
                return p
        return None

    def _get_selected(self):
        return self._get_part(self.selected_pid)

    def _try_select_part(self, cursor, w, h):
        """Pick the part whose projected centroid is closest to cursor."""
        cx_s, cy_s = w // 2, h // 2
        best_dist  = 80 * 80  # pixel² threshold
        best_pid   = None
        for p in self.parts:
            verts = p.world_verts(self.scene_rx, self.scene_ry)
            centroid = verts.mean(axis=0)
            proj = _perspective_proj([centroid], cx_s, cy_s)[0]
            dx = cursor[0] - proj[0]
            dy = cursor[1] - proj[1]
            d2 = dx*dx + dy*dy
            if d2 < best_dist:
                best_dist = d2
                best_pid  = p.pid
        # clear old selection
        for p in self.parts:
            p.selected = False
        if best_pid is not None:
            self.selected_pid = best_pid
            self._get_part(best_pid).selected = True
            self._notify(f"SELECTED: PART-{best_pid:02d}")
        else:
            self.selected_pid = None

    def _clone_selected(self):
        src = self._get_selected()
        if src is None:
            self._notify("NO PART SELECTED"); return
        new_pos = src.pos + np.array([0.2, 0.15, 0.0], dtype=np.float32)
        p = self._add_part(src.kind, new_pos, src.material)
        p.rot[:] = src.rot
        self._notify(f"CLONED -> PART-{p.pid:02d}")

    def _do_attach(self):
        """Attach selected part to the part closest to it (parent)."""
        sel = self._get_selected()
        if sel is None or len(self.parts) < 2:
            self._notify("NEED 2 PARTS TO ATTACH"); return
        # find closest other part
        best_d = 1e9
        best_p = None
        for p in self.parts:
            if p.pid == sel.pid:
                continue
            d = np.linalg.norm(sel.pos - p.pos)
            if d < best_d:
                best_d = d
                best_p = p
        if best_p:
            sel.attached_to = best_p.pid
            if sel.pid not in best_p.children:
                best_p.children.append(sel.pid)
            self._notify(f"PART-{sel.pid:02d} ATTACHED TO PART-{best_p.pid:02d}")

    def _do_detach(self):
        sel = self._get_selected()
        if sel is None or sel.attached_to is None:
            self._notify("PART NOT ATTACHED"); return
        parent = self._get_part(sel.attached_to)
        if parent and sel.pid in parent.children:
            parent.children.remove(sel.pid)
        sel.attached_to = None
        self._notify(f"PART-{sel.pid:02d} DETACHED")

    def _do_delete(self):
        sel = self._get_selected()
        if sel is None:
            self._notify("NO PART SELECTED"); return
        # remove from any parent
        if sel.attached_to:
            parent = self._get_part(sel.attached_to)
            if parent and sel.pid in parent.children:
                parent.children.remove(sel.pid)
        # detach children
        for cpid in sel.children:
            cp = self._get_part(cpid)
            if cp:
                cp.attached_to = None
        self.parts = [p for p in self.parts if p.pid != sel.pid]
        self.selected_pid = self.parts[-1].pid if self.parts else None
        self._notify(f"PART-{sel.pid:02d} DELETED")

    def _cycle_material(self):
        sel = self._get_selected()
        if sel is None:
            self._notify("NO PART SELECTED"); return
        sel.material = (sel.material + 1) % len(MATERIALS)
        self._notify(f"MATERIAL: {sel.mat_name}")

    def _notify(self, msg):
        self._notif = msg
        self._notif_timer = 90  # ~3 s at 30fps

    # ── Rendering ─────────────────────────────────────────────────────────

    def _draw_background(self, frame, w, h, t):
        # Subtle global dark tint
        dark = np.full_like(frame, (6, 10, 14), dtype=np.uint8)
        cv2.addWeighted(dark, 0.28, frame, 0.72, 0, frame)

        # Scan line
        sweep_y = int((t * 80) % h)
        band = 22
        for dy in range(-band, band):
            sy = sweep_y + dy
            if 0 <= sy < h:
                a = 1.0 - abs(dy)/band
                add = tuple(int(c*a*0.10) for c in C_CYAN)
                frame[sy] = np.clip(frame[sy].astype(np.int32) + add, 0, 255)

        # Title bar
        _alpha_rect(frame, 0, 0, w, 38, C_DARK, 0.80)
        cv2.line(frame, (0, 38), (w, 38),
                 tuple(int(c*0.5) for c in C_CYAN), 1)
        pulse = 0.6 + 0.4 * math.sin(t * 2.0)
        tc    = tuple(int(c*pulse) for c in C_CYAN)
        _glow_text(frame, "STARK FORGE  //  HOLOGRAPHIC MODELLER",
                   14, 24, _FONTM, 0.50, tc, 1)

    def _draw_grid(self, frame, w, h):
        """Perspective floor grid."""
        cx, cy = w//2, h//2 + 60
        grid_c = tuple(int(c*0.18) for c in C_CYAN)
        ext = 5
        for gx in range(-ext, ext+1):
            step = 0.4
            p1 = _perspective_proj(
                [_rot_y(_rot_x(np.array([[gx*step, -0.7, -ext*step]]), self.scene_rx),
                        self.scene_ry)[0]], cx, cy)[0]
            p2 = _perspective_proj(
                [_rot_y(_rot_x(np.array([[gx*step, -0.7,  ext*step]]), self.scene_rx),
                        self.scene_ry)[0]], cx, cy)[0]
            cv2.line(frame, p1, p2, grid_c, 1)
        for gz in range(-ext, ext+1):
            step = 0.4
            p1 = _perspective_proj(
                [_rot_y(_rot_x(np.array([[-ext*step, -0.7, gz*step]]), self.scene_rx),
                        self.scene_ry)[0]], cx, cy)[0]
            p2 = _perspective_proj(
                [_rot_y(_rot_x(np.array([[ ext*step, -0.7, gz*step]]), self.scene_rx),
                        self.scene_ry)[0]], cx, cy)[0]
            cv2.line(frame, p1, p2, grid_c, 1)

    def _draw_parts(self, frame, w, h, t):
        cx, cy = w//2, h//2 + 30

        # Sort back-to-front by z depth (painter's algorithm)
        def z_key(p):
            v = p.world_verts(self.scene_rx, self.scene_ry)
            return -v[:, 2].mean()

        for part in sorted(self.parts, key=z_key):
            age = t - (part.spawn_t - self.t0)
            entry_scale = min(1.0, age * 4.0)  # 0.25s entry grow

            # ambient slow spin for unselected parts
            extra_spin = self._ambient_spin * 0.25 if not part.selected else 0.0

            verts3d = part.world_verts(self.scene_rx, self.scene_ry,
                                       extra_spin=extra_spin)
            # scale entry
            if entry_scale < 1.0:
                verts3d = verts3d * entry_scale

            proj = _perspective_proj(verts3d, cx, cy)

            mat_color = part.mat_color
            is_sel    = (part.pid == self.selected_pid)

            # Fill (semi-transparent polygon faces — draw first)
            fill_a = MATERIALS[part.material][1]
            if is_sel:
                fill_a *= 2.5
            # draw each edge
            for (a, b) in part.edges:
                if a >= len(proj) or b >= len(proj):
                    continue
                p1, p2 = proj[a], proj[b]
                # clip to frame
                if not (0 <= p1[0] < w and 0 <= p1[1] < h):
                    continue
                if not (0 <= p2[0] < w and 0 <= p2[1] < h):
                    continue
                thick = 2 if is_sel else 1
                bright = 1.0 if is_sel else 0.65
                c = tuple(int(ch * bright) for ch in mat_color)
                _glow_line(frame, p1, p2, c, thick, layers=1 if not is_sel else 2)

            # centroid label
            centroid3 = verts3d.mean(axis=0)
            cent_px   = _perspective_proj([centroid3], cx, cy)[0]
            label_c   = tuple(int(c * 0.8) for c in mat_color)
            pid_str   = f"P{part.pid:02d}"
            if is_sel:
                pid_str = f"[P{part.pid:02d}]"
            _glow_text(frame, pid_str,
                       cent_px[0] - 14, cent_px[1] - 6,
                       _FONTM, 0.36, label_c, 1)

            # Attachment lines (to parent)
            if part.attached_to:
                parent = self._get_part(part.attached_to)
                if parent:
                    pv = parent.world_verts(self.scene_rx, self.scene_ry)
                    pcent = _perspective_proj([pv.mean(axis=0)], cx, cy)[0]
                    link_c = tuple(int(c*0.55) for c in C_MAGENTA)
                    cv2.line(frame, cent_px, pcent, link_c, 1)
                    # dot at midpoint
                    mid = ((cent_px[0]+pcent[0])//2, (cent_px[1]+pcent[1])//2)
                    cv2.circle(frame, mid, 3, link_c, -1)

            # Selected: spinning outer ring
            if is_sel:
                ring_r = int(40 * entry_scale)
                spin_a = t * 180
                for seg in range(6):
                    a_s = spin_a + seg*60
                    a_e = a_s + 28
                    cv2.ellipse(frame, cent_px, (ring_r, ring_r),
                                0, a_s, a_e, C_CYAN, 2)

    def _draw_toolbar(self, frame, w, h, t):
        """Horizontal hex-button toolbar at the bottom."""
        n   = len(TOOLBAR)
        btn_w = max(54, (w - 20) // n)
        btn_h = 44
        y1    = h - btn_h - 8
        y2    = h - 8

        _alpha_rect(frame, 0, y1 - 6, w, h, C_DARK, 0.82)
        cv2.line(frame, (0, y1-6), (w, y1-6),
                 tuple(int(c*0.4) for c in C_CYAN), 1)

        self._tb_rects = []
        for i, tool in enumerate(TOOLBAR):
            x1 = 10 + i * btn_w
            x2 = x1 + btn_w - 4
            # clamp
            if x2 > w - 4:
                break

            is_hover = (self._tb_hover == i)
            pulse    = 0.55 + 0.45 * math.sin(t * 3.0 + i * 0.7)

            # background
            fill_a = 0.22 if is_hover else 0.08
            base_c = C_CYAN if not is_hover else C_GOLD
            _alpha_rect(frame, x1, y1, x2, y2, base_c, fill_a)

            # border
            border_c = tuple(int(c * (pulse if is_hover else 0.4)) for c in base_c)
            cv2.rectangle(frame, (x1, y1), (x2, y2), border_c, 1)

            # label
            label = tool["label"]
            (tw, th), _ = cv2.getTextSize(label, _FONTM, 0.36, 1)
            tx = x1 + (btn_w - tw) // 2
            ty = y1 + (btn_h + th) // 2 - 2
            tc = tuple(int(c * (1.0 if is_hover else 0.65)) for c in base_c)
            _glow_text(frame, label, tx, ty, _FONTM, 0.36, tc, 1)

            # fist confirm progress bar
            if self._pending_action == tool["id"] and self._fist_t > 0:
                prog = min(1.0, self._fist_t / 0.4)
                bw   = int((x2 - x1) * prog)
                cv2.rectangle(frame, (x1, y2-3), (x1+bw, y2), C_MAGENTA, -1)

            self._tb_rects.append((x1, y1, x2, y2, tool["id"]))

    def _draw_hud(self, frame, w, h, gesture_info, t):
        """Right-side info panel."""
        sel = self._get_selected()
        panel_x = w - 180
        panel_y = 48
        panel_h = 170

        _alpha_rect(frame, panel_x - 8, panel_y - 8,
                    w - 4, panel_y + panel_h, C_DARK, 0.75)
        cv2.rectangle(frame, (panel_x-8, panel_y-8),
                      (w-4, panel_y+panel_h),
                      tuple(int(c*0.3) for c in C_CYAN), 1)

        lines = [
            f"PARTS : {len(self.parts)}",
            f"SEL   : {'P'+str(sel.pid).zfill(2) if sel else 'NONE'}",
            f"KIND  : {sel.kind if sel else '---'}",
            f"MAT   : {sel.mat_name if sel else '---'}",
            f"ATTACH: {'P'+str(sel.attached_to).zfill(2) if sel and sel.attached_to else 'FREE'}",
            f"CHILD : {len(sel.children) if sel else 0}",
            f"RX    : {math.degrees(self.scene_rx):.0f}d",
            f"RY    : {math.degrees(self.scene_ry):.0f}d",
        ]
        for li, line in enumerate(lines):
            lc = tuple(int(c * 0.55) for c in C_CYAN)
            _glow_text(frame, line, panel_x, panel_y + li*20,
                       _FONTM, 0.37, lc, 1)

        # Gesture indicator
        gesture = gesture_info.get("gesture", "none").upper()
        gc = {"OPEN": C_GREEN, "POINT": C_CYAN,
              "PINCH": C_MAGENTA, "FIST": C_RED}.get(gesture, C_CYAN_DIM)
        _glow_text(frame, gesture, 16, 60, _FONTM, 0.55, gc, 1)

        # Fist hold bar
        if self._fist_t > 0:
            prog = min(1.0, self._fist_t / 0.4)
            bw   = int((w // 3) * prog)
            by   = 70
            cv2.rectangle(frame, (16, by), (16+bw, by+6), C_MAGENTA, -1)
            _glow_text(frame, self._pending_action.upper() if self._pending_action else "",
                       20, by+22, _FONTM, 0.38, C_MAGENTA, 1)

        # Notification
        if self._notif:
            pulse = 0.7 + 0.3 * math.sin(t * 6)
            nc    = tuple(int(c*pulse) for c in C_GOLD)
            (nw, _), _ = cv2.getTextSize(self._notif, _FONTM, 0.50, 1)
            nx = (w - nw) // 2
            _glow_text(frame, self._notif, nx, h - 68, _FONTM, 0.50, nc, 1)

        # Controls legend (bottom-left)
        legend = [
            "OPEN   = ORBIT",
            "POINT  = MOVE PART",
            "PINCH  = SELECT",
            "FIST   = CONFIRM",
            "SWIPE< = EXIT",
        ]
        for li, line in enumerate(legend):
            lc = tuple(int(c * 0.25) for c in C_CYAN)
            cv2.putText(frame, line, (14, 90+li*16), _FONTM, 0.30, lc, 1)

    def _draw_cursor(self, frame, cursor, pinch_p, t):
        x, y  = int(cursor[0]), int(cursor[1])
        spin  = t * 140
        r_out = 20 + int(3 * math.sin(t*4))
        # inner dot
        cv2.circle(frame, (x, y), 3, C_WHITE, -1)
        # spinning hex
        pts = []
        for i in range(6):
            ang = math.radians(60*i + spin % 360)
            pts.append((int(x + 10*math.cos(ang)),
                        int(y + 10*math.sin(ang))))
        for i in range(6):
            cv2.line(frame, pts[i], pts[(i+1)%6], C_CYAN, 1)
        # outer segments
        for seg in range(8):
            a_s = spin*0.6 + seg*45
            a_e = a_s + 18
            cv2.ellipse(frame, (x,y), (r_out, r_out), 0, a_s, a_e, C_CYAN, 2)
        # pinch arc
        if pinch_p > 0.05:
            sweep = int(360*pinch_p)
            cv2.ellipse(frame, (x,y), (r_out+5, r_out+5),
                        0, -90, -90+sweep, C_MAGENTA, 2)
