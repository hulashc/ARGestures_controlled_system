"""Menu state — FORGE only.

Single entry: launches the 3-D holographic modeller directly.
No navigation, no sub-menus, no other apps.
"""


class MenuItem:
    def __init__(self, name, desc='', icon='', app_id=None):
        self.name      = name
        self.children  = []
        self.desc      = desc
        self.icon      = icon
        self.app_id    = app_id
        self.activated = False

    @property
    def is_submenu(self):
        return len(self.children) > 0

    @property
    def is_app(self):
        return self.app_id is not None


class MenuState:
    """Minimal state machine — always FORGE, boots straight to app."""

    def __init__(self):
        self.state            = 'APP_LAUNCH'
        self.activation_timer = 0
        self.transition_timer = 0
        self.transition_type  = 'none'

    # ── The single item ───────────────────────────────────────────────────────

    @property
    def current_items(self):
        return [MenuItem('FORGE',
                         desc='Holographic 3-D part modeller',
                         icon='3D',
                         app_id='modeller_3d')]

    @property
    def selected_item(self):
        return self.current_items[0]

    @property
    def selected_index(self):
        return 0

    @property
    def depth(self):
        return 0

    @property
    def breadcrumb(self):
        return 'STARK OS  //  FORGE'

    @property
    def is_root(self):
        return True

    # ── No-op navigation ──────────────────────────────────────────────────────

    def next_item(self):  pass
    def prev_item(self):  pass
    def go_back(self):    pass

    def activate(self):
        """Always returns 'modeller_3d'."""
        item = self.selected_item
        item.activated        = True
        self.activation_timer = 20
        self.state            = 'APP_LAUNCH'
        return item.app_id

    def update(self, hand_present=True):
        if self.activation_timer > 0:
            self.activation_timer -= 1
            if self.activation_timer == 0:
                self.state = 'APP_LAUNCH'
        if self.transition_timer > 0:
            self.transition_timer -= 1
            if self.transition_timer == 0:
                self.transition_type = 'none'

    def render_info(self):
        return {
            'items':            self.current_items,
            'selected_index':   0,
            'state':            self.state,
            'activation_timer': self.activation_timer,
            'depth':            self.depth,
            'breadcrumb':       self.breadcrumb,
            'is_root':          True,
            'transition_type':  self.transition_type,
            'transition_timer': self.transition_timer,
        }
