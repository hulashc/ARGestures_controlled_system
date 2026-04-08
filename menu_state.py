"""Menu state machine — supports nested submenus.

Structure:
    MAIN
    ├── FILES   → [Documents, Downloads, Images, Videos]
    ├── APPS    → [Browser, Terminal, Settings, Calculator]
    ├── SYSTEM  → [CPU, Memory, Storage, Battery]
    ├── NETWORK → [WiFi, Bluetooth, Ethernet, Proxy]
    ├── SCAN    → [Scan All, Quick Scan, Deep Scan]
    ├── FORGE   → 3-D Holographic Modeller
    └── TERMINAL → (direct action)

Controls:
    Swipe Right → Next item
    Swipe Left  → Go back
    Hover + Pinch → Enter submenu or activate
"""


class MenuItem:
    def __init__(self, name, children=None, desc="", icon="", app_id=None):
        self.name     = name
        self.children = children or []
        self.desc     = desc
        self.icon     = icon
        self.app_id   = app_id   # non-None -> launches an app module
        self.activated = False

    @property
    def is_submenu(self):
        return len(self.children) > 0

    @property
    def is_app(self):
        return self.app_id is not None


class MenuState:
    def __init__(self):
        self.root  = self._build_menu()
        self.stack = [self.root]
        self.selected_index  = 0
        self.state           = "NAVIGATING"
        self.activation_timer = 0
        self.transition_timer = 0
        self.transition_type  = "none"   # "enter" | "exit" | "none"
        self._TRANS_FRAMES    = 20

    def _build_menu(self):
        return MenuItem("MAIN", [
            MenuItem("FILES", [
                MenuItem("Documents", desc="Manage and view documents",   icon="DOC"),
                MenuItem("Downloads", desc="Recent downloads folder",      icon="DL"),
                MenuItem("Images",    desc="Photo gallery and editor",     icon="IMG"),
                MenuItem("Videos",    desc="Video library and player",     icon="VID"),
            ], desc="File management system", icon="FS"),
            MenuItem("APPS", [
                MenuItem("Browser",    desc="Open web browser",            icon="WEB"),
                MenuItem("Terminal",   desc="Command line interface",      icon="CLI"),
                MenuItem("Settings",   desc="System preferences",          icon="CFG"),
                MenuItem("Calculator", desc="Quick calculator tool",       icon="CAL"),
            ], desc="Application launcher", icon="APP"),
            MenuItem("SYSTEM", [
                MenuItem("CPU",     desc="Processor usage monitor",       icon="CPU"),
                MenuItem("Memory",  desc="RAM usage statistics",          icon="RAM"),
                MenuItem("Storage", desc="Disk space analyzer",           icon="HDD"),
                MenuItem("Battery", desc="Power management info",         icon="PWR"),
            ], desc="System diagnostics", icon="SYS"),
            MenuItem("NETWORK", [
                MenuItem("WiFi",      desc="Wireless network settings",   icon="WFI"),
                MenuItem("Bluetooth", desc="Bluetooth device manager",    icon="BT"),
                MenuItem("Ethernet",  desc="Wired connection status",     icon="ETH"),
                MenuItem("Proxy",     desc="Proxy configuration",         icon="PRX"),
            ], desc="Network connections", icon="NET"),
            MenuItem("SCAN", [
                MenuItem("Scan All",   desc="Full system scan",           icon="ALL"),
                MenuItem("Quick Scan", desc="Fast security check",        icon="QCK"),
                MenuItem("Deep Scan",  desc="Thorough analysis mode",     icon="DPS"),
            ], desc="Security scanner", icon="SEC"),
            MenuItem("FORGE",
                     desc="Holographic 3-D part modeller",
                     icon="3D",
                     app_id="modeller_3d"),
            MenuItem("TERMINAL", desc="Open command terminal directly",   icon=">\_"),
        ])

    # ── Navigation ───────────────────────────────────────────────────────────

    @property
    def current_menu(self):
        return self.stack[-1]

    @property
    def current_items(self):
        return self.current_menu.children

    @property
    def selected_item(self):
        if self.current_items:
            return self.current_items[self.selected_index]
        return None

    @property
    def depth(self):
        return len(self.stack) - 1

    @property
    def breadcrumb(self):
        return " / ".join(m.name for m in self.stack)

    @property
    def is_root(self):
        return len(self.stack) <= 1

    def next_item(self):
        if not self.current_items:
            return
        self.selected_index = (self.selected_index + 1) % len(self.current_items)
        self._clear_activation()

    def prev_item(self):
        if not self.current_items:
            return
        self.selected_index = (self.selected_index - 1) % len(self.current_items)
        self._clear_activation()

    def activate(self):
        item = self.selected_item
        if item is None:
            return None
        if item.is_app:
            item.activated = True
            self.activation_timer = 20
            self.state = "APP_LAUNCH"
            print(f"  >> LAUNCH APP: {item.app_id}")
            return item.app_id
        if item.is_submenu:
            self.stack.append(item)
            self.selected_index  = 0
            self.transition_type = "enter"
            self.transition_timer = self._TRANS_FRAMES
            print(f"  >> ENTER: {item.name}")
        else:
            item.activated = True
            self.activation_timer = 30
            self.state = "SELECTED"
            print(f"  >> ACTIVATE: {item.name}")
        return None

    def go_back(self):
        if self.is_root:
            return
        self.stack.pop()
        self.selected_index  = 0
        self.transition_type = "exit"
        self.transition_timer = self._TRANS_FRAMES
        print(f"  >> BACK to {self.current_menu.name}")

    def update(self, hand_present=True):
        if hand_present and self.state == "IDLE":
            self.state = "NAVIGATING"
        if self.activation_timer > 0:
            self.activation_timer -= 1
            if self.activation_timer == 0:
                self._clear_activation()
        if self.transition_timer > 0:
            self.transition_timer -= 1
            if self.transition_timer == 0:
                self.transition_type = "none"

    def _clear_activation(self):
        for item in self.current_items:
            item.activated = False
        self.state = "NAVIGATING"

    def render_info(self):
        return {
            "items":            self.current_items,
            "selected_index":   self.selected_index,
            "state":            self.state,
            "activation_timer": self.activation_timer,
            "depth":            self.depth,
            "breadcrumb":       self.breadcrumb,
            "is_root":          self.is_root,
            "transition_type":  self.transition_type,
            "transition_timer": self.transition_timer,
        }
