"""Menu state machine — supports nested submenus.

Structure:
    MAIN
    ├── FILES → [Documents, Downloads, Images, Videos]
    ├── APPS  → [Browser, Terminal, Settings, Calculator]
    ├── SYSTEM → [CPU, Memory, Storage, Battery]
    ├── NETWORK → [WiFi, Bluetooth, Ethernet, Proxy]
    ├── SCAN → [Scan All, Quick Scan, Deep Scan]
    └── TERMINAL → (direct action)

Controls:
    Point / Swipe Right → Move down
    Swipe Left → Go back to parent
    Pinch → Enter submenu or activate action
"""


class MenuItem:
    def __init__(self, name, children=None, desc=""):
        self.name = name
        self.children = children or []
        self.desc = desc
        self.activated = False

    @property
    def is_submenu(self):
        return len(self.children) > 0


class MenuState:
    def __init__(self):
        self.root = self._build_menu()
        self.stack = [self.root]  # navigation stack
        self.selected_index = 0
        self.state = "NAVIGATING"
        self.activation_timer = 0
        self.transition_timer = 0  # animation for submenu enter/exit
        self.transition_type = "none"  # "enter", "exit", "none"

    def _build_menu(self):
        return MenuItem("MAIN", [
            MenuItem("FILES", [
                MenuItem("Documents", desc="View and manage documents"),
                MenuItem("Downloads", desc="Recent downloads folder"),
                MenuItem("Images", desc="Photo gallery and editor"),
                MenuItem("Videos", desc="Video library and player"),
            ], desc="File management system"),
            MenuItem("APPS", [
                MenuItem("Browser", desc="Open web browser"),
                MenuItem("Terminal", desc="Command line interface"),
                MenuItem("Settings", desc="System preferences"),
                MenuItem("Calculator", desc="Quick calculator tool"),
            ], desc="Application launcher"),
            MenuItem("SYSTEM", [
                MenuItem("CPU", desc="Processor usage monitor"),
                MenuItem("Memory", desc="RAM usage statistics"),
                MenuItem("Storage", desc="Disk space analyzer"),
                MenuItem("Battery", desc="Power management info"),
            ], desc="System diagnostics"),
            MenuItem("NETWORK", [
                MenuItem("WiFi", desc="Wireless network settings"),
                MenuItem("Bluetooth", desc="Bluetooth device manager"),
                MenuItem("Ethernet", desc="Wired connection status"),
                MenuItem("Proxy", desc="Proxy configuration"),
            ], desc="Network connections"),
            MenuItem("SCAN", [
                MenuItem("Scan All", desc="Full system scan"),
                MenuItem("Quick Scan", desc="Fast security check"),
                MenuItem("Deep Scan", desc="Thorough analysis mode"),
            ], desc="Security scanner"),
            MenuItem("TERMINAL", desc="Open command terminal directly"),
        ])

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
        """Pinch: enter submenu or trigger leaf action."""
        item = self.selected_item
        if item is None:
            return

        if item.is_submenu:
            # Enter submenu
            self.stack.append(item)
            self.selected_index = 0
            self.transition_type = "enter"
            self.transition_timer = 15
            print(f"  >> ENTER: {item.name}")
        else:
            # Leaf action
            item.activated = True
            self.activation_timer = 30
            self.state = "SELECTED"
            print(f"  >> ACTIVATE: {item.name}")

    def go_back(self):
        """Swipe left: go back to parent menu."""
        if self.is_root:
            return
        self.stack.pop()
        self.selected_index = 0
        self.transition_type = "exit"
        self.transition_timer = 15
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
            "items": self.current_items,
            "selected_index": self.selected_index,
            "state": self.state,
            "activation_timer": self.activation_timer,
            "depth": self.depth,
            "breadcrumb": self.breadcrumb,
            "is_root": self.is_root,
            "transition_type": self.transition_type,
            "transition_timer": self.transition_timer,
        }
