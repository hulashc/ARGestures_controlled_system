"""Central configuration for the AR Gesture Control System."""

# Camera
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 30

# Colors (BGR format for OpenCV)
COLOR_CYAN = (0, 255, 255)
COLOR_MAGENTA = (255, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_RED = (0, 0, 255)
COLOR_DARK = (30, 30, 30)

# MediaPipe Hands
MAX_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5

# Gesture thresholds
PINCH_THRESHOLD = 0.06          # Normalised distance for pinch detection
SWIPE_THRESHOLD = 0.08          # Normalised x-delta for swipe detection
LANDMARK_STALE_FRAMES = 4       # Frames before cached landmarks are cleared

# Cursor smoothing
CURSOR_SMOOTHER_ALPHA = 0.35    # Higher = more responsive, lower = smoother

# UI tile layout
TILE_COLS = 2
TILE_ROWS = 3
TILE_W = 160
TILE_H = 100
TILE_GAP = 10
TILE_HEADER_H = 40
