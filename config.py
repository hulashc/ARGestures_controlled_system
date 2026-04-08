"""Central configuration for the AR Gesture Control System."""

# Camera
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 30

# Colors (BGR format for OpenCV)
COLOR_CYAN    = (0, 255, 255)
COLOR_MAGENTA = (255, 0, 255)
COLOR_GREEN   = (0, 255, 0)
COLOR_WHITE   = (255, 255, 255)
COLOR_RED     = (0, 0, 255)
COLOR_DARK    = (30, 30, 30)

# MediaPipe Hands — TWO hands
MAX_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.65
MIN_TRACKING_CONFIDENCE  = 0.5

# Gesture thresholds
PINCH_THRESHOLD        = 0.06   # normalised distance for pinch
SWIPE_THRESHOLD        = 0.08   # normalised x-delta for swipe
LANDMARK_STALE_FRAMES  = 4     # frames before cached landmarks cleared

# Cursor smoothing
CURSOR_SMOOTHER_ALPHA = 0.35

# Two-hand interaction
TWO_HAND_SCALE_SENS   = 3.0    # scale sensitivity multiplier
TWO_HAND_ROTATE_SENS  = 4.0    # Z-rotation sensitivity (radians per normalised delta)
