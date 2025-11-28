# ============================
# Camera and Debug Settings
# ============================
USE_CAMERA = False
DEBUG = False

# ============================
# ORB Feature Matching Parameters
# ============================
ORB_MAX_FEATURES = 5000
ORB_KEEP_PERCENT = 0.20

# ============================
# Delta-E Thresholds
# ============================
DELTA_E_PIXEL_THRESHOLD = 10
MEAN_DIFF = 6.0
MAX_DIFF = 25.0
AREA_PERCENT = 3.0

# ============================
# Noise Filtering Parameters
# ============================
MIN_DEFECT_SIZE = 50
MIN_CIRCULARITY = 0.15

# ============================
# Morphological Operations
# ============================
MORPH_OPEN_KERNEL_SIZE = 5
MORPH_OPEN_ITERATIONS = 1
MORPH_CLOSE_KERNEL_SIZE = 5
MORPH_CLOSE_ITERATIONS = 1

# ============================
# Helper function to get all config as dictionary
# ============================
def get_config():
    """Returns all configuration parameters as a dictionary."""
    return {
        "ORB_MAX_FEATURES": ORB_MAX_FEATURES,
        "ORB_KEEP_PERCENT": ORB_KEEP_PERCENT,
        "DELTA_E_PIXEL_THRESHOLD": DELTA_E_PIXEL_THRESHOLD,
        "MEAN_DIFF": MEAN_DIFF,
        "MAX_DIFF": MAX_DIFF,
        "AREA_PERCENT": AREA_PERCENT,
        "MIN_DEFECT_SIZE": MIN_DEFECT_SIZE,
        "MIN_CIRCULARITY": MIN_CIRCULARITY,
        "MORPH_OPEN_KERNEL_SIZE": MORPH_OPEN_KERNEL_SIZE,
        "MORPH_OPEN_ITERATIONS": MORPH_OPEN_ITERATIONS,
        "MORPH_CLOSE_KERNEL_SIZE": MORPH_CLOSE_KERNEL_SIZE,
        "MORPH_CLOSE_ITERATIONS": MORPH_CLOSE_ITERATIONS
    }
