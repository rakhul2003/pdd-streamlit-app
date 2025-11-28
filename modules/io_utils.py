# modules/io_utils.py
import cv2
import os
from datetime import datetime

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    return img

def save_image(path, image):
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, image)

def create_session_output(base_output="output"):
    ts = datetime.now().strftime("%d-%m-%Y(%H-%M-%S)")
    session_dir = os.path.join(base_output, f"session_{ts}")
    ensure_dir(session_dir)
    return session_dir
