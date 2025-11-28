# modules/heatmap.py
import cv2
import numpy as np
from modules.io_utils import save_image

def generate_heatmap(delta_e, base_image, out_path):
    """
    Save an overlay heatmap (jet) blended with base_image to out_path.
    """
    norm = cv2.normalize(delta_e, None, 0, 255, cv2.NORM_MINMAX).astype('uint8') # type: ignore
    heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(base_image, 0.5, heatmap, 0.8, 0)
    save_image(out_path, overlay)
    return out_path

def generate_heatmap_in_memory(delta_e_normalized, golden):
    """
    Generate heatmap overlay in-memory without saving to disk.
    
    Args:
        delta_e_normalized: normalized delta-E map
        golden: golden sample image
        
    Returns:
        numpy array of heatmap overlay
    """
    import cv2
    
    heatmap_colored = cv2.applyColorMap(delta_e_normalized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(golden, 0.6, heatmap_colored, 0.4, 0)
    return overlay
