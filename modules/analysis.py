# modules/analysis.py
import cv2
import numpy as np



def filter_noise_defects(defect_mask, min_size, min_circularity,
                         morph_open_kernel_size, morph_open_iterations,
                         morph_close_kernel_size, morph_close_iterations):
    """
    Clean defect mask with morphology + contour filtering.
    Inputs:
      - defect_mask: uint8 (0/255)
    Returns: filtered_mask (uint8 0/255)
    """
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open_kernel_size, morph_open_kernel_size))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close_kernel_size, morph_close_kernel_size))

    opened = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, kernel_open, iterations=morph_open_iterations)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=morph_close_iterations)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)# 
    filtered_mask = np.zeros_like(defect_mask)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_size:
            continue
        if area < min_size * 3:
            perim = cv2.arcLength(cnt, True)
            if perim == 0:
                continue
            circ = 4 * np.pi * area / (perim * perim)
            if circ < min_circularity:
                continue
        cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)
    return filtered_mask

def analyze_defect(delta_e_map, thresholds):
    """
    thresholds: dict with keys 'mean_diff','max_diff','area_percent','delta_e_pixel_threshold'
    Returns: is_defect(bool), mean_diff, max_diff, area_percent
    """
    vals = delta_e_map.flatten()
    if vals.size == 0:
        return False, 0.0, 0.0, 0.0
    mean_diff = float(vals.mean())
    max_diff = float(vals.max())
    area_percent = float((vals > thresholds['delta_e_pixel_threshold']).sum() / vals.size * 100.0)

    is_defect = (mean_diff > thresholds['mean_diff'] or
                 max_diff > thresholds['max_diff'] or
                 area_percent > thresholds['area_percent'])
    return is_defect, mean_diff, max_diff, area_percent
