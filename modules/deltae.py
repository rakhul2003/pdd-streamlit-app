# modules/deltae.py
import cv2
import numpy as np



def compute_delta_e(golden, test):
    """
    Compute Euclidean distance in CIE Lab space (approximate Delta-E).
    Returns a float32 2D array with distances.
    """
    golden_lab = cv2.cvtColor(golden, cv2.COLOR_BGR2Lab)
    test_lab = cv2.cvtColor(test, cv2.COLOR_BGR2Lab)
    delta = golden_lab.astype("float32") - test_lab.astype("float32")
    delta_e = np.sqrt(np.sum(delta**2, axis=2))
    return delta_e
