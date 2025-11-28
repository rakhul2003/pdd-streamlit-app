# modules/align.py
import cv2
import numpy as np
import imutils



def align_images(template, image, orb_max_features, orb_keep_percent):
    """
    Align image to template using ORB + homography.
    Returns aligned image (same size as template).
    Raises RuntimeError on failure.
    """
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(int(orb_max_features))
    (kpsA, descsA) = orb.detectAndCompute(imageGray, None)
    (kpsB, descsB) = orb.detectAndCompute(templateGray, None)

    if descsA is None or descsB is None or len(kpsA) < 4 or len(kpsB) < 4:
        raise RuntimeError("Not enough keypoints/descriptors for alignment")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False) 
    knn_matches = matcher.knnMatch(descsA, descsB, k=2)

    # Ratio test
    good_matches = []
    for pair in knn_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    if len(good_matches) < 12:
        # fallback to crossCheck matching
        matcher2 = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches2 = matcher2.match(descsA, descsB)
        matches2 = sorted(matches2, key=lambda x: x.distance)
        keep = max(12, int(len(matches2) * orb_keep_percent))
        good_matches = matches2[:keep]

    if len(good_matches) < 4:
        raise RuntimeError("Insufficient good matches for homography")

    ptsA = np.zeros((len(good_matches), 2), dtype="float")
    ptsB = np.zeros((len(good_matches), 2), dtype="float")
    for i, m in enumerate(good_matches):
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt

    H, mask = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    if H is None:
        raise RuntimeError("Failed to compute homography matrix")

    (h, w) = template.shape[:2]
    aligned = cv2.warpPerspective(image, H, (w, h))
    return aligned
