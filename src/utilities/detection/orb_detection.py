import cv2
import numpy as np
from scipy.ndimage.filters import maximum_filter

detector = cv2.ORB_create()
detector.setPatchSize(75)
detector.setMaxFeatures(10000)
# detector.setScoreType(0)


def get_key_points(img):
    key_pts = detector.detect(img)

    return key_pts


def get_pts_from_key_points(key_pts):
    pts = np.zeros((len(key_pts), 2), dtype=int)

    for i in range(len(key_pts)):
        pts[i] = (key_pts[i].pt[0], key_pts[i].pt[1])

    return pts


def get_descriptors(img, key_pts):
    key_pts, descriptors = detector.compute(img, key_pts)

    return key_pts, descriptors


def non_max_suppression(img, key_pts, suppression_size=75):
    key_pts_mat = np.empty((len(img[0]), len(img)), dtype=cv2.KeyPoint)
    R_mat = np.zeros((len(img[0]), len(img)), dtype=float)

    for key_pt in key_pts:
        x = int(key_pt.pt[0])
        y = int(key_pt.pt[1])

        key_pts_mat[x, y] = key_pt
        R_mat[x, y] = key_pt.response

    R_suppressed = R_mat == maximum_filter(R_mat, suppression_size)
    indices = np.flatnonzero(R_suppressed)

    key_pts_mat = key_pts_mat.flatten()
    return key_pts_mat[indices]






# def suppress_points(img):
