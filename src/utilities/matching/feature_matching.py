import cv2
import numpy as np


def get_dmatches(desc1, desc2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    dmatches = bf.match(desc1, desc2)
    dmatches = sorted(dmatches, key=lambda x: x.distance)

    return dmatches


def get_match_inds_from_dmatches(dmatches):
    matches = np.zeros((len(dmatches), 2), dtype=int)

    for i in range(len(dmatches)):
        matches[i] = (dmatches[i].queryIdx, dmatches[i].trainIdx)

    return matches