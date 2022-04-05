import cv2
from matplotlib.patches import ConnectionPatch as cp
from matplotlib import pyplot as plt
import numpy as np


def add_pts_to_img(img, pts, radius=5, thickness=5):

    image_with_pts = img.copy()

    for pt in pts:
        cv2.circle(image_with_pts, pt, radius, (0, 255, 0), thickness=thickness)

    return image_with_pts


# def cv_add_matches_to_img(img1, key_pts1, img2, key_pts2, matches, numMatches=20):
#     matched_img = cv2.drawMatches(img1, key_pts1, img2, key_pts2, matches[-numMatches:], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, matchColor=(0,0,255))
#
#     return matched_img


def add_match_lines_to_plt(pts1, pts2, matches, axarr, num_matches=20):

    for i in range(num_matches):
        con = cp(xyA=pts1[matches[i][0]], xyB=pts2[matches[i][1]], coordsA='data', coordsB='data', axesA=axarr[0], axesB=axarr[1], color=(0, 0, 1))
        axarr[1].add_artist(con)


def add_imgs_and_pts_to_plt(img, pts, ax):
    img_with_pts = add_pts_to_img(img, pts)
    ax.imshow(img_with_pts)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
