import numpy as np
from matplotlib import pyplot as plt
import cv2
import time

import utilities.detection.orb_detection as orb
import utilities.visualization as vis
import utilities.matching.feature_matching as match
import utilities.ransac as ransac
import utilities.odometry.vis_odometry as od

if __name__ == '__main__':

    start = time.time()

    img1 = cv2.cvtColor(cv2.imread("resources/testing/door/IMG_2019.jpg"), cv2.COLOR_BGR2RGB)

    key_pts1 = orb.get_key_points(img1)
    supp_key_pts1 = orb.non_max_suppression(img1, key_pts1)

    supp_key_pts1, desc1 = orb.get_descriptors(img1, supp_key_pts1)
    pts1 = orb.get_pts_from_key_points(supp_key_pts1)
    print("Pts 1 Detected: ", len(pts1))
    # print("Time: ", time.time() - start)


    img2 = cv2.cvtColor(cv2.imread("resources/testing/door/IMG_2020.jpg"), cv2.COLOR_BGR2RGB)

    key_pts2 = orb.get_key_points(img2)
    supp_key_pts2 = orb.non_max_suppression(img2, key_pts2)

    supp_key_pts2, desc2 = orb.get_descriptors(img2, supp_key_pts2)
    pts2 = orb.get_pts_from_key_points(supp_key_pts2)
    print("Pts 2 Detected: ", len(pts2))
    # print("Time: ", time.time() - start)


    matches = match.get_match_inds_from_dmatches(match.get_dmatches(desc1, desc2))
    print("Pre-Ransac Matches: ", len(matches))
    # print("Time: ", time.time() - start)


    #---attempt at different feature matching techniques---

    # A, ransac_matches = ransac.ransac_affine(pts1, pts2, matches)
    # H, ransac_matches = ransac.ransac_homography(pts1, pts2, matches)
    F, ransac_matches = ransac.ransac_fundamental(pts1, pts2, matches)
    # F, ransac_matches = ransac.lsrl_fundamental(pts1, pts2, matches)

    print("Post-Ransac Matches: ", len(ransac_matches))
    # print("Time: ", time.time() - start)


    f, axarr = plt.subplots(1, 2)

    f.set_size_inches(16, 8)

    pts1_matches = pts1[ransac_matches[:, 0]]
    pts2_matches = pts2[ransac_matches[:, 1]]

    vis.add_imgs_and_pts_to_plt(img1, pts1_matches, axarr[0])
    vis.add_imgs_and_pts_to_plt(img2, pts2_matches, axarr[1])

    vis.add_match_lines_to_plt(pts1, pts2, ransac_matches, axarr, len(ransac_matches))

    k = np.array([[26, 0, 2016],
                              [0, 26*(4/3), 1512],
                              [0, 0, 1]])

    essential_mat = k.T @ F.reshape((3, 3)) @ k

    rot1, rot2, translation = od.get_vis_odometry(essential_mat)

    print("Rot1: ", rot1, " Rot2: ", rot2, " Translation: ", translation)

    print("Time: ", time.time() - start)
    plt.show()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
