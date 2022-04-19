import numpy as np
from src.utilities.transformations.affine_transformation import get_affine_transformation_params, get_errors
from src.utilities.transformations.homography import get_homography_params, get_homography_errors
from src.utilities.epipolar_geometry.epipolar_geometry import get_fundamental_errors, get_fundamental_params_LSRL


def ransac_affine(pts1, pts2, matches, num_iterations=1000):
    best_A = None
    inlier_matches = None

    ##############################
    for i in range(num_iterations):

        match_samples = matches[np.random.choice(len(matches), 3, False)]

        pts1_sample = np.zeros((3, 2))
        pts2_sample = np.zeros((3, 2))

        for i in range(3):
            pts1_sample[i] = pts1[match_samples[i][0]]
            pts2_sample[i] = pts2[match_samples[i][1]]

        affine = get_affine_transformation_params(pts1_sample, pts2_sample)

        if affine is None:
            continue

        curr_inliers = get_affine_match_inliers(pts1, pts2, matches, affine, 3)

        if inlier_matches is None or len(curr_inliers) > len(inlier_matches):
            best_A = affine
            inlier_matches = curr_inliers

    # print(len(inlier_matches))

    ##############################

    return best_A, inlier_matches


def get_affine_match_inliers(pts1, pts2, matches, affine, threshold):

    inlier_matches = np.empty((0, 2), dtype=int)

    errors = get_errors(pts1, pts2, matches, affine)

    for i in range(len(errors)):
        if abs(errors[i][0]) < threshold and abs(errors[i][1]) < threshold:
            inlier_matches = np.append(inlier_matches, [matches[i]], axis=0)

    return inlier_matches


def ransac_homography(pts1, pts2, matches, num_iterations=1000):
    best_H = None
    inlier_matches = None

    ##############################
    for i in range(num_iterations):

        match_samples = matches[np.random.choice(len(matches), 4, False)]

        pts1_sample = np.zeros((4, 2))
        pts2_sample = np.zeros((4, 2))

        for i in range(4):
            pts1_sample[i] = pts1[match_samples[i][0]]
            pts2_sample[i] = pts2[match_samples[i][1]]

        homography = get_homography_params(pts1_sample, pts2_sample)

        if homography is None:
            continue

        curr_inliers = get_homography_match_inliers(pts1, pts2, matches, homography, 3)

        if inlier_matches is None or len(curr_inliers) > len(inlier_matches):
            best_H = homography
            inlier_matches = curr_inliers

    # print(len(inlier_matches))

    ##############################

    return best_H, inlier_matches


def get_homography_match_inliers(pts1, pts2, matches, homography, threshold=3):

    inlier_matches = np.empty((0, 2), dtype=int)

    errors = get_homography_errors(pts1, pts2, matches, homography)

    for i in range(len(errors)):
        if abs(errors[i][0]) < threshold and abs(errors[i][1]) < threshold:
            inlier_matches = np.append(inlier_matches, [matches[i]], axis=0)

    return inlier_matches


def ransac_fundamental(pts1, pts2, matches, num_iterations=150):
    best_F = None
    inlier_matches = None

    ##############################
    for i in range(num_iterations):

        match_samples = matches[np.random.choice(len(matches), 8, False)]

        pts1_sample = np.zeros((8, 2))
        pts2_sample = np.zeros((8, 2))

        for i in range(8):
            pts1_sample[i] = pts1[match_samples[i][0]]
            pts2_sample[i] = pts2[match_samples[i][1]]

        F = get_fundamental_params_LSRL(pts1_sample, pts2_sample)

        if F is None:
            continue

        curr_inliers = get_fundamental_match_inliers(pts1, pts2, matches, F, 3)

        if inlier_matches is None or len(curr_inliers) > len(inlier_matches):
            best_F = F
            inlier_matches = curr_inliers

    # print(len(inlier_matches))

    ##############################

    return best_F, inlier_matches


def get_fundamental_match_inliers(pts1, pts2, matches, F, threshold=3):

    inlier_matches = np.empty((0, 2), dtype=int)

    errors = get_fundamental_errors(pts1, pts2, matches, F)

    for i in range(len(errors)//2):
        if abs(errors[2*i]) < threshold and abs(errors[2*i+1]) < threshold:
            inlier_matches = np.append(inlier_matches, [matches[i]], axis=0)

    return inlier_matches

def lsrl_fundamental(pts1, pts2, matches):

    xLs = pts1[[match[0] for match in matches]]
    xRs = pts2[[match[1] for match in matches]]

    F = get_fundamental_params_LSRL(xLs, xRs)

    try:
        U, S, Vt = np.linalg.svd(F)
        F = U @ np.array([[S[0], 0, 0], [0, S[1], 0], [0, 0, 0]]) @ Vt
    except np.linalg.LinAlgError:
        print("Error")

    inliers = get_fundamental_match_inliers(pts1, pts2, matches, F, 50)

    return F, inliers



