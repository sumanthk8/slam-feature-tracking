import numpy as np


def get_homography_params(x0s, x1s):
    A = np.empty((0, 9))
    # b = np.zeros(12)

    for i in range(4):
        A = np.append(A, np.array([[x0s[i][0], x0s[i][1], 1, 0, 0, 0, -x0s[i][0]*x1s[i][0], -x0s[i][1]*x1s[i][0], -x1s[i][0]]]), axis=0)
        A = np.append(A, np.array([[0, 0, 0, x0s[i][0], x0s[i][1], 1, -x0s[i][0]*x1s[i][1], -x0s[i][1]*x1s[i][1], -x1s[i][1]]]), axis=0)

    try:
        U, S, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None

    V = Vt.T

    return V[:, -1]


def get_homography_errors(pts1, pts2, matches, homography):

    errors = np.zeros((len(matches), 2))

    homography_matrix = np.reshape(homography, (3, 3))

    for i in range(len(matches)):
        x1, y1 = pts1[matches[i][0]]
        x2, y2 = pts2[matches[i][1]]

        projected = homography_matrix @ np.array([x1, y1, 1])

        errors[i] = (projected[0]/projected[2]-x2, projected[1]/projected[2]-y2)

    return errors



