import numpy as np


def get_affine_transformation_params(x0s, x1s):
    A = np.empty((0, 6))
    b = np.zeros(6)

    for i in range(3):
        A = np.append(A, np.array([[x0s[i][0], x0s[i][1], 0, 0, 1, 0]]), axis=0)
        A = np.append(A, np.array([[0, 0, x0s[i][0], x0s[i][1], 0, 1]]), axis=0)
        b[2*i] = x1s[i][0]
        b[2*i+1] = x1s[i][1]

    if np.linalg.det(A) == 0:
        return None

    params = np.linalg.solve(A, b)

    return params


def get_errors(pts1, pts2, matches, affine):

    errors = np.zeros((len(matches), 2))

    for i in range(len(matches)):
        x1, y1 = pts1[matches[i][0]]
        x2, y2 = pts2[matches[i][1]]
        initial = np.array([[x1, y1, 0, 0, 1, 0],
                             [0, 0, x1, y1, 0, 1]])
        projected = initial @ affine

        errors[i] = (projected[0]-x2, projected[1]-y2)

    return errors



