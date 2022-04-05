import math

import numpy as np
from scipy.optimize import least_squares


def get_fundamental_params(xLs, xRs):
    A = np.ones((9, 9))

    for i in range(len(xLs)):
        uL, vL = xLs[i]
        uR, vR = xRs[i]

        A[i] = np.array([uL*uR, uL*vR, uL, vL*uR, vL*vR, vL, uR, vR, 1])

    try:
        F = np.linalg.solve(A, np.zeros((9, 1)))
    except np.linalg.LinAlgError:
        return None

    F = F.reshape((3, 3))

    try:
        U, S, Vt = np.linalg.svd(F)
    except np.linalg.LinAlgError:
        return None

    F = U @ np.array([[S[0], 0, 0], [0, S[1], 0], [0, 0, 0]]) @ Vt
    # F = F/np.linalg.norm(F)

    return F.flatten()


def get_fundamental_params_LSRL(xLs, xRs):
    # A = np.ones((len(xLs), 9))
    #
    # for i in range(len(xLs)):
    #     uL, vL = xLs[i]
    #     uR, vR = xRs[i]
    #
    #     A[i] = np.array([uL * uR, uL * vR, uL, vL * uR, vL * vR, vL, uR, vR, 1])

    res = least_squares(fun=objective_func, x0=np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]]).flatten(), \
                        method='lm', jac='2-point', args=(xLs, xRs))

    optimized_F = res.x

    return optimized_F


def objective_func(F, xLs, xRs):
    errors = np.empty([])

    F_matrix = F.reshape((3,3))

    for i in range(len(xLs)):
        x1, y1 = xLs[i]
        x2, y2 = xRs[i]

        errors = np.append(errors, get_pt_to_line_error(F_matrix @ np.array([x2, y2, 1]), np.array([x1, y1, 1])))
        errors = np.append(errors, get_pt_to_line_error(F_matrix.T @ np.array([x1, y1, 1]), np.array([x2, y2, 1])))

    return errors


def get_fundamental_errors(pts1, pts2, matches, F):

    errors = np.zeros((len(matches), 2))

    F_matrix = np.reshape(F, (3, 3))
    print(F_matrix)

    for i in range(len(matches)):
        x1, y1 = pts1[matches[i][0]]
        x2, y2 = pts2[matches[i][1]]

        errors[i, 0] = get_pt_to_line_error(F_matrix @ np.array([x2, y2, 1]), np.array([x1, y1, 1]))
        errors[i, 1] = get_pt_to_line_error(F_matrix.T @ np.array([x1, y1, 1]), np.array([x2, y2, 1]))

    return errors


def get_pt_to_line_error(line, pt):
    a, b, c = line
    u, v, w = pt

    error = (a * u + b * v + c) / math.sqrt(a ** 2 + b ** 2)

    return error

# def get_essential_matrix():