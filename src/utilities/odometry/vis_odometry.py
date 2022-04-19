import numpy as np
from scipy.spatial.transform import Rotation

def get_vis_odometry(e_matrix):
    U, S, Vt = np.linalg.svd(e_matrix)

    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    Z = np.array([[0, 1, 0],
                  [-1, 0, 0],
                  [0, 0, 0]])

    R_one = U @ W @ Vt
    R_two = U @ W.T @ Vt

    R1 = Rotation.from_matrix(R_one).as_euler('zyx', degrees=True)
    R2 = Rotation.from_matrix(R_two).as_euler('zyx', degrees=True)

    t = U[:, 2]

    if t[0] < 0:
        t = -t
    elif t[0] == 0 and t[1] < 0:
        t = -t
    elif t[0] == 0 and t[2] == 0 and t[2] < 0:
        t = -t

    ##############################

    return R1, R2, t