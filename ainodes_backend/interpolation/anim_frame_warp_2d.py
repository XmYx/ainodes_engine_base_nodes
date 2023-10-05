import cv2
import numpy as np
from functools import reduce


def anim_frame_warp_2d(prev_img_cv2, flip2d, angle, zoom, trans_x, trans_y, flip_t, flip_p, flip_g, flip_f):
    angle = angle
    zoom = zoom
    translation_x = trans_x
    translation_y = trans_y
    w = prev_img_cv2.shape[0]
    h = prev_img_cv2.shape[1]
    center = (w // 2, h // 2)
    trans_mat = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
    rot_mat = cv2.getRotationMatrix2D(center, angle, zoom)
    trans_mat = np.vstack([trans_mat, [0,0,1]])
    rot_mat = np.vstack([rot_mat, [0,0,1]])
    if flip2d:
        perspective_flip_theta = flip_t
        perspective_flip_phi = flip_p
        perspective_flip_gamma = flip_g
        perspective_flip_fv = flip_f
        M,sl = warpMatrix(w, h, perspective_flip_theta, perspective_flip_phi, perspective_flip_gamma, 1., perspective_flip_fv)
        post_trans_mat = np.float32([[1, 0, (w-sl)/2], [0, 1, (h-sl)/2]])
        post_trans_mat = np.vstack([post_trans_mat, [0,0,1]])
        bM = np.matmul(M, post_trans_mat)
        xform = np.matmul(bM, rot_mat, trans_mat)
    else:
        xform = np.matmul(rot_mat, trans_mat)

    return cv2.warpPerspective(
        prev_img_cv2,
        xform,
        (prev_img_cv2.shape[1], prev_img_cv2.shape[0]),
        borderMode=cv2.BORDER_WRAP# if anim_args.border == 'wrap' else cv2.BORDER_REPLICATE
    )

def warpMatrix(W, H, theta, phi, gamma, scale, fV):
    # M is to be estimated
    #M = np.eye(4, 4)

    fVhalf = np.deg2rad(fV / 2.)
    d = np.sqrt(W * W + H * H)
    sideLength = scale * d / np.cos(fVhalf)
    h = d / (2.0 * np.sin(fVhalf))
    n = h - (d / 2.0)
    f = h + (d / 2.0)

    # Translation along Z-axis by -h
    T = np.eye(4, 4)
    T[2, 3] = -h

    # Rotation matrices around x,y,z
    R = getRotationMatrixManual([phi, gamma, theta])

    # Projection Matrix
    P = np.eye(4, 4)
    P[0, 0] = 1.0 / np.tan(fVhalf)
    P[1, 1] = P[0, 0]
    P[2, 2] = -(f + n) / (f - n)
    P[2, 3] = -(2.0 * f * n) / (f - n)
    P[3, 2] = -1.0

    # pythonic matrix multiplication
    F = reduce(lambda x, y: np.matmul(x, y), [P, T, R])

    # shape should be 1,4,3 for ptsIn and ptsOut since perspectiveTransform() expects data in this way.
    # In C++, this can be achieved by Mat ptsIn(1,4,CV_64FC3);
    ptsIn = np.array([[
        [-W / 2., H / 2., 0.], [W / 2., H / 2., 0.], [W / 2., -H / 2., 0.], [-W / 2., -H / 2., 0.]
    ]])
    #ptsOut = np.array(np.zeros((ptsIn.shape), dtype=ptsIn.dtype))
    ptsOut = cv2.perspectiveTransform(ptsIn, F)

    ptsInPt2f, ptsOutPt2f = getPoints_for_PerspectiveTranformEstimation(ptsIn, ptsOut, W, H, sideLength)

    # check float32 otherwise OpenCV throws an error
    assert (ptsInPt2f.dtype == np.float32)
    assert (ptsOutPt2f.dtype == np.float32)
    M33 = cv2.getPerspectiveTransform(ptsInPt2f, ptsOutPt2f)

    return M33, sideLength

def getRotationMatrixManual(rotation_angles):
    rotation_angles = [np.deg2rad(x) for x in rotation_angles]

    phi = rotation_angles[0]  # around x
    gamma = rotation_angles[1]  # around y
    theta = rotation_angles[2]  # around z

    # X rotation
    Rphi = np.eye(4, 4)
    sp = np.sin(phi)
    cp = np.cos(phi)
    Rphi[1, 1] = cp
    Rphi[2, 2] = Rphi[1, 1]
    Rphi[1, 2] = -sp
    Rphi[2, 1] = sp

    # Y rotation
    Rgamma = np.eye(4, 4)
    sg = np.sin(gamma)
    cg = np.cos(gamma)
    Rgamma[0, 0] = cg
    Rgamma[2, 2] = Rgamma[0, 0]
    Rgamma[0, 2] = sg
    Rgamma[2, 0] = -sg

    # Z rotation (in-image-plane)
    Rtheta = np.eye(4, 4)
    st = np.sin(theta)
    ct = np.cos(theta)
    Rtheta[0, 0] = ct
    Rtheta[1, 1] = Rtheta[0, 0]
    Rtheta[0, 1] = -st
    Rtheta[1, 0] = st

    R = reduce(lambda x, y: np.matmul(x, y), [Rphi, Rgamma, Rtheta])

    return R

def getPoints_for_PerspectiveTranformEstimation(ptsIn, ptsOut, W, H, sidelength):
    ptsIn2D = ptsIn[0, :]
    ptsOut2D = ptsOut[0, :]
    ptsOut2Dlist = []
    ptsIn2Dlist = []

    for i in range(0, 4):
        ptsOut2Dlist.append([ptsOut2D[i, 0], ptsOut2D[i, 1]])
        ptsIn2Dlist.append([ptsIn2D[i, 0], ptsIn2D[i, 1]])

    pin = np.array(ptsIn2Dlist) + [W / 2., H / 2.]
    pout = (np.array(ptsOut2Dlist) + [1., 1.]) * (0.5 * sidelength)
    pin = pin.astype(np.float32)
    pout = pout.astype(np.float32)

    return pin, pout
