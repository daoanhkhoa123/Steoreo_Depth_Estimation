from cv2.typing import MatLike
from numpy.typing import NDArray

import cv2
import random as rd
import numpy as np


def draw_keypoints_and_match(img1: MatLike, img2: MatLike, nfeatures=500) -> tuple[NDArray, NDArray, MatLike]:
    """This function is used for finding keypoints and dercriptors in the image and
        find best matches using brute force/FLANN based matcher."""

    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # return kp1_list, kp2_list, img
    return np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32), \
        np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32), \
        cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


def compute_Fundamental_matrix(kp1_list: NDArray, kp2_list: NDArray) -> NDArray:
    """This function is used to calculate the F matrix from a set of 8 points using SVD.
        Furthermore, the rank of F matrix is reduced from 3 to 2 to make the epilines converge."""

    assert len(kp1_list) == 8, f"len(kp1_list) must be 8 for 8 points. Got {
        len(kp1_list)} != 8"
    assert len(kp2_list) == 8, f"len(kp2_list) must be 8 for 8 points. Got {
        len(kp2_list)} != 8"

    A_mat = np.zeros((len(kp1_list), 9))

    for i in range(len(kp1_list)):
        x1 = kp1_list[i][0]
        y1 = kp1_list[i][1]

        x2 = kp2_list[i][0]
        y2 = kp2_list[i][1]

        A_mat[i] = np.array([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1])

    Vt = np.linalg.svd(A_mat)[-1]
    F_mat = Vt[-1, :].reshape(3, 3)

    # force F to rank 2
    Uf, Df, Vft = np.linalg.svd(F_mat)
    Df[2] = 0

    # return F mat
    return Uf.dot(np.dot(np.diag(Df[i]), Vft))


def calculate_F_matrix(list_kp1, list_kp2):
    """This function is used to calculate the F matrix from a set of 8 points using SVD.
        Furthermore, the rank of F matrix is reduced from 3 to 2 to make the epilines converge."""

    A = np.zeros(shape=(len(list_kp1), 9))

    for i in range(len(list_kp1)):
        x1, y1 = list_kp1[i][0], list_kp1[i][1]
        x2, y2 = list_kp2[i][0], list_kp2[i][1]
        A[i] = np.array([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1])

    U, s, Vt = np.linalg.svd(A)
    F = Vt[-1, :]
    F = F.reshape(3, 3)

    # Downgrading the rank of F matrix from 3 to 2
    Uf, Df, Vft = np.linalg.svd(F)
    Df[2] = 0
    s = np.zeros((3, 3))
    for i in range(3):
        s[i][i] = Df[i]

    F = np.dot(Uf, np.dot(s, Vft))
    return F


def RANSAC_F_mat(kp1_list: NDArray, kp2_list: NDArray, max_inliers=20, threshold=0.05, max_iter=1000) -> NDArray:
    """This method is used to shortlist the best F matrix using RANSAC based on the number of inliers."""

    assert len(kp1_list) == len(kp2_list), f"kp1_list must has the same length as kp2_list. Got {
        len(kp1_list)} != len(kp2_list)"

    inliers_count = 0
    for i in range(max_iter):
        sample_idx = np.random.choice(len(kp1_list), size=8)
        F_mat = compute_Fundamental_matrix(
            kp1_list[sample_idx], kp2_list[sample_idx])

        # NOTE: just need one list as we only need to
        # see how many inliers are there, i.e length of list
        sample_inliers1 = list()
        # sample_inliers_2 = list()

        # get inliers list
        for i in range(len(kp1_list)):
            # homogeneous coordinate of machted points
            x1 = np.array([kp1_list[i][0], kp1_list[i][1], 1])
            x2 = np.array([kp2_list[i][0], kp2_list[i][1], 1])

            # if distance < threshold then add to list
            if np.abs(x2.T.dot(F_mat.dot(x1))) < threshold:
                sample_inliers1.append(kp1_list[i])
                # sample_inliers_2.append(kp2_list[i])

        Best_F_mat = None

        # if len(sample_inliers1) > inliers_count:
        #     inliers_count = len(sample_inliers1)
        #     Best_F = F_mat

        if len(sample_inliers1) > max_inliers:
            # print("Number of inliers", len(sample_inliers_1))
            max_inliers = len(sample_inliers1)

            Best_F = F_mat

        return Best_F


def compute_Essential_matrix(F_mat: NDArray, K1: NDArray, K2: NDArray) -> NDArray:
    """Calculation of Essential matrix"""
    return K2.T.dot(F_mat.dot(K1))


def get_camerapose(E_mat: NDArray) -> list[list[NDArray]]:
    """This function extracts all the camera pose solutions from the E matrix

    Consult at: https://www.uio.no/studier/emner/matnat/its/nedlagte-emner/UNIK4690/v16/forelesninger/lecture_7_3-pose-from-epipolar-geometry.pdf
    Slide 10
    """

    U_mat, _, Vt = np.linalg.svd(E_mat)
    W_mat = np.array([[0, 1, 0],
                      [-1, 0, 0],
                      [0, 0, 1]])

    R1_mat = U_mat.dot(W_mat.dot(Vt))
    R2_mat = U_mat.dot(W_mat.T.dot(Vt))
    C1_mat = U_mat[:, 2]
    C2_mat = -U_mat[:, 2]

    # return camera_poses
    return [[R1_mat, C1_mat], [R1_mat, C2_mat], [R2_mat, C1_mat], [R2_mat, C2_mat]]


def RANSAC_F_matrix(list_of_cood_list):
    """This method is used to shortlist the best F matrix using RANSAC based on the number of inliers."""

    list_kp1 = list_of_cood_list[0]
    list_kp2 = list_of_cood_list[1]
    pairs = list(zip(list_kp1, list_kp2))
    max_inliers = 20
    threshold = 0.05  # Tune this value

    for i in range(1000):
        pairs = np.random.sample(pairs, 8)
        rd_list_kp1, rd_list_kp2 = zip(*pairs)
        F = calculate_F_matrix(rd_list_kp1, rd_list_kp2)

        tmp_inliers_img1 = []
        tmp_inliers_img2 = []

        for i in range(len(list_kp1)):
            img1_x = np.array([list_kp1[i][0], list_kp1[i][1], 1])
            img2_x = np.array([list_kp2[i][0], list_kp2[i][1], 1])
            distance = abs(np.dot(img2_x.T, np.dot(F, img1_x)))
            # print(distance)

            if distance < threshold:
                tmp_inliers_img1.append(list_kp1[i])
                tmp_inliers_img2.append(list_kp2[i])

        num_of_inliers = len(tmp_inliers_img1)

        # if num_of_inliers > inlier_count:
        #     inlier_count = num_of_inliers
        #     Best_F = F

        if num_of_inliers > max_inliers:
            print("Number of inliers", num_of_inliers)
            max_inliers = num_of_inliers
            Best_F = F
            inliers_img1 = tmp_inliers_img1
            inliers_img2 = tmp_inliers_img2
            # print("Best F matrix", Best_F)

    return Best_F


def calculate_E_matrix(F, K1, K2):
    """Calculation of Essential matrix"""

    E = np.dot(K2.T, np.dot(F, K1))
    return E


def extract_camerapose(E):
    """This function extracts all the camera pose solutions from the E matrix"""

    U, s, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])

    C1, C2 = U[:, 2], -U[:, 2]
    R1, R2 = np.dot(U, np.dot(W, Vt)), np.dot(U, np.dot(W.T, Vt))
    # print("C1", C1, "\n", "C2", C2, "\n", "R1", R1, "\n", "R2", R2, "\n")

    camera_poses = [[R1, C1], [R1, C2], [R2, C1], [R2, C2]]
    return camera_poses


def disambiguate_camerapose(camera_pose: list[list[NDArray]], list_kp1: NDArray):
    """This fucntion is used to find the correct camera pose based on the chirelity condition from all 4 solutions."""
    max_len = 0
    best_camera_pose = None

    # calculating 3d points
    for pose in camera_pose:
        front_points = list()

        for point in list_kp1:
            # chirelity check
            X_mat = np.array([point[0], point[1], 1])  # homogeneous
            V_mat = X_mat = pose[1]

            # if point stand in front of camera
            if np.dot(pose[0][2], V_mat) > 0:
                front_points.append(point)

        # find the pose with the most front point
        if len(front_points) > max_len:
            max_len = len(front_points)
            best_camera_pose = pose

    return best_camera_pose


def drawlines(img1src: MatLike, img2src: MatLike, lines, pts1src, pts2src, random_seed=0) -> tuple[MatLike, MatLike]:
    """This fucntion is used to visualize the epilines on the images
        img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines """

    assert len(img1src.shape) == 2, f"imgage 1 must be gray image. i.e 2d matrix. Got {
        len(img1src.shape)} dimension"

    assert len(img2src.shape) == 2, f"imgage 2 must be gray image. i.e 2d matrix. Got {
        len(img2src.shape)} dimension"

    _, h = img1src.shape
    img1color = cv2.cvtColor(img1src, cv2.COLOR_GRAY2BGR)
    img2color = cv2.cvtColor(img2src, cv2.COLOR_GRAY2BGR)

    np.random.seed(random_seed)
    for r, pt1, pt2 in zip(lines, pts1src, pts2src):
        color = np.random.randint(0, 255, 3)

        # TODO:
        x0 = 0
        y0 = int(-r[2]/r[1])
        x1 = h
        y1 = int(-(r[2]+r[0]*h)/r[1])

        img1color = cv2.line(img1color, (x0, y0), (x1, y1),
                             color=color, thickness=1)
        img1color = cv2.circle(img1color, tuple(pt1), 5, color, -1)
        img2color = cv2.line(img2color, tuple(pt2), 5, color, -1)

    return img1color, img2color


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-port", type=int, default=5000)
    args = parser.parse_args()
    port = args.port

    print(port)
