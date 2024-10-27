from cv2.typing import MatLike
from numpy.typing import NDArray
from numba import jit

import numpy as np
import cv2
from typing import Tuple

def draw_keypoints_and_match(img1: NDArray, img2: NDArray, nfeatures: int = 500) -> Tuple[NDArray, NDArray, NDArray]:
    """This function is used for finding keypoints and dercriptors in the image and
        find best matches using brute force/FLANN based matcher."""
    #tao ORB phat hien dac diem chung
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    #Tạo bruteforce và tìm các điểm khớp
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    #Tạo danh sách tọa độ điểm khớp nhau
    kp1_pts = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
    kp2_pts = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)

    
    matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return kp1_pts, kp2_pts, matched_img



def compute_Fundamental_matrix(kp1_list: NDArray, kp2_list: NDArray) -> NDArray:
    """This function is used to calculate the F matrix from a set of 8 points using SVD.
        Furthermore, the rank of F matrix is reduced from 3 to 2 to make the epilines converge."""

    assert len(kp1_list) == 8, f"len(kp1_list) must be 8 for 8 points. Got {len(kp1_list)} != 8"
    assert len(kp2_list) == 8, f"len(kp2_list) must be 8 for 8 points. Got {len(kp2_list)} != 8"


    A_mat = np.array([
        [kp1_list[i][0] * kp2_list[i][0], kp1_list[i][0] * kp2_list[i][1], kp1_list[i][0],
         kp1_list[i][1] * kp2_list[i][0], kp1_list[i][1] * kp2_list[i][1], kp1_list[i][1],
         kp2_list[i][0], kp2_list[i][1], 1]
        for i in range(8)
    ])
    _, _, Vt = np.linalg.svd(A_mat)
    F_mat = Vt[-1].reshape(3, 3)

    Uf, Df, Vft = np.linalg.svd(F_mat)
    Df[2] = 0 

    return Uf @ np.diag(Df) @ Vft


def RANSAC_F_mat(kp1_list: NDArray, kp2_list: NDArray, max_inliers=20, threshold=0.05, max_iter=1000) -> NDArray:
    """This method is used to shortlist the best F matrix using RANSAC based on the number of inliers."""

    assert len(kp1_list) == len(kp2_list), f"kp1_list must have the same length as kp2_list. Got {len(kp1_list)} != {len(kp2_list)}"


    best_F_mat = None
    best_inliers_count = 0

    for _ in range(max_iter):
        sample_idx = np.random.choice(len(kp1_list), size=8, replace=False)
        F_mat = compute_Fundamental_matrix(kp1_list[sample_idx], kp2_list[sample_idx])

        # Tìm inliers dựa trên điều kiện khoảng cách
        inliers = []
        for i in range(len(kp1_list)):
            x1 = np.array([kp1_list[i][0], kp1_list[i][1], 1])
            x2 = np.array([kp2_list[i][0], kp2_list[i][1], 1])

            # Nếu khoảng cách nhỏ hơn threshold thì thêm vào inliers
            if np.abs(x2.T @ F_mat @ x1) < threshold:
                inliers.append(i)

        # Cập nhật nếu số inliers hiện tại lớn hơn giá trị tốt nhất
        if len(inliers) > best_inliers_count:
            best_inliers_count = len(inliers)
            best_F_mat = F_mat

            # Nếu số inliers đã vượt quá ngưỡng max_inliers, thoát khỏi vòng lặp
            if best_inliers_count >= max_inliers:
                break

    return best_F_mat
  

def compute_Essential_matrix(F_mat: NDArray, K1: NDArray, K2: NDArray) -> NDArray:
    """Optimized calculation of the Essential matrix with input validation."""
    
    
    if F_mat.shape != (3, 3):
        raise ValueError("F_mat must be a 3x3 matrix.")
    if K1.shape != (3, 3) or K2.shape != (3, 3):
        raise ValueError("K1 and K2 must be 3x3 intrinsic matrices.")
    
    
    K2_T = K2.T
    E = K2_T @ F_mat @ K1
    
    return E


@jit(nopython=True)
def get_camerapose(E_mat: NDArray) -> list[list[NDArray]]:
    """
    This function extracts all the camera pose solutions from the E matrix

    Consult at: https://www.uio.no/studier/emner/matnat/its/nedlagte-emner/UNIK4690/v16/forelesninger/lecture_7_3-pose-from-epipolar-geometry.pdf
    Slide 10
    
    """
    U_mat, _, Vt = np.linalg.svd(E_mat)
    
    W_mat = np.array([[0, 1, 0],
                      [-1, 0, 0],
                      [0, 0, 1]])

    #(Ma Trận quay)
    R1_mat = U_mat @ W_mat @ Vt
    R2_mat = U_mat @ W_mat.T @ Vt
    
    #đảm bảo ma trận quay chính xác :q
    if np.linalg.det(R1_mat) < 0:
        R1_mat = -R1_mat
    if np.linalg.det(R2_mat) < 0:
        R2_mat = -R2_mat

    # hai khả năng cho vector t? (vị trí camera) từ cột thứ ??3 của ma trận U
    C1_mat = U_mat[:, 2]
    C2_mat = -U_mat[:, 2]

    return [[R1_mat, C1_mat], [R1_mat, C2_mat], [R2_mat, C1_mat], [R2_mat, C2_mat]]



def disambiguate_camerapose(camera_pose: list[list[NDArray]], list_kp1: NDArray):
    """This fucntion is used to find the correct camera pose based on the chirelity condition from all 4 solutions."""
    max_len = 0
    best_camera_pose = None

    # calculating 3d points
    for pose in camera_pose:
        R, C = pose
        front_points = list()

        for point in list_kp1:
            # chirelity check
            X_mat = np.array([point[0], point[1], 1])
            V_mat = R @ X_mat + C

            # if point stand in front of camera
            if V_mat[2] > 0 and (R[2] @ (X_mat - C) > 0):
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

    assert len(img1src.shape) == 2, f"Image 1 must be grayscale (2D matrix). Got {len(img1src.shape)} dimensions."
    assert len(img2src.shape) == 2, f"Image 2 must be grayscale (2D matrix). Got {len(img2src.shape)} dimensions."

    h, w = img1src.shape
    img1color = cv2.cvtColor(img1src, cv2.COLOR_GRAY2BGR)
    img2color = cv2.cvtColor(img2src, cv2.COLOR_GRAY2BGR)

    np.random.seed(random_seed)
    for r, pt1, pt2 in zip(lines, pts1src, pts2src):
        color = tuple(np.random.randint(0, 255, 3).tolist())

        x0, y0 = 0, int(-r[2] / r[1])
        x1, y1 = w, int(-(r[2] + r[0] * w) / r[1])

        img1color = cv2.line(img1color, (x0, y0), (x1, y1), color=color, thickness=1)
        img1color = cv2.circle(img1color, (int(pt1[0]), int(pt1[1])), 5, color, -1)
        img2color = cv2.circle(img2color, (int(pt2[0]), int(pt2[1])), 5, color, -1)

    return img1color, img2color


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-port", type=int, default=5000)
    args = parser.parse_args()
    port = args.port

    print(port)
