from numpy.typing import NDArray
from cv2.typing import MatLike

import numpy as np
import cv2


def rectification(img1: MatLike, img2: MatLike, pts1: NDArray, pts2: NDArray, F: NDArray) -> tuple[NDArray, NDArray, MatLike, MatLike]:
    """Rectify images using stereoRectifyUncalibrated to align epilines as horizontal without camera distortion parameters."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Stereo rectification using uncalibrated method
    valid, H1, H2 = cv2.stereoRectifyUncalibrated(
        pts1, pts2, F, imgSize=(w1, h1))
    assert valid, "Stereo rectification failed. Check the fundamental matrix and matching points."

    # Apply homography matrices to the feature points
    pts1_homogeneous = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_homogeneous = np.hstack((pts2, np.ones((pts2.shape[0], 1))))

    pts1_homogeneous = (H1 @ pts1_homogeneous.T).T
    pts2_homogeneous = (H2 @ pts2_homogeneous.T).T

    # Normalize points
    pts1_homogeneous /= pts1_homogeneous[:, 2].reshape(-1, 1)
    pts2_homogeneous /= pts2_homogeneous[:, 2].reshape(-1, 1)

    # Remove homogeneous coordinate
    pts1_homogeneous = pts1_homogeneous[:, :2].astype(int)
    pts2_homogeneous = pts2_homogeneous[:, :2].astype(int)

    # Rectify the images
    img1_rectified = cv2.warpPerspective(img1, H1, (w1, h1))
    img2_rectified = cv2.warpPerspective(img2, H2, (w2, h2))

    return pts1_homogeneous, pts2_homogeneous, img1_rectified, img2_rectified
