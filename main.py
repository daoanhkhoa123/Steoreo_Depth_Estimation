from cv2.typing import MatLike
from numpy.typing import NDArray

import cv2
import numpy as np
import matplotlib.pyplot as plt

from modules.calibration import draw_keypoints_and_match, drawlines, RANSAC_F_matrix, compute_Essential_matrix
from modules.correspondence import ssd_correspondence
from modules.depth import disparity_to_depth
from modules import draw_show_img_utils as dsiu

from datasets.load_dataset import Data, CallibTyping


class Steroeo_Disparity:
    def __init__(self) -> None:
        self.img1 = None
        self.img2 = None
        self.pts1 = None
        self.pts2 = None

    def fit(self, img1, img2, show_matched=False) -> MatLike:
        self.img1 = img1
        self.img2 = img2

        self.pts1, self.pts2, matched_img = draw_keypoints_and_match(
            img1, img2)

        if show_matched:
            plt.imshow(show_matched)

        return matched_img

    def get_fundamental_matrix(self, max_iter=64, max_inliers=64):
        return RANSAC_F_matrix(self.pts1, self.pts2, max_iter, max_inliers)

    def get_disparity(self, block_size=15, x_search_block_size=50) -> tuple[MatLike, MatLike]:
        return ssd_correspondence(
            self.img1, self.img2, block_size, x_search_block_size)

    def get_depth(self, baseline, focal, block_size=15, x_search_block_size=50) -> tuple[MatLike, MatLike]:
        disparity_map, _ = self.get_disparity(block_size, x_search_block_size)

        return disparity_to_depth(baseline, focal, disparity_map)

    def global_smoothing(self, depth_map):
        return cv2.ximgproc.createFastGlobalSmootherFilter(
            self.img1, 8378666, 1.5).filter(depth_map)


if __name__ == "__main__":
    import os
    img1 = cv2.imread(os.path.join("datasets", "house_left.jpg"))
    img2 = cv2.imread(os.path.join("datasets", "house_right.jpg"))

    stereo = Steroeo_Disparity()
    matched = stereo.fit(img1, img2)
    dsiu.cv2_imshow(matched)

    print(stereo.get_fundamental_matrix())

    disparity, disparity_scaled = stereo.get_disparity()
    plt.imshow(disparity_scaled, cmap="jet")
    plt.show()

    img_n = stereo.global_smoothing(disparity)
    # img_n = cv2.normalize(src=disparity_scaled, dst=None, alpha=0,
    #                       beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    plt.imshow(img_n, cmap="jet")
    plt.show()
    cv2.imwrite("house_disparity.png", img_n)
