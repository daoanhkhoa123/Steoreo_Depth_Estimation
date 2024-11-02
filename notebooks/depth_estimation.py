<<<<<<< HEAD
from cv2.typing import MatLike
import draw_show_img_utils as dsiu
import cv2
import numpy as np


def depth_map(imgL: MatLike, imgR: MatLike, window_size=5):

    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=-1,  # NOTE: Debug
        numDisparities=16*16,  # max_disp has to be dividable by 16 f. E. HH 192, 256 NOTE: Debug
        blockSize=window_size,
        P1=9 * 3 * window_size,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=128 * 3 * window_size,
        disp12MaxDiff=12,
        uniquenessRatio=40,
        speckleWindowSize=50,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 70000
    sigma = 1.7

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(
        matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    # important to put "imgL" here!!!
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)
    filteredImg = cv2.normalize(
        src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)

    return filteredImg


if __name__ == "__main__":
    left_path, right_path = dsiu.get_img_path("house")
    image_left = cv2.imread(left_path)
    image_right = cv2.imread(right_path)

    house_disparity = depth_map(image_left, image_right)
    image_left = cv2.pyrDown(image_left)
    image_right = cv2.pyrDown(image_right)
    dsiu.cv2_imshow(house_disparity)
=======
from cv2.typing import MatLike
import draw_show_img_utils as dsiu
import cv2
import numpy as np


def depth_map(imgL: MatLike, imgR: MatLike, window_size=5):

    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=-1,  # NOTE: Debug
        numDisparities=16*16,  # max_disp has to be dividable by 16 f. E. HH 192, 256 NOTE: Debug
        blockSize=window_size,
        P1=9 * 3 * window_size,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=128 * 3 * window_size,
        disp12MaxDiff=12,
        uniquenessRatio=40,
        speckleWindowSize=50,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 70000
    sigma = 1.7

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(
        matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    # important to put "imgL" here!!!
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)
    filteredImg = cv2.normalize(
        src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)

    return filteredImg


if __name__ == "__main__":
    left_path, right_path = dsiu.get_img_path("house")
    image_left = cv2.imread(left_path)
    image_right = cv2.imread(right_path)

    house_disparity = depth_map(image_left, image_right)
    image_left = cv2.pyrDown(image_left)
    image_right = cv2.pyrDown(image_right)
    dsiu.cv2_imshow(house_disparity)
>>>>>>> 60f0e62cdf3bc6b7eabef67302821120c02bcc84
