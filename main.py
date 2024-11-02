import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

from modules.calibration import draw_keypoints_and_match, drawlines, RANSAC_F_mat, compute_Essential_matrix
from old.rectification import rectification
from modules.correspondence import ssd_correspondence
from modules.depth import disparity_to_depth
from modules import draw_show_img_utils as dsiu
from datasets.load_dataset import Data


# Its all about resolution - Its a trade off between resolution and time of computation

def main():
    number = 0

    width = int(img1.shape[1] * 0.3)  # 0.3
    height = int(img1.shape[0] * 0.3)  # 0.3

    img1 = cv2.resize(img1, (width, height), interpolation=cv2.INTER_AREA)
    # img1 = cv2.GaussianBlur(img1,(5,5),0)
    img2 = cv2.resize(img2, (width, height), interpolation=cv2.INTER_AREA)
    # img2 = cv2.GaussianBlur(img2,(5,5),0)

    # __________________Camera Parameters________________________________
    K11 = np.array([[5299.313,  0,   1263.818],
                    [0,      5299.313, 977.763],
                    [0,          0,       1]])
    K12 = np.array([[5299.313,   0,    1438.004],
                    [0,      5299.313,  977.763],
                    [0,           0,      1]])

    K21 = np.array([[4396.869, 0, 1353.072],
                    [0, 4396.869, 989.702],
                    [0, 0, 1]])
    K22 = np.array([[4396.869, 0, 1538.86],
                    [0, 4396.869, 989.702],
                    [0, 0, 1]])

    K31 = np.array([[5806.559, 0, 1429.219],
                    [0, 5806.559, 993.403],
                    [0, 0, 1]])
    K32 = np.array([[5806.559, 0, 1543.51],
                    [0, 5806.559, 993.403],
                    [0, 0, 1]])
    camera_params = [(K11, K12), (K21, K22), (K31, K32)]

    data = Data(1)
    img1 = data.img1
    img2 = data.img2
    camera_params = data.calib["cam0"], data.calib["cam1"]
    baseline = data.calib["baseline"]
    f = data.calib[""]

    list_kp1, list_kp2, matched_image = draw_keypoints_and_match(img1, img2)
    dsiu.cv2_imshow(matched_image)

    # _______________________________Calibration_______________________________

    F = RANSAC_F_mat(list_kp1, list_kp2)
    print("F matrix", F)
    print("=="*20, '\n')
    number = 1
    K1, K2 = camera_params[number-1]
    E = compute_Essential_matrix(F, K1, K2)
    print("E matrix", E)
    print("=="*20, '\n')
    pts1 = np.int32(list_kp1)
    pts2 = np.int32(list_kp2)

    # ____________________________Rectification________________________________

    rectified_pts1, rectified_pts2, img1_rectified, img2_rectified = rectification(
        img1, img2, pts1, pts2, F)

    # Find epilines corresponding to points in right image (second image) and drawing its lines on left image

    lines1 = cv2.computeCorrespondEpilines(
        rectified_pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img1_rectified, img2_rectified,
                           lines1, rectified_pts1, rectified_pts2)

    # Find epilines corresponding to points in left image (first image) and drawing its lines on right image

    lines2 = cv2.computeCorrespondEpilines(
        rectified_pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img2_rectified, img1_rectified,
                           lines2, rectified_pts2, rectified_pts1)

    dsiu.cv2_imshow(winname="3", img=img3)
    dsiu.cv2_imshow(winname="4", img=img4)
    dsiu.cv2_imshow(winname="5", img=img5)
    dsiu.cv2_imshow(winname="6", img=img6)

    # ____________________________Correspondance________________________________

    disparity_map_unscaled, disparity_map_scaled = ssd_correspondence(
        img1_rectified, img2_rectified)
#    cv2.imwrite(f"disparity_map_{number}.png", disparity_map_scaled)

    img_n = cv2.normalize(src=disparity_map_scaled, dst=None, alpha=0,
                          beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap1 = cv2.applyColorMap(img_n, cv2.COLORMAP_HOT)
    cv2.imwrite(f"disparity_heat_map_{number}.png", heatmap1)
    plt.figure(1)
    plt.title('Disparity Map Graysacle')
    plt.imshow(disparity_map_scaled, cmap='gray')
    plt.figure(2)
    plt.title('Disparity Map Hot')
    plt.imshow(disparity_map_scaled, cmap='hot')
    print("Done Correspondance")

    # ________________________________Depth______________________________________

    depth_map, depth_array = disparity_to_depth(
        baseline, f, disparity_map_unscaled)

    plt.figure(3)
    plt.title('Depth Map Graysacle')
    plt.imshow(depth_map, cmap='gray')
    plt.figure(4)
    plt.title('Depth Map Hot')
    plt.imshow(depth_map, cmap='hot')
    plt.show()

    print("=="*20)
    # print("Depth values", depth_array)

    # ____________________________________________________________________________


if __name__ == "__main__":
    main()
