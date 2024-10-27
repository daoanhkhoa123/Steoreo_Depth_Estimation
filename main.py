import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from calibration import draw_keypoints_and_match, drawlines, RANSAC_F_matrix, calculate_E_matrix, extract_camerapose, disambiguate_camerapose
from rectification import rectification
from correspondence import ssd_correspondence
from depth import disparity_to_depth

def main():
    number = 1

    img1 = cv2.imread("house_0.png", 0)
    img2 = cv2.imread("house_1.png", 0)

    width = int(img1.shape[1] * 0.3)
    height = int(img1.shape[0] * 0.3)
    img1 = cv2.resize(img1, (width, height), interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, (width, height), interpolation=cv2.INTER_AREA)

    #camera thingy
    camera_params = [
        (np.array([[5299.313, 0, 1263.818], [0, 5299.313, 977.763], [0, 0, 1]]),
         np.array([[5299.313, 0, 1438.004], [0, 5299.313, 977.763], [0, 0, 1]])),
        (np.array([[4396.869, 0, 1353.072], [0, 4396.869, 989.702], [0, 0, 1]]),
         np.array([[4396.869, 0, 1538.86], [0, 4396.869, 989.702], [0, 0, 1]])),
        (np.array([[5806.559, 0, 1429.219], [0, 5806.559, 993.403], [0, 0, 1]]),
         np.array([[5806.559, 0, 1543.51], [0, 5806.559, 993.403], [0, 0, 1]]))
    ]

    max_iter = 20
    for i in range(max_iter):
        try:
            list_kp1, list_kp2 = draw_keypoints_and_match(img1, img2)
            F = RANSAC_F_matrix([list_kp1, list_kp2])
            K1, K2 = camera_params[number - 1]
            E = calculate_E_matrix(F, K1, K2)
            camera_poses = extract_camerapose(E)
            best_camera_pose = disambiguate_camerapose(camera_poses, list_kp1)
            
            pts1 = np.int32(list_kp1)
            pts2 = np.int32(list_kp2)

            rectified_pts1, rectified_pts2, img1_rectified, img2_rectified = rectification(
                img1, img2, pts1, pts2, F)
            break
        except Exception:
            continue

    lines1 = cv2.computeCorrespondEpilines(rectified_pts2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    img5, img6 = drawlines(img1_rectified, img2_rectified, lines1, rectified_pts1, rectified_pts2)

    lines2 = cv2.computeCorrespondEpilines(rectified_pts1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    img3, img4 = drawlines(img2_rectified, img1_rectified, lines2, rectified_pts2, rectified_pts1)

    disparity_map_unscaled, disparity_map_scaled = ssd_correspondence(img1_rectified, img2_rectified)
    plt.imshow(disparity_map_scaled, cmap='hot')

    #DEPTH numbers
    params = [(177.288, 5299.313), (144.049, 4396.869), (174.019, 5806.559)]
    baseline, f = params[number - 1]
    depth_map, depth_array = disparity_to_depth(baseline, f, disparity_map_unscaled)

    plt.figure(3)
    plt.imshow(depth_map, cmap='hot')
    plt.show()

if __name__ == "__main__":
    main()
