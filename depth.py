import numpy as np
import cv2

def disparity_to_depth(baseline, f, img):
    """This is used to compute the depth values from the disparity map"""

    
    epsilon = 1e-6
    img = img + epsilon

    depth_map = 1 / img
    depth_array = (baseline * f) / img

    return depth_map, depth_array
