from cv2.typing import MatLike
import numpy as np


def disparity_to_depth(baseline: float, f: float, img: MatLike) -> tuple[MatLike, MatLike]:
    """This is used to compute the depth values from the disparity map"""
    np.seterr(divide="ignore")
    depth_map = np.divide(1, img)  # for inf to occur
    depth_array = np.divide(baseline * f, img)

    return depth_map, depth_array
