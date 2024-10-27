from cv2.typing import MatLike


def disparity_to_depth(baseline: float, f: float, img: MatLike) -> tuple[MatLike, MatLike]:
    """This is used to compute the depth values from the disparity map"""

    epsilon = 1e-6
    img = img + epsilon

    depth_map = 1 / img
    depth_array = (baseline * f) / img

    return depth_map, depth_array
