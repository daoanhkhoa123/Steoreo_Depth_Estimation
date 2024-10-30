from numba import jit
import numpy as np
from numpy.typing import NDArray


@jit(nopython=True)
def _sum_of_squared_diff(pixel_vals_1: NDArray[np.float32], pixel_vals_2: NDArray[np.float32]) -> float:
    """Sum of squared distances for Correspondence"""
    return np.sum((pixel_vals_1 - pixel_vals_2) ** 2)


@jit(nopython=True)
def _block_comparison(y: int, x: int, block_left: NDArray[np.float32], right_array: NDArray[np.float32], block_size: int, x_search_block_size: int, y_search_block_size: int) -> int:
    """Block comparison function to find minimum SSD match
    Return x axis index
    """
    x_min = max(0, x - x_search_block_size)
    x_max = min(right_array.shape[1], x + x_search_block_size)
    y_min = max(0, y - y_search_block_size)
    y_max = min(right_array.shape[0], y + y_search_block_size)

    min_ssd = float('inf')
    min_index = x

    for j in range(y_min, y_max):
        for i in range(x_min, x_max):
            block_right = right_array[j:j + block_size, i:i + block_size]

            if block_right.shape == block_left.shape:
                ssd = _sum_of_squared_diff(block_left, block_right)

                if ssd < min_ssd:
                    min_ssd = ssd
                    min_index = i

    return min_index


@jit(nopython=True)
def ssd_correspondence(img1: NDArray[np.float32], img2: NDArray[np.float32], block_size=5, x_search_block_size=50, y_search_block_size=1) -> tuple[NDArray[np.float32], NDArray[np.uint8]]:
    """Compute the disparity map using SSD correspondence"""
    h, w = img1.shape[:2]
    disparity_map = np.zeros((h, w), dtype=np.float32)

    for y in range(block_size, h - block_size):
        for x in range(block_size, w - block_size):
            block_left = img1[y:y + block_size, x:x + block_size]
            index = _block_comparison(
                y, x, block_left, img2, block_size, x_search_block_size, y_search_block_size)
            disparity_map[y, x] = abs(index - x)

    max_pixel = np.max(disparity_map)
    min_pixel = np.min(disparity_map)

    disparity_map_scaled = ((disparity_map - min_pixel)
                            * 255 / (max_pixel - min_pixel)).astype(np.uint8)

    return disparity_map, disparity_map_scaled
