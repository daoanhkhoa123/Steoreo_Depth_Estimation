import numpy as np
import cv2
from numba import jit

@jit(nopython=True)
def sum_of_squared_diff(pixel_vals_1, pixel_vals_2):
    """Sum of squared distances for Correspondence"""
    if pixel_vals_1.shape != pixel_vals_2.shape:
        return -1
    return np.sum((pixel_vals_1 - pixel_vals_2) ** 2)

@jit(nopython=True)
def block_comparison(y, x, block_left, right_array, block_size, x_search_block_size, y_search_block_size):
    """Block comparison function to find minimum SSD match"""
    x_min = max(0, x - x_search_block_size)
    x_max = min(right_array.shape[1], x + x_search_block_size)
    y_min = max(0, y - y_search_block_size)
    y_max = min(right_array.shape[0], y + y_search_block_size)
    
    min_ssd = float('inf')
    min_index = (y, x)

    for j in range(y_min, y_max):
        for i in range(x_min, x_max):
            block_right = right_array[j:j + block_size, i:i + block_size]
            ssd = sum_of_squared_diff(block_left, block_right)
            if ssd < min_ssd:
                min_ssd = ssd
                min_index = (j, i)

    return min_index

@jit(nopython=True)
def ssd_correspondence(img1, img2):
    """Compute the disparity map using SSD correspondence"""
    block_size = 15
    x_search_block_size = 50
    y_search_block_size = 1
    h, w = img1.shape
    disparity_map = np.zeros((h, w))

    for y in range(block_size, h - block_size):
        for x in range(block_size, w - block_size):
            block_left = img1[y:y + block_size, x:x + block_size]
            index = block_comparison(y, x, block_left, img2, block_size, x_search_block_size, y_search_block_size)
            disparity_map[y, x] = abs(index[1] - x)

    max_pixel = np.max(disparity_map)
    min_pixel = np.min(disparity_map)

    disparity_map_scaled = ((disparity_map - min_pixel) * 255 / (max_pixel - min_pixel)).astype(np.uint8)
    
    return disparity_map, disparity_map_scaled
