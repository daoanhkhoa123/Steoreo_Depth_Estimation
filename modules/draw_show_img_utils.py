from typing import Sequence
from cv2.typing import MatLike

import cv2
from matplotlib import pyplot as plt
import numpy as np


def draw_quadrange(x, y, color=None, limX=None, limY=None):
    """Vẽ một hình chữ nhật của các điểm trên đồ thị."""
    plt.scatter(x, y, color=color)
    if limX and limY:
        plt.xlim(limX)
        plt.ylim(limY)


def draw_2_img(im1: MatLike, im2: MatLike, label1=None, label2=None, cmap="gray", figsize=(10, 5), sharexy=False):
    """Vẽ hai hình ảnh cạnh nhau với các tiêu đề và thang màu tuỳ chọn."""
    fig, axes = plt.subplots(1, 2, figsize=figsize,
                             sharex=sharexy, sharey=sharexy)
    for ax, img, label in zip(axes, [im1, im2], [label1, label2]):
        ax.set_title(label)
        ax.imshow(img, cmap=cmap)
    return fig, axes


def cv2_imshow(img: MatLike, winname="Image"):
    """Hiển thị hình ảnh sử dụng OpenCV."""
    cv2.imshow(winname, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def figure_to_img(fig):
    """Chuyển đổi matplotlib figure thành ảnh OpenCV."""
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)


def draw_many_imgs(imgs: Sequence[np.ndarray], labels: Sequence[str], cmap="gray", figsize=(15, 5), sharexy=False):
    """Vẽ nhiều hình ảnh cạnh nhau với các tiêu đề."""
    fig, axes = plt.subplots(
        1, len(imgs), figsize=figsize, sharex=sharexy, sharey=sharexy)
    for ax, img, label in zip(axes, imgs, labels):
        ax.set_title(label)
        ax.imshow(img, cmap=cmap)
    return fig, axes


def print_debug(*args):
    """In ra các biến để kiểm tra nhanh."""
    for i, arg in enumerate(args):
        print(f"Variable {i}:\n{arg}\n" + "_"*10)

