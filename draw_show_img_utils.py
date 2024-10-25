from typing import Sequence
from cv2.typing import MatLike
import os

import cv2
from matplotlib import pyplot as plt
import numpy as np

""" FROM CPV LAB 1"""


def draw_quadrange(x, y, color=None, limX=None, limY=None) -> None:
    """draw a regtangle of points
    Bundle a scatter, color, xlim, ylim in one function

    Args:
        x, y (np array shape): data positoin
        color (str, optional): color of points.
        limX , limy (Iterable[int]): matplotlib Xlim Ylim
    """

    plt.scatter(x, y, color=color)

    if limX is not None and limY is not None:
        plt.xlim(limX)
        plt.ylim(limY)


""" FROM CPV LAB 2 """


def draw_2_img(im1: MatLike, im2: MatLike, label1: str = None, label2: str = None,  cmap="grey", figsize: tuple[int, int] = None, sharexy: bool = False):
    fig, axes = plt.subplots(1, 2, figsize=figsize,
                             sharex=sharexy, sharey=sharexy)

    # configuration
    axes[0].set_title(label1)
    axes[1].set_title(label2)

    axes[0].imshow(im1, cmap=cmap)
    axes[1].imshow(im2, cmap=cmap)

    return fig, axes


def cv2_imshow(img, winname=str()) -> None:
    cv2.imshow(winname, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def figure_to_img(fig):
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)


""" FROM CPV LAB 3"""


def draw_many_imgs(imgs: Sequence[MatLike], labels: Sequence[str], cmap="grey", figsize: tuple[int, int] = None, sharexy: bool = False):
    """Extension to draw_2_img"""
    assert len(imgs) == len(labels), f"Images and labels must have the same length. Got {
        len(imgs)} != {len(labels)}"

    fig, axes = plt.subplots(1, len(imgs), figsize=figsize,
                             sharex=sharexy, sharey=sharexy)

    for i in range(len(axes)):
        axes[i].set_title(labels[i])
        axes[i].imshow(imgs[i], cmap=cmap)

    return fig, axes


""" FROM CPV LAB 4"""


def print_debug(*args):
    for i, arg in enumerate(args):
        print("Vaiable", i)
        print(arg)

    print("_"*10)


""" FOR THIS IPYNB """


def get_img_path(name: str):
    return os.path.join(os.path.dirname(os.getcwd()),
                        "data_segs", f"{name}_0.png"), \
        os.path.join(os.path.dirname(os.getcwd()),
                     "data_segs", f"{name}_1.png")
