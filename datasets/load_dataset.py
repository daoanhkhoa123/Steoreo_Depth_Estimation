from typing import TypedDict
from numpy.typing import NDArray

import os
import warnings
import cv2
import numpy as np


def pmf_reader(path) -> tuple[NDArray, float]:
    with open(path, 'rb') as f:
        header = f.readline().decode().strip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dimensions = f.readline().decode().strip()
        width, height = map(int, dimensions.split())
        scale = float(f.readline().decode().strip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'

        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)

        return np.flipud(data.reshape(shape)), scale


class CallibTyping(TypedDict):
    """" 
    cam0,1:        camera matrices for the rectified views, in the form [f 0 cx; 0 f cy; 0 0 1], where
    f:           focal length in pixels
    cx, cy:      principal point  (note that cx differs between view 0 and 1)

    doffs:         x-difference of principal points, doffs = cx1 - cx0

    baseline:      camera baseline in mm

    width, height: image size

    ndisp:         a conservative bound on the number of disparity levels;
                the stereo algorithm MAY utilize this bound and search from d = 0 .. ndisp-1

    isint:         whether the GT disparites only have integer precision (true for the older datasets;
                in this case submitted floating-point disparities are rounded to ints before evaluating)

    vmin, vmax:    a tight bound on minimum and maximum disparities, used for color visualization;
                the stereo algorithm MAY NOT utilize this information

    dyavg, dymax:  average and maximum absolute y-disparities, providing an indication of
                the calibration error present in the imperfect datasets.
    """
    cam0: NDArray
    cam1: NDArray
    focal: float
    doffs: float
    baseline: float
    width: float
    height: float
    ndisp: float
    isint: float
    vmin: float
    vmax: float
    dyavg: float
    dymax: float


class DataSetLoader:
    def __init__(self) -> None:
        pass


class Data:
    def __init__(self, index=0, flags=cv2.IMREAD_UNCHANGED) -> None:
        self._left_path = os.path.join("datasets", f"img{index}_0.png")
        self._right_path = os.path.join("datasets", f"img{index}_1.png")
        self.img1 = cv2.imread(self._left_path, flags=flags)
        self.img2 = cv2.imread(self._right_path, flags=flags)

        try:
            self._calib_path = os.path.join("datasets", f"calib{index}.txt")
            self._groundtruthL_path = os.path.join(
                "datasets", f"disp{index}GT.pfm")

            self.calib: CallibTyping = self._get_calib()
            self.GTruth = pmf_reader(
                self._groundtruthL_path)[0] + self.calib["doffs"]

        except FileNotFoundError:
            warnings.warn(
                "This Dataset does not have calibration or ground truth to evaluate", UserWarning)

    def _get_calib(self):
        calib_data = dict()
        with open(self._calib_path, 'r') as file:
            for line in file:
                # Remove whitespace and split on '='
                line = line.strip()
                if line:  # Skip empty lines
                    key, value = line.split('=')

                    # Handle matrix data (cam0 and cam1)
                    if key in ['cam0', 'cam1']:
                        # Remove brackets and split the matrix data
                        matrix_str = value.replace('[', '').replace(']', '')
                        # Split rows by semicolon
                        rows = matrix_str.split(';')
                        # Convert string numbers to float and create matrix
                        matrix = [[float(num) for num in row.split()]
                                  for row in rows]
                        calib_data[key] = np.asarray(matrix)

                        calib_data["focal"] = matrix[0][0]

                    # Handle scalar values
                    else:
                        # Try to convert to float
                        calib_data[key] = float(value)

        return calib_data

    def normalized_gtruth(self) -> NDArray:
        gt = self.GTruth
        gt[np.isposinf(gt)] = self.calib["ndisp"]
        return gt


if __name__ == "__main__":
    import cv2
    for i in range(2, 6):
        data = Data(i)
        # print(data.calib)
        print(data.GTruth.shape == data.img1.shape[:2])
        # print(data.img1.shape)
