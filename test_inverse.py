import cv2
import torch
import numpy as np
import os

from cmconv import (
    convert_inverse,
    make_distorted_grid_image,
    undistort_image,
    OpenCv,
)


if __name__ == "__main__":

    out_dir = "./output/"
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_params_except_intrinsics(model):
        for k, v in model.params.items():
            # Fix the intrinsic parameters
            if k in ["fx", "fy", "cx", "cy"]:
                continue
            v.requires_grad = True

    opencv_zero = OpenCv(device, 960, 768, 487.5, 495.0, 479.0, 386.0, 0, 0, 0, 0)
    nodist = make_distorted_grid_image(
        opencv_zero, opencv_zero.width, line_color=(255, 0, 0)
    )
    cv2.imwrite(out_dir + "grid.png", nodist)
    
    # Let's try to convert the fisheye model to full OpenCV model
    opencv_full = OpenCv(device, 960, 768, 487.5, 495.0, 479.0, 386.0, -0.4,
                         0.3, 0.01, 0.01, -0.001, -0.002, 0.001, 0.001)
    set_params_except_intrinsics(opencv_full)
    print("Convert to OpenCV Full")
    distorted = make_distorted_grid_image(
        opencv_full, opencv_full.width, line_color=(255, 0, 0)
    )
    cv2.imwrite(out_dir + "distort.png", distorted)
    set_params_except_intrinsics(opencv_zero)
    convert_inverse(opencv_full, opencv_zero)
    #print(nodist.shape)
    distort_as_undistort = undistort_image(opencv_zero, nodist)
    cv2.imwrite(out_dir + "distort_as_undistort.png", distort_as_undistort)
    print()
