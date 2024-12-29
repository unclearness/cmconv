import cv2
import torch
import numpy as np
import os

from cmconv import (
    convert,
    make_distorted_grid_image,
    undistort_image,
    OpenCv,
    OpenCvFisheye,
    UnifiedCameraModel,
    EnhancedUnifiedCameraModel,
    DoubleSphere,
)


if __name__ == "__main__":

    out_dir = "./output/"
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # OpenCV FishEye model is given by:
    fisheye = OpenCvFisheye(
        device, 960, 768, 487.5, 495.0, 479.0, 386.0, -0.11, -0.06, 0.02, 0.01
    )
    fisheye_grid_img = make_distorted_grid_image(fisheye, fisheye.width)
    cv2.imwrite(out_dir + "grid_fisheye.png", fisheye_grid_img)

    # Undistort the fisheye image
    img = cv2.imread(out_dir + "grid_fisheye.png")
    K = np.array(
        [
            [float(fisheye.params["fx"]), 0, float(fisheye.params["cx"])],
            [0, float(fisheye.params["fy"]), float(fisheye.params["cy"])],
            [0, 0, 1],
        ]
    )
    D = np.array(
        [
            float(fisheye.params["k1"]),
            float(fisheye.params["k2"]),
            float(fisheye.params["k3"]),
            float(fisheye.params["k4"]),
        ]
    )
    undistorted = cv2.fisheye.undistortImage(img, K, D, Knew=K)
    cv2.imwrite(out_dir + "grid_fisheye_undistorted.png", undistorted)

    def set_params_except_intrinsics(model):
        for k, v in model.params.items():
            # Fix the intrinsic parameters
            if k in ["fx", "fy", "cx", "cy"]:
                continue
            v.requires_grad = True

    # Let's try to convert the fisheye model to full OpenCV model
    opencv_full = OpenCv(device, 960, 768, 487.5, 495.0, 479.0, 386.0, 0, 0, 0, 0)
    set_params_except_intrinsics(opencv_full)
    print("Convert to OpenCV Full")
    convert(fisheye, opencv_full)
    img = make_distorted_grid_image(
        opencv_full, fisheye.width, img=fisheye_grid_img, line_color=(255, 0, 0)
    )
    cv2.imwrite(out_dir + "grid_opencvfull_approx.png", img)
    img = undistort_image(opencv_full, img)
    cv2.imwrite(out_dir + "grid_opencvfull_approx_undistorted.png", img)
    print()

    # Let's try to convert the fisheye model to OpenCV model with less parameters
    opencv = OpenCv(device, 960, 768, 487.5, 495.0, 479.0, 386.0, 0, 0, 0, 0)
    set_params_except_intrinsics(opencv)
    # Disable higher order distortion coefficients
    opencv.params["k3"].requires_grad = False
    opencv.params["k4"].requires_grad = False
    opencv.params["k5"].requires_grad = False
    opencv.params["k6"].requires_grad = False
    print("Convert to OpenCV")
    convert(fisheye, opencv)
    img = make_distorted_grid_image(
        opencv, fisheye.width, img=fisheye_grid_img, line_color=(255, 120, 120)
    )
    cv2.imwrite(out_dir + "grid_opencv_approx.png", img)
    img = undistort_image(opencv, img)
    cv2.imwrite(out_dir + "grid_opencv_approx_undistorted.png", img)
    print()

    # Let's try to convert the fisheye model to Unified Camera Model
    ucm = UnifiedCameraModel(device, 960, 768, 487.5, 495.0, 479.0, 386.0, 0)
    set_params_except_intrinsics(ucm)
    print("Convert to UCM")
    convert(fisheye, ucm)
    img = make_distorted_grid_image(
        ucm, fisheye.width, img=fisheye_grid_img, line_color=(0, 255, 0)
    )
    cv2.imwrite(out_dir + "grid_ucm_approx.png", img)
    img = undistort_image(ucm, img)
    cv2.imwrite(out_dir + "grid_ucm_approx_undistorted.png", img)
    print()

    # Let's try to convert the fisheye model to Enhanced Unified Camera Model
    eucm = EnhancedUnifiedCameraModel(
        device, 960, 768, 487.5, 495.0, 479.0, 386.0, 0, 0
    )
    set_params_except_intrinsics(eucm)
    print("Convert to EUCM")
    convert(fisheye, eucm)
    img = make_distorted_grid_image(
        eucm, fisheye.width, img=fisheye_grid_img, line_color=(0, 0, 255)
    )
    cv2.imwrite(out_dir + "grid_eucm_approx.png", img)
    img = undistort_image(eucm, img)
    cv2.imwrite(out_dir + "grid_eucm_approx_undistorted.png", img)
    print()

    # Let's try to convert the fisheye model to Double Sphere model
    ds = DoubleSphere(device, 960, 768, 487.5, 495.0, 479.0, 386.0, 0, 0)
    set_params_except_intrinsics(ds)
    print("Convert to Double Sphere")
    convert(fisheye, ds)
    img = make_distorted_grid_image(
        ds, fisheye.width, img=fisheye_grid_img, line_color=(255, 255, 0)
    )
    cv2.imwrite(out_dir + "grid_ds_approx.png", img)
    img = undistort_image(ds, img)
    cv2.imwrite(out_dir + "grid_ds_approx_undistorted.png", img)
    print()
