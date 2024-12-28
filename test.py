from camcalibconv.camcalibconv import convert, make_distorted_grid_image, OpenCv, OpenCvFisheye, UnifiedCameraModel, EnhancedUnifiedCameraModel, DoubleSphere
import cv2
import torch

import cv2
import numpy as np

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opencv = OpenCv(device, 960, 768, 487.5, 495.0, 479.0, 386.0, 0, 0, 0, 0)
    img = make_distorted_grid_image(opencv)
    cv2.imwrite('grid_opencv.png', img)

    # fisheye = OpenCvFisheye(device, 960, 768, 487.5, 495.0, 479.0, 386.0, 0, 0, 0, 0)
    fisheye = OpenCvFisheye(device, 960, 768, 487.5, 495.0, 479.0, 386.0, -0.051, -0.033, 0.02, 0.001)
    fisheye = OpenCvFisheye(device, 960, 768, 487.5, 495.0, 479.0, 386.0, -0.11, -0.06, 0.02, 0.01)

    img = make_distorted_grid_image(fisheye)
    cv2.imwrite('grid_fisheye.png', img)
    
    ucm = UnifiedCameraModel(device, 960, 768, 487.5, 495.0, 479.0, 386.0, 0)
    img = make_distorted_grid_image(ucm)
    cv2.imwrite('grid_ucm.png', img)    
    
    eucm = EnhancedUnifiedCameraModel(device, 960, 768, 487.5, 495.0, 479.0, 386.0, 0, 0)
    img = make_distorted_grid_image(eucm)
    cv2.imwrite('grid_eucm.png', img)

    ds = DoubleSphere(device, 960, 768, 487.5, 495.0, 479.0, 386.0, 0, 0)
    img = make_distorted_grid_image(ds)
    cv2.imwrite('grid_ds.png', img)

    eucm.params["alpha"].requires_grad = True
    eucm.params["beta"].requires_grad = True
    convert(fisheye, eucm)
    print(eucm.params["alpha"], eucm.params["beta"])
    tgt_img = cv2.imread('grid_fisheye.png')
    img = make_distorted_grid_image(eucm, img=tgt_img, line_color=(0, 255, 0))
    cv2.imwrite('grid_eucm_fisheye.png', img)

    ds.params["alpha"].requires_grad = True
    ds.params["xi"].requires_grad = True
    convert(fisheye, ds)
    print(ds.params["alpha"], ds.params["xi"])
    tgt_img = cv2.imread('grid_fisheye.png')
    img = make_distorted_grid_image(ds, img=tgt_img, line_color=(0, 255, 0))
    cv2.imwrite('grid_ds_fisheye.png', img)