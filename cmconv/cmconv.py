import math

import torch
import numpy as np
import cv2

CAMCALIBCONV_EPS = 1e-8


class CameraModelBase(object):
    def __init__(self, device: torch.device):
        self.device = device
        self.params = {}
        pass

    def distort(self, pts: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def undistort(self, pts: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def project(self, pts: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def limit_range(self) -> None:
        raise NotImplementedError


class OpenCv(CameraModelBase):
    def __init__(
        self,
        device: torch.device,
        width,
        height,
        fx,
        fy,
        cx,
        cy,
        k1,
        k2,
        p1,
        p2,
        k3=0,
        k4=0,
        k5=0,
        k6=0,
    ):
        super().__init__(device)
        self.params = {}
        self.width = width
        self.height = height
        self.params["fx"] = torch.tensor(fx, device=device, dtype=torch.float32)
        self.params["fy"] = torch.tensor(fy, device=device, dtype=torch.float32)
        self.params["cx"] = torch.tensor(cx, device=device, dtype=torch.float32)
        self.params["cy"] = torch.tensor(cy, device=device, dtype=torch.float32)
        self.params["k1"] = torch.tensor(k1, device=device, dtype=torch.float32)
        self.params["k2"] = torch.tensor(k2, device=device, dtype=torch.float32)
        self.params["p1"] = torch.tensor(p1, device=device, dtype=torch.float32)
        self.params["p2"] = torch.tensor(p2, device=device, dtype=torch.float32)
        self.params["k3"] = torch.tensor(k3, device=device, dtype=torch.float32)
        self.params["k4"] = torch.tensor(k4, device=device, dtype=torch.float32)
        self.params["k5"] = torch.tensor(k5, device=device, dtype=torch.float32)
        self.params["k6"] = torch.tensor(k6, device=device, dtype=torch.float32)

    def distort(self, pts: torch.Tensor) -> torch.Tensor:
        fx = self.params["fx"]
        fy = self.params["fy"]
        cx = self.params["cx"]
        cy = self.params["cy"]
        k1 = self.params["k1"]
        k2 = self.params["k2"]
        p1 = self.params["p1"]
        p2 = self.params["p2"]
        k3 = self.params["k3"]
        k4 = self.params["k4"]
        k5 = self.params["k5"]
        k6 = self.params["k6"]

        x = pts[:, 0]
        y = pts[:, 1]
        norm_cx = cx / self.width
        norm_cy = cy / self.height
        norm_fx = fx / self.width
        norm_fy = fy / self.height
        u1 = (x - norm_cx) / norm_fx
        v1 = (y - norm_cy) / norm_fy
        u2 = u1 * u1
        v2 = v1 * v1
        r2 = u2 + v2
        _2uv = 2 * u1 * v1
        kr = (1 + ((k3 * r2 + k2) * r2 + k1) * r2) / (
            1 + ((k6 * r2 + k5) * r2 + k4) * r2
        )
        dx = norm_fx * (u1 * kr + p1 * _2uv + p2 * (r2 + 2 * u2)) + norm_cx
        dy = norm_fy * (v1 * kr + p1 * (r2 + 2 * v2) + p2 * _2uv) + norm_cy
        return torch.stack([dx, dy], dim=1)

    def undistort(self, pts: torch.Tensor) -> torch.Tensor:
        fx = self.params["fx"]
        fy = self.params["fy"]
        cx = self.params["cx"]
        cy = self.params["cy"]
        k1 = self.params["k1"]
        k2 = self.params["k2"]
        p1 = self.params["p1"]
        p2 = self.params["p2"]
        k3 = self.params["k3"]
        k4 = self.params["k4"]
        k5 = self.params["k5"]
        k6 = self.params["k6"]

        x = pts[:, 0]
        y = pts[:, 1]
        norm_cx = cx / self.width
        norm_cy = cy / self.height
        norm_fx = fx / self.width
        norm_fy = fy / self.height
        x0 = (x - norm_cx) / norm_fx
        y0 = (y - norm_cy) / norm_fy
        x = x0.detach().clone()
        y = y0.detach().clone()

        # https://github.com/egonSchiele/OpenCV/blob/master/modules/imgproc/src/undistort.cpp#L345
        # https://github.com/opencv/opencv/blob/master/modules/calib3d/src/undistort.dispatch.cpp#L385

        # Compensate distortion iteratively
        max_iter = 5
        for j in range(max_iter):
            r2 = x * x + y * y
            icdist = (1.0 + ((k6 * r2 + k5) * r2 + k4) * r2) / (
                1.0 + ((k3 * r2 + k2) * r2 + k1) * r2
            )
            deltaX = 2.0 * p1 * x * y + p2 * (r2 + 2 * x * x)
            deltaY = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
            x = (x0 - deltaX) * icdist
            y = (y0 - deltaY) * icdist
        u = x * norm_fx + norm_cx
        v = y * norm_fy + norm_cy
        return torch.stack([u, v], dim=1)

    def project(self, pts: torch.Tensor) -> torch.Tensor:
        fx = self.params["fx"]
        fy = self.params["fy"]
        cx = self.params["cx"]
        cy = self.params["cy"]
        k1 = self.params["k1"]
        k2 = self.params["k2"]
        p1 = self.params["p1"]
        p2 = self.params["p2"]
        k3 = self.params["k3"]
        k4 = self.params["k4"]
        k5 = self.params["k5"]
        k6 = self.params["k6"]

        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        norm_cx = cx / self.width
        norm_cy = cy / self.height
        norm_fx = fx / self.width
        norm_fy = fy / self.height
        u1 = x / z
        v1 = y / z
        u2 = u1 * u1
        v2 = v1 * v1
        r2 = u2 + v2
        _2uv = 2 * u1 * v1
        kr = (1 + ((k3 * r2 + k2) * r2 + k1) * r2) / (
            1 + ((k6 * r2 + k5) * r2 + k4) * r2
        )
        dx = norm_fx * (u1 * kr + p1 * _2uv + p2 * (r2 + 2 * u2)) + norm_cx
        dy = norm_fy * (v1 * kr + p1 * (r2 + 2 * v2) + p2 * _2uv) + norm_cy
        return torch.stack([dx, dy], dim=1)

    def limit_range(self) -> None:
        pass


class OpenCvFisheye(CameraModelBase):
    def __init__(
        self, device: torch.device, width, height, fx, fy, cx, cy, k1, k2, k3, k4
    ):
        super().__init__(device)
        self.params = {}
        self.width = width
        self.height = height
        self.params["fx"] = torch.tensor(fx, device=device, dtype=torch.float32)
        self.params["fy"] = torch.tensor(fy, device=device, dtype=torch.float32)
        self.params["cx"] = torch.tensor(cx, device=device, dtype=torch.float32)
        self.params["cy"] = torch.tensor(cy, device=device, dtype=torch.float32)
        self.params["k1"] = torch.tensor(k1, device=device, dtype=torch.float32)
        self.params["k2"] = torch.tensor(k2, device=device, dtype=torch.float32)
        self.params["k3"] = torch.tensor(k3, device=device, dtype=torch.float32)
        self.params["k4"] = torch.tensor(k4, device=device, dtype=torch.float32)

    def distort(self, pts: torch.Tensor) -> torch.Tensor:
        fx = self.params["fx"]
        fy = self.params["fy"]
        cx = self.params["cx"]
        cy = self.params["cy"]
        k1 = self.params["k1"]
        k2 = self.params["k2"]
        k3 = self.params["k3"]
        k4 = self.params["k4"]

        # NOTE:
        # Even with no distortion (k1=k2=k3=k4=0), points are moved/distorted.
        # This behavior is consistent with cv2.fisheye.distortPoints()
        # https://github.com/opencv/opencv/blob/94bccbecc047a4e8cecbb31866849dd1e860dac5/modules/calib3d/src/fisheye.cpp#L257
        # cv2.fisheye.undistortPoints() with no distortion will recover the original points.
        x = pts[:, 0]
        y = pts[:, 1]
        norm_cx = cx / self.width
        norm_cy = cy / self.height
        norm_fx = fx / self.width
        norm_fy = fy / self.height

        u = (x - norm_cx) / norm_fx
        v = (y - norm_cy) / norm_fy

        r = torch.sqrt(u**2 + v**2)
        theta = torch.atan(r)
        rd = theta * (1 + k1 * theta**2 + k2 * theta**4 + k3 * theta**6 + k4 * theta**8)
        scale = torch.where(r > 0, rd / r, 1)
        ud = u * scale
        vd = v * scale
        dx = norm_fx * ud + norm_cx
        dy = norm_fy * vd + norm_cy
        return torch.stack([dx, dy], dim=1)

    def limit_range(self) -> None:
        pass

    def project(self, pts: torch.Tensor) -> torch.Tensor:
        fx = self.params["fx"]
        fy = self.params["fy"]
        cx = self.params["cx"]
        cy = self.params["cy"]
        k1 = self.params["k1"]
        k2 = self.params["k2"]
        k3 = self.params["k3"]
        k4 = self.params["k4"]

        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        norm_cx = cx / self.width
        norm_cy = cy / self.height
        norm_fx = fx / self.width
        norm_fy = fy / self.height

        u = x / z
        v = y / z

        r = torch.sqrt(u**2 + v**2)
        theta = torch.atan(r)
        rd = theta * (1 + k1 * theta**2 + k2 * theta**4 + k3 * theta**6 + k4 * theta**8)
        scale = torch.where(r > 0, rd / r, 1)
        ud = u * scale
        vd = v * scale
        dx = norm_fx * ud + norm_cx
        dy = norm_fy * vd + norm_cy
        return torch.stack([dx, dy], dim=1)


class UnifiedCameraModel(CameraModelBase):
    def __init__(self, device: torch.device, width, height, fx, fy, cx, cy, alpha):
        super().__init__(device)
        self.params = {}
        self.width = width
        self.height = height
        self.params["fx"] = torch.tensor(fx, device=device, dtype=torch.float32)
        self.params["fy"] = torch.tensor(fy, device=device, dtype=torch.float32)
        self.params["cx"] = torch.tensor(cx, device=device, dtype=torch.float32)
        self.params["cy"] = torch.tensor(cy, device=device, dtype=torch.float32)
        self.params["alpha"] = torch.tensor(alpha, device=device, dtype=torch.float32)

    def distort(self, pts: torch.Tensor) -> torch.Tensor:
        fx = self.params["fx"]
        fy = self.params["fy"]
        cx = self.params["cx"]
        cy = self.params["cy"]
        alpha = self.params["alpha"]

        x = pts[:, 0]
        y = pts[:, 1]
        norm_cx = cx / self.width
        norm_cy = cy / self.height
        norm_fx = fx / self.width
        norm_fy = fy / self.height
        u = (x - norm_cx) / norm_fx
        v = (y - norm_cy) / norm_fy
        z = 1.0  # Normalized depth

        d = torch.sqrt(u**2 + v**2 + z**2)
        # Avoid no-backwarding with alpha == 0
        alpha = torch.where(alpha == 0, alpha + CAMCALIBCONV_EPS, alpha)
        denom = alpha * d + (1.0 - alpha) * z

        ud = u / denom
        vd = v / denom
        dx = norm_fx * ud + norm_cx
        dy = norm_fy * vd + norm_cy
        return torch.stack([dx, dy], dim=1)

    def project(self, pts: torch.Tensor) -> torch.Tensor:
        fx = self.params["fx"]
        fy = self.params["fy"]
        cx = self.params["cx"]
        cy = self.params["cy"]
        alpha = self.params["alpha"]

        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]

        norm_cx = cx / self.width
        norm_cy = cy / self.height
        norm_fx = fx / self.width
        norm_fy = fy / self.height

        u = x / z
        v = y / z

        # d = sqrt(u^2 + v^2 + 1) is sqrt(x^2+y^2+z^2)/z
        d = torch.sqrt(u**2 + v**2 + 1)

        alpha = torch.where(alpha == 0, alpha + CAMCALIBCONV_EPS, alpha)
        denom = alpha * d + (1.0 - alpha)

        ud = u / denom
        vd = v / denom

        dx = norm_fx * ud + norm_cx
        dy = norm_fy * vd + norm_cy
        return torch.stack([dx, dy], dim=1)

    def limit_range(self) -> None:
        with torch.no_grad():
            self.params["alpha"].data.clamp(CAMCALIBCONV_EPS, 1.0)


class EnhancedUnifiedCameraModel(CameraModelBase):
    def __init__(
        self, device: torch.device, width, height, fx, fy, cx, cy, alpha, beta
    ):
        super().__init__(device)
        self.params = {}
        self.width = width
        self.height = height
        self.params["fx"] = torch.tensor(fx, device=device, dtype=torch.float32)
        self.params["fy"] = torch.tensor(fy, device=device, dtype=torch.float32)
        self.params["cx"] = torch.tensor(cx, device=device, dtype=torch.float32)
        self.params["cy"] = torch.tensor(cy, device=device, dtype=torch.float32)
        self.params["alpha"] = torch.tensor(alpha, device=device, dtype=torch.float32)
        self.params["beta"] = torch.tensor(beta, device=device, dtype=torch.float32)

    def distort(self, pts: torch.Tensor) -> torch.Tensor:
        fx = self.params["fx"]
        fy = self.params["fy"]
        cx = self.params["cx"]
        cy = self.params["cy"]
        alpha = self.params["alpha"]
        beta = self.params["beta"]

        x = pts[:, 0]
        y = pts[:, 1]
        norm_cx = cx / self.width
        norm_cy = cy / self.height
        norm_fx = fx / self.width
        norm_fy = fy / self.height
        u = (x - norm_cx) / norm_fx
        v = (y - norm_cy) / norm_fy
        z = 1.0  # Normalized depth

        d = torch.sqrt(beta * (u**2 + v**2) + z**2)
        # Avoid no-backwarding with alpha == 0
        alpha = torch.where(alpha == 0, alpha + CAMCALIBCONV_EPS, alpha)
        denom = alpha * d + (1.0 - alpha) * z

        ud = u / denom
        vd = v / denom
        dx = norm_fx * ud + norm_cx
        dy = norm_fy * vd + norm_cy
        return torch.stack([dx, dy], dim=1)

    def project(self, pts: torch.Tensor) -> torch.Tensor:
        fx = self.params["fx"]
        fy = self.params["fy"]
        cx = self.params["cx"]
        cy = self.params["cy"]
        alpha = self.params["alpha"]
        beta = self.params["beta"]

        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        norm_cx = cx / self.width
        norm_cy = cy / self.height
        norm_fx = fx / self.width
        norm_fy = fy / self.height

        u = x / z
        v = y / z

        d = torch.sqrt(beta * (u**2 + v**2) + 1)

        alpha = torch.where(alpha == 0, alpha + CAMCALIBCONV_EPS, alpha)
        denom = alpha * d + (1.0 - alpha)

        ud = u / denom
        vd = v / denom
        dx = norm_fx * ud + norm_cx
        dy = norm_fy * vd + norm_cy
        return torch.stack([dx, dy], dim=1)

    def limit_range(self) -> None:
        with torch.no_grad():
            self.params["alpha"].data.clamp(CAMCALIBCONV_EPS, 1.0)
            self.params["beta"].data.clamp(CAMCALIBCONV_EPS, 99999.9)


class DoubleSphere(CameraModelBase):
    def __init__(self, device: torch.device, width, height, fx, fy, cx, cy, xi, alpha):
        super().__init__(device)
        self.params = {}
        self.width = width
        self.height = height
        self.params["fx"] = torch.tensor(fx, device=device, dtype=torch.float32)
        self.params["fy"] = torch.tensor(fy, device=device, dtype=torch.float32)
        self.params["cx"] = torch.tensor(cx, device=device, dtype=torch.float32)
        self.params["cy"] = torch.tensor(cy, device=device, dtype=torch.float32)
        self.params["xi"] = torch.tensor(xi, device=device, dtype=torch.float32)
        self.params["alpha"] = torch.tensor(alpha, device=device, dtype=torch.float32)

    def distort(self, pts: torch.Tensor) -> torch.Tensor:
        fx = self.params["fx"]
        fy = self.params["fy"]
        cx = self.params["cx"]
        cy = self.params["cy"]
        xi = self.params["xi"]
        alpha = self.params["alpha"]

        x = pts[:, 0]
        y = pts[:, 1]
        norm_cx = cx / self.width
        norm_cy = cy / self.height
        norm_fx = fx / self.width
        norm_fy = fy / self.height
        u = (x - norm_cx) / norm_fx
        v = (y - norm_cy) / norm_fy
        z = 1.0  # Normalized depth

        d1 = torch.sqrt(u**2 + v**2 + z**2)
        d2 = torch.sqrt(u**2 + v**2 + (xi * d1 + z) ** 2)
        denom = alpha * d2 + (1 - alpha) * (xi * d1 + z)
        ud = u / denom
        vd = v / denom
        dx = norm_fx * ud + norm_cx
        dy = norm_fy * vd + norm_cy
        return torch.stack([dx, dy], dim=1)

    def project(self, pts: torch.Tensor) -> torch.Tensor:
        fx = self.params["fx"]
        fy = self.params["fy"]
        cx = self.params["cx"]
        cy = self.params["cy"]
        xi = self.params["xi"]
        alpha = self.params["alpha"]

        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]

        norm_cx = cx / self.width
        norm_cy = cy / self.height
        norm_fx = fx / self.width
        norm_fy = fy / self.height

        d1 = torch.sqrt(x**2 + y**2 + z**2)
        z_xi = xi * d1 + z
        d2 = torch.sqrt(x**2 + y**2 + z_xi**2)

        alpha = torch.where(alpha == 0, alpha + CAMCALIBCONV_EPS, alpha)
        denom = alpha * d2 + (1.0 - alpha) * z_xi

        u = x / denom
        v = y / denom

        dx = norm_fx * u + norm_cx
        dy = norm_fy * v + norm_cy
        return torch.stack([dx, dy], dim=1)

    def limit_range(self) -> None:
        pass


def make_distorted_grid_image(
    camera_model: CameraModelBase,
    width: int = 640,
    n_grids: int = 10,
    bkg_color: tuple = (0, 0, 0),
    line_color: tuple = (255, 255, 255),
    img: np.ndarray | None = None,
) -> np.ndarray:
    uniform_pts1d = torch.linspace(0, 1, n_grids, device=camera_model.device)
    grid_x, grid_y = torch.meshgrid(uniform_pts1d, uniform_pts1d, indexing="ij")
    uniform_pts2d = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    distorted_pts2d = camera_model.distort(uniform_pts2d)

    height = int(camera_model.height * width / camera_model.width)
    distorted_pts2d_np = distorted_pts2d.cpu().detach().numpy()
    distorted_pts2d_np[:, 0] *= width - 1
    distorted_pts2d_np[:, 1] *= height - 1
    grid_points = distorted_pts2d_np.astype(np.int32)

    img_size = (height, width, 3)
    if img is None:
        img = (np.ones(img_size) * bkg_color).astype(np.uint8)
    else:
        img = cv2.resize(img, (width, height)).astype(np.uint8)

    rows, cols = grid_x.size()
    line_width = int(max(1.0, max(img_size) / 200))
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            # Right
            if j < cols - 1 or (i == 0 and j == 0):
                next_idx = idx + 1
                pt1 = tuple(grid_points[idx])
                pt2 = tuple(grid_points[next_idx])
                cv2.line(img, pt1, pt2, line_color, line_width)
            # Down
            if i < rows - 1 or (i == 0 and j == 0):
                next_idx = idx + cols
                pt1 = tuple(grid_points[idx])
                pt2 = tuple(grid_points[next_idx])
                cv2.line(img, pt1, pt2, line_color, line_width)

    radius = int(line_width * 2.5)
    for pt in grid_points:
        cv2.circle(img, tuple(pt), radius, line_color, -1)

    return img


def visualize_projected_points(src_projected, dst_projected,
                               width, height, skip=1):
    src_projected[..., 0] *= width - 1
    src_projected[..., 1] *= height - 1
    dst_projected[..., 0] *= width - 1
    dst_projected[..., 1] *= height - 1
    src_projected = src_projected.cpu().detach().numpy()
    dst_projected = dst_projected.cpu().detach().numpy()

    vis = np.zeros((height, width, 3), dtype=np.uint8)
    for src_pt, dst_pt in zip(src_projected[::skip], dst_projected[::skip]):
        pt1 = tuple(src_pt.astype(np.int32))
        pt2 = tuple(dst_pt.astype(np.int32))
        cv2.circle(vis, pt1, 2, (0, 0, 255), -1)
        cv2.circle(vis, pt2, 2, (255, 0, 0), -1)
    return vis


def undistort_image(model: CameraModelBase, img: np.ndarray) -> np.ndarray:
    height, width = img.shape[:2]

    # Create a grid of normalized pixel coordinates (x, y)
    yv, xv = torch.meshgrid(
        torch.linspace(0, 1, height, device=model.device),
        torch.linspace(0, 1, width, device=model.device),
        indexing="ij",
    )
    pixel_coords = torch.stack([xv, yv], dim=-1).reshape(-1, 2)  # Shape: (H*W, 2)

    # Apply DISTORTION
    distorted_coords = model.distort(pixel_coords)

    # Convert back to numpy
    distorted_coords = distorted_coords.cpu().detach().numpy().reshape(height, width, 2)

    # Use cv2.remap to remap the image using undistorted coordinates
    distorted_coords_x = distorted_coords[..., 0] * (width - 1)
    distorted_coords_y = distorted_coords[..., 1] * (height - 1)

    undistorted_img = cv2.remap(
        img.astype(np.uint8),
        distorted_coords_x.astype(np.float32),
        distorted_coords_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    return undistorted_img.astype(np.uint8)


def convert(
    src: CameraModelBase,
    dst: CameraModelBase,
    th_iter: int = 2000,
    th_loss: float = 1e-8,
    n_grids: int = 512,
    lr: float = 0.01,
    x_weight: float | torch.Tensor = 1.0,
    y_weight: float | torch.Tensor = 1.0,
) -> None:
    # Setup optimizer
    opt_params = []
    for k, v in dst.params.items():
        if v.requires_grad:
            print("Optimize", k, v.item())
            opt_params.append(v)
    num_opt_params = len(opt_params)
    print("Number of optimized parameters:", num_opt_params)
    if num_opt_params < 1:
        print("Please set requires_grad=True for at least one parameter in dst.")
        return
    optimizer = torch.optim.Adam(opt_params, lr=lr)

    # Prepare grid points
    uniform_pts1d = torch.linspace(0, 1, n_grids, device=src.device)
    grid_x, grid_y = torch.meshgrid(uniform_pts1d, uniform_pts1d, indexing="ij")
    uniform_pts2d = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

    # Source distorted grid points are fixed
    src_distorted = src.distort(uniform_pts2d).detach()

    for i in range(th_iter):
        optimizer.zero_grad()
        # Apply distortion with the current parameters
        dst_distorted = dst.distort(uniform_pts2d)

        # Take the pixel-wise L2 loss
        loss_pixwise = (src_distorted - dst_distorted).pow(2)

        # Apply weights to x and y directions
        loss_pixwise[:, 0] *= x_weight
        loss_pixwise[:, 1] *= y_weight

        # Compute the mean and update the parameters
        loss = torch.mean(loss_pixwise)
        loss.backward()
        optimizer.step()

        # Limit the range of the parameters
        dst.limit_range()

        if loss.item() < th_loss:
            break
        if i % 100 == 0:
            print("Iter:", i, "Loss:", loss.item())
    print("Iter:", i, "Loss:", loss.item())
    for k, v in dst.params.items():
        if v.requires_grad:
            print("Optimized", k, v.item())


def convert_inverse(
    src: CameraModelBase,
    dst: CameraModelBase,
    th_iter: int = 4000,
    th_loss: float = 1e-8,
    n_grids: int = 1024,
    lr: float = 0.01,
    x_weight: float | torch.Tensor = 1.0,
    y_weight: float | torch.Tensor = 1.0,
) -> None:
    # Setup optimizer
    opt_params = []
    for k, v in dst.params.items():
        if v.requires_grad:
            print("Optimize", k, v.item())
            opt_params.append(v)

    # Prepare grid points
    uniform_pts1d = torch.linspace(0, 1, n_grids, device=dst.device)
    grid_x, grid_y = torch.meshgrid(uniform_pts1d, uniform_pts1d, indexing="ij")
    uniform_pts2d = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

    # Source distorted grid points are fixed
    undistorted = src.undistort(uniform_pts2d).detach()
    optimizer = torch.optim.Adam(opt_params, lr=lr)
    num_opt_params = len(opt_params)
    print("Number of optimized parameters:", num_opt_params)
    if num_opt_params < 1:
        print("Please set requires_grad=True for at least one parameter in dst.")
        return

    for i in range(th_iter):
        optimizer.zero_grad()
        # Apply distortion with the current parameters
        dst_distorted = dst.distort(uniform_pts2d)

        # Take the pixel-wise L2 loss
        loss_pixwise = (undistorted - dst_distorted).pow(2)

        # Apply weights to x and y directions
        loss_pixwise[:, 0] *= x_weight
        loss_pixwise[:, 1] *= y_weight

        # Compute the mean and update the parameters
        loss = torch.mean(loss_pixwise)
        loss.backward()
        optimizer.step()

        # Limit the range of the parameters
        dst.limit_range()

        if loss.item() < th_loss:
            break
        if i % 100 == 0:
            print("Iter:", i, "Loss:", loss.item())
    print("Iter:", i, "Loss:", loss.item())
    for k, v in dst.params.items():
        if v.requires_grad:
            print("Optimized", k, v.item())


def warp_to_frustum(n_grids, near, far, fovx_deg, device):
    uniform_pts1d = torch.linspace(0, 1, n_grids, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(
        uniform_pts1d, uniform_pts1d, uniform_pts1d, indexing="ij"
    )

    # liearly interpolate z
    z = near + grid_z * (far - near)

    fov_rad = math.radians(fovx_deg)
    tan_half_fov = torch.tan(torch.tensor(fov_rad / 2.0, device=device))

    # Since grid_x and grid_y take [0,1], shift to [-0.5, 0.5] to center 0, then scale to [-1,1].
    # The width according to depth z is z * tan_half_fov, so multiply in each direction
    x = (grid_x - 0.5) * 2 * z * tan_half_fov
    y = (grid_y - 0.5) * 2 * z * tan_half_fov

    return x, y, z


def convert3d(
    src: CameraModelBase,
    dst: CameraModelBase,
    th_iter: int = 5000,
    th_loss: float = 1e-8,
    n_grids: int = 128,
    near: float = 0.1,
    far: float = 10.0,
    approx_fovx_deg: float = 130.0,
    lr: float = 0.001,
) -> None:
    # Setup optimizer
    opt_params = []
    for k, v in dst.params.items():
        if v.requires_grad:
            print("Optimize", k, v.item())
            opt_params.append(v)
    num_opt_params = len(opt_params)
    print("Number of optimized parameters:", num_opt_params)
    if num_opt_params < 1:
        print("Please set requires_grad=True for at least one parameter in dst.")
        return
    optimizer = torch.optim.Adam(opt_params, lr=lr)

    if approx_fovx_deg < 0:
        # Prepare grid points
        uniform_pts1d = torch.linspace(0, 1, n_grids, device=src.device)
        grid_x, grid_y, grid_z = torch.meshgrid(
            uniform_pts1d, uniform_pts1d, uniform_pts1d, indexing="ij"
        )
        xy_min = -far / 2
        # -far / 2 ~ far / 2
        grid_x = grid_x * far + xy_min
        grid_y = grid_y * far + xy_min
        grid_z = grid_z * (far - near) + near
    else:
        approx_fovx_deg = min(max(approx_fovx_deg, 90), 179)
        grid_x, grid_y, grid_z = warp_to_frustum(
            n_grids, near, far, approx_fovx_deg, src.device
        )
    uniform_pts3d = torch.stack(
        [grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1
    )

    # Source distorted grid points are fixed
    src_projected = src.project(uniform_pts3d).detach()
    for i in range(th_iter):
        optimizer.zero_grad()
        # Apply distortion with the current parameters
        dst_projected = dst.project(uniform_pts3d)

        # Take the pixel-wise L2 loss
        loss_pixwise = (src_projected - dst_projected).pow(2)

        # Compute the mean and update the parameters
        loss = torch.mean(loss_pixwise)
        loss.backward()
        optimizer.step()

        # Limit the range of the parameters
        dst.limit_range()

        if loss.item() < th_loss:
            break
        if i % 100 == 0:
            print("Iter:", i, "Loss:", loss.item())
    print("Iter:", i, "Loss:", loss.item())
    for k, v in dst.params.items():
        if v.requires_grad:
            print("Optimized", k, v.item())

    return src_projected.detach(), dst_projected.detach()
