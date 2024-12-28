import torch
import numpy as np
import cv2


class CameraModelBase(object):
    def __init__(self, device: torch.device):
        self.device = device
        self.params = {}
        pass

    def distort(self, pts: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class OpenCv(CameraModelBase):
    def __init__(self, device: torch.device, width, height, fx, fy, cx, cy, k1, k2, p1, p2,
                 k3=0, k4=0, k5=0, k6=0):
        super().__init__(device)
        self.params = {}
        self.width = width
        self.height = height
        self.params['fx'] = torch.tensor(fx, device=device, dtype=torch.float64)
        self.params['fy'] = torch.tensor(fy, device=device, dtype=torch.float64)
        self.params['cx'] = torch.tensor(cx, device=device, dtype=torch.float64)
        self.params['cy'] = torch.tensor(cy, device=device, dtype=torch.float64)
        self.params['k1'] = torch.tensor(k1, device=device, dtype=torch.float64)
        self.params['k2'] = torch.tensor(k2, device=device, dtype=torch.float64)
        self.params['p1'] = torch.tensor(p1, device=device, dtype=torch.float64)
        self.params['p2'] = torch.tensor(p2, device=device, dtype=torch.float64)
        self.params['k3'] = torch.tensor(k3, device=device, dtype=torch.float64)
        self.params['k4'] = torch.tensor(k4, device=device, dtype=torch.float64)
        self.params['k5'] = torch.tensor(k5, device=device, dtype=torch.float64)
        self.params['k6'] = torch.tensor(k6, device=device, dtype=torch.float64)

    def distort(self, pts: torch.Tensor) -> torch.Tensor:
        fx = self.params['fx']
        fy = self.params['fy']
        cx = self.params['cx']
        cy = self.params['cy']
        k1 = self.params['k1']
        k2 = self.params['k2']
        p1 = self.params['p1']
        p2 = self.params['p2']
        k3 = self.params['k3']
        k4 = self.params['k4']
        k5 = self.params['k5']
        k6 = self.params['k6']

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
        kr = (1 + ((k3 * r2 + k2) * r2 + k1) * r2) /\
             (1 + ((k6 * r2 + k5) * r2 + k4) * r2)
        dx = norm_fx * (u1 * kr + p1 * _2uv + p2 * (r2 + 2 * u2)) + norm_cx
        dy = norm_fy * (v1 * kr + p1 * (r2 + 2 * v2) + p2 * _2uv) + norm_cy
        return torch.stack([dx, dy], dim=1)


class OpenCvFisheye(CameraModelBase):
    def __init__(self, device: torch.device, width, height, fx, fy, cx, cy, k1, k2, k3, k4):
        super().__init__(device)
        self.params = {}
        self.width = width
        self.height = height
        self.params['fx'] = torch.tensor(fx, device=device, dtype=torch.float64)
        self.params['fy'] = torch.tensor(fy, device=device, dtype=torch.float64)
        self.params['cx'] = torch.tensor(cx, device=device, dtype=torch.float64)
        self.params['cy'] = torch.tensor(cy, device=device, dtype=torch.float64)
        self.params['k1'] = torch.tensor(k1, device=device, dtype=torch.float64)
        self.params['k2'] = torch.tensor(k2, device=device, dtype=torch.float64)
        self.params['k3'] = torch.tensor(k3, device=device, dtype=torch.float64)
        self.params['k4'] = torch.tensor(k4, device=device, dtype=torch.float64)

    def distort(self, pts: torch.Tensor) -> torch.Tensor:
        fx = self.params['fx']
        fy = self.params['fy']
        cx = self.params['cx']
        cy = self.params['cy']
        k1 = self.params['k1']
        k2 = self.params['k2']
        k3 = self.params['k3']
        k4 = self.params['k4']

        # NOTE:
        # Even with no distortion (k1=k2=k3=k4=0), points are moved/distorted.
        # This behavior is consistent with cv2.fisheye.distortPoints()
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


class UnifiedCameraModel(CameraModelBase):
    def __init__(self, device: torch.device, width, height, fx, fy, cx, cy, alpha):
        super().__init__(device)
        self.params = {}
        self.width = width
        self.height = height
        self.params['fx'] = torch.tensor(fx, device=device, dtype=torch.float64)
        self.params['fy'] = torch.tensor(fy, device=device, dtype=torch.float64)
        self.params['cx'] = torch.tensor(cx, device=device, dtype=torch.float64)
        self.params['cy'] = torch.tensor(cy, device=device, dtype=torch.float64)
        self.params['alpha'] = torch.tensor(alpha, device=device, dtype=torch.float64)

    def distort(self, pts: torch.Tensor) -> torch.Tensor:  
        fx = self.params['fx']
        fy = self.params['fy']
        cx = self.params['cx']
        cy = self.params['cy']
        alpha = self.params['alpha']

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
        epsilon = 1e-8  # Avoid no-backwarding with alpha == 0
        denom = (alpha + epsilon) * d + (1.0 - alpha) * z

        ud = u / denom
        vd = v / denom
        dx = norm_fx * ud + norm_cx
        dy = norm_fy * vd + norm_cy
        return torch.stack([dx, dy], dim=1)


class EnhancedUnifiedCameraModel(CameraModelBase):
    def __init__(self, device: torch.device, width, height, fx, fy, cx, cy, alpha, beta):
        super().__init__(device)
        self.params = {}
        self.width = width
        self.height = height
        self.params['fx'] = torch.tensor(fx, device=device, dtype=torch.float64)
        self.params['fy'] = torch.tensor(fy, device=device, dtype=torch.float64)
        self.params['cx'] = torch.tensor(cx, device=device, dtype=torch.float64)
        self.params['cy'] = torch.tensor(cy, device=device, dtype=torch.float64)
        self.params['alpha'] = torch.tensor(alpha, device=device, dtype=torch.float64)
        self.params['beta'] = torch.tensor(beta, device=device, dtype=torch.float64)

    def distort(self, pts: torch.Tensor) -> torch.Tensor:  
        fx = self.params['fx']
        fy = self.params['fy']
        cx = self.params['cx']
        cy = self.params['cy']
        alpha = self.params['alpha']
        beta = self.params['beta']

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
        epsilon = 1e-8  # Avoid no-backwarding with alpha == 0
        denom = (alpha + epsilon) * d + (1.0 - alpha) * z

        ud = u / denom
        vd = v / denom
        dx = norm_fx * ud + norm_cx
        dy = norm_fy * vd + norm_cy
        return torch.stack([dx, dy], dim=1)


class DoubleSphere(CameraModelBase):
    def __init__(self, device: torch.device, width, height, fx, fy, cx, cy, xi, alpha):
        super().__init__(device)
        self.params = {}
        self.width = width
        self.height = height
        self.params['fx'] = torch.tensor(fx, device=device, dtype=torch.float64)
        self.params['fy'] = torch.tensor(fy, device=device, dtype=torch.float64)
        self.params['cx'] = torch.tensor(cx, device=device, dtype=torch.float64)
        self.params['cy'] = torch.tensor(cy, device=device, dtype=torch.float64)
        self.params['xi'] = torch.tensor(xi, device=device, dtype=torch.float64)
        self.params['alpha'] = torch.tensor(alpha, device=device, dtype=torch.float64)

    def distort(self, pts: torch.Tensor) -> torch.Tensor:
        fx = self.params['fx']
        fy = self.params['fy']
        cx = self.params['cx']
        cy = self.params['cy']
        xi = self.params['xi']
        alpha = self.params['alpha']

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
        d2 = torch.sqrt(u**2 + v**2 + (xi * d1 + z)**2)
        denom = alpha * d2 + (1 - alpha) * (xi * d1 + z)
        ud = u / denom
        vd = v / denom
        dx = norm_fx * ud + norm_cx
        dy = norm_fy * vd + norm_cy
        return torch.stack([dx, dy], dim=1)    


def make_distorted_grid_image(camera_model: CameraModelBase, width: int = 640,
                              n_grids: int = 10,
                              bkg_color: tuple = (0, 0, 0),
                              line_color: tuple = (255, 255, 255),
                              img: np.ndarray | None = None):
    uniform_pts1d = torch.linspace(0, 1, n_grids, device=camera_model.device)
    grid_x, grid_y = torch.meshgrid(uniform_pts1d, uniform_pts1d, indexing='ij')
    uniform_pts2d = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    torch.set_anomaly_enabled(True, True)
    torch.autograd.set_detect_anomaly(True)
    distorted_pts2d = camera_model.distort(uniform_pts2d)

    height = int(camera_model.height * width / camera_model.width)
    distorted_pts2d_np = distorted_pts2d.cpu().detach().numpy()
    distorted_pts2d_np[:, 0] *= width
    distorted_pts2d_np[:, 1] *= height
    grid_points = distorted_pts2d_np.astype(np.int32)

    img_size = (height, width, 3)
    if img is None:
        img = np.ones(img_size, dtype=np.uint8) * bkg_color

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


def convert(src: CameraModelBase, dst: CameraModelBase,
            th_iter=2000, th_loss=1e-8,
            n_grids=1000, lr=0.01):

    uniform_pts1d = torch.linspace(0, 1, n_grids, device=src.device)
    grid_x, grid_y = torch.meshgrid(uniform_pts1d, uniform_pts1d, indexing='ij')
    uniform_pts2d = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

    src_distorted = src.distort(uniform_pts2d).detach()
    opt_params = []
    for k, v in dst.params.items():
        if v.requires_grad:
            print('Optimize', k)
            opt_params.append(v)
    num_opt_params = len(opt_params)
    print('Number of optimized parameters:', num_opt_params)
    if num_opt_params == 0:
        print('Please set requires_grad=True for at least one parameter in dst.')
        return
    optimizer = torch.optim.Adam(opt_params, lr=lr)
    for i in range(th_iter):
        optimizer.zero_grad()
        dst_distorted = dst.distort(uniform_pts2d)
        loss = torch.mean((src_distorted - dst_distorted).pow(2))
        loss.backward()
        optimizer.step()
        if loss.item() < th_loss:
            break
        if i % 100 == 0:
            print('Iter:', i, 'Loss:', loss.item())
    print('Iter:', i, 'Loss:', loss.item())