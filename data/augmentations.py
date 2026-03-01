"""
SpikeGuard — Physics-Inspired Tampering Augmentations (Training Set)

Differentiable PyTorch transforms simulating real-world camera-trap tampering:
- Dust/tape/vegetation occlusion
- IR glare from poacher lights
- Rain/motion blur
- Combined attacks with intensity control

All transforms operate on batched tensors (B, C, H, W).
"""
import torch
import torch.nn.functional as F
import random
import math


class DustOcclusion:
    """
    Random rectangular occlusion masks simulating dust, tape, or vegetation
    covering the camera lens. Fills with brownish/greenish colors.
    """

    def __init__(self, max_area_ratio=0.5, min_area_ratio=0.05):
        self.max_area_ratio = max_area_ratio
        self.min_area_ratio = min_area_ratio

    def __call__(self, x):
        B, C, H, W = x.shape
        result = x.clone()
        for i in range(B):
            area_ratio = random.uniform(self.min_area_ratio, self.max_area_ratio)
            h_occ = int(H * math.sqrt(area_ratio) * random.uniform(0.5, 1.5))
            w_occ = int(W * math.sqrt(area_ratio) * random.uniform(0.5, 1.5))
            h_occ = min(max(h_occ, 2), H - 1)
            w_occ = min(max(w_occ, 2), W - 1)
            top = random.randint(0, H - h_occ)
            left = random.randint(0, W - w_occ)

            # Brownish/greenish fill (simulating vegetation or dust)
            fill_r = random.uniform(0.2, 0.5)
            fill_g = random.uniform(0.25, 0.45)
            fill_b = random.uniform(0.1, 0.3)
            fill = torch.tensor([fill_r, fill_g, fill_b],
                                device=x.device).view(3, 1, 1)

            # Soft edge blending
            mask = torch.ones(1, h_occ, w_occ, device=x.device)
            edge = max(1, min(h_occ, w_occ) // 4)
            if edge > 1:
                mask = F.avg_pool2d(
                    mask.unsqueeze(0), kernel_size=edge,
                    stride=1, padding=edge // 2,
                ).squeeze(0)[:, :h_occ, :w_occ]

            result[i, :, top:top + h_occ, left:left + w_occ] = (
                result[i, :, top:top + h_occ, left:left + w_occ] * (1 - mask) +
                fill * mask
            )
        return result


class IRGlare:
    """
    Gaussian brightness flooding + saturation simulating IR glare
    from torches or headlights pointed at the camera.
    """

    def __init__(self, intensity_range=(0.5, 2.5), sigma_range=(0.15, 0.5)):
        self.intensity_range = intensity_range
        self.sigma_range = sigma_range

    def __call__(self, x):
        B, C, H, W = x.shape
        device = x.device

        yy = torch.linspace(-1, 1, H, device=device)
        xx = torch.linspace(-1, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")

        result = x.clone()
        for i in range(B):
            cy = random.uniform(-0.7, 0.7)
            cx = random.uniform(-0.7, 0.7)
            sigma = random.uniform(*self.sigma_range)
            intensity = random.uniform(*self.intensity_range)

            # IR-like warm glow (reddish-white)
            gaussian = torch.exp(
                -((grid_y - cy) ** 2 + (grid_x - cx) ** 2) / (2 * sigma ** 2)
            )
            # Weight channels: more red, less blue (IR characteristics)
            r_weight = intensity * random.uniform(1.0, 1.5)
            g_weight = intensity * random.uniform(0.8, 1.2)
            b_weight = intensity * random.uniform(0.4, 0.8)
            weights = torch.tensor([r_weight, g_weight, b_weight],
                                   device=device).view(3, 1, 1)

            result[i] = result[i] + weights * gaussian.unsqueeze(0)

        return result


class MotionRainBlur:
    """
    Directional motion blur simulating rain, dust, or camera vibration.
    Uses a 1D convolution kernel at a random angle.
    """

    def __init__(self, kernel_range=(3, 11)):
        self.kernel_range = kernel_range

    def __call__(self, x):
        B, C, H, W = x.shape
        k_size = random.choice(range(self.kernel_range[0],
                                     self.kernel_range[1] + 1, 2))
        angle = random.uniform(0, math.pi)

        kernel = torch.zeros(k_size, k_size, device=x.device)
        center = k_size // 2
        for j in range(k_size):
            y = int(center + (j - center) * math.sin(angle))
            xc = int(center + (j - center) * math.cos(angle))
            if 0 <= y < k_size and 0 <= xc < k_size:
                kernel[y, xc] = 1.0

        kernel = kernel / (kernel.sum() + 1e-8)
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)

        padding = k_size // 2
        return F.conv2d(x, kernel, padding=padding, groups=C)


class SensorNoise:
    """
    Gaussian + salt-and-pepper noise simulating sensor degradation
    in extreme weather (rain, humidity, electrical interference).
    """

    def __init__(self, gaussian_std_range=(0.05, 0.3), salt_pepper_ratio=0.02):
        self.gaussian_std_range = gaussian_std_range
        self.salt_pepper_ratio = salt_pepper_ratio

    def __call__(self, x):
        std = random.uniform(*self.gaussian_std_range)
        noise = torch.randn_like(x) * std
        result = x + noise

        # Salt and pepper
        sp_mask = torch.rand_like(x[:, :1, :, :])
        result[sp_mask.expand_as(x) < self.salt_pepper_ratio / 2] = 1.0
        result[sp_mask.expand_as(x) > 1 - self.salt_pepper_ratio / 2] = 0.0

        return result


class FoliageCoverage:
    """
    Simulates partial foliage/vegetation covering the camera lens.
    Creates irregular green-tinted patches with soft edges.
    """

    def __init__(self, num_patches_range=(1, 4), max_size_ratio=0.25):
        self.num_patches_range = num_patches_range
        self.max_size_ratio = max_size_ratio

    def __call__(self, x):
        B, C, H, W = x.shape
        result = x.clone()
        for i in range(B):
            n_patches = random.randint(*self.num_patches_range)
            for _ in range(n_patches):
                ph = random.randint(4, max(5, int(H * self.max_size_ratio)))
                pw = random.randint(4, max(5, int(W * self.max_size_ratio)))
                top = random.randint(0, H - ph)
                left = random.randint(0, W - pw)

                # Green-tinted with variation
                green = torch.tensor([
                    random.uniform(0.05, 0.2),
                    random.uniform(0.3, 0.6),
                    random.uniform(0.05, 0.15),
                ], device=x.device).view(3, 1, 1)

                alpha = random.uniform(0.4, 0.85)
                result[i, :, top:top + ph, left:left + pw] = (
                    (1 - alpha) * result[i, :, top:top + ph, left:left + pw] +
                    alpha * green
                )
        return result


class TamperCompose:
    """
    Randomly applies 1 to max_transforms from the pool during training.
    Configurable intensity via attack_intensity parameter.
    """

    def __init__(self, transforms=None, max_transforms=3):
        if transforms is None:
            transforms = [
                DustOcclusion(),
                IRGlare(),
                MotionRainBlur(),
                SensorNoise(),
                FoliageCoverage(),
            ]
        self.transforms = transforms
        self.max_transforms = max_transforms

    def __call__(self, x):
        n = random.randint(1, min(self.max_transforms, len(self.transforms)))
        selected = random.sample(self.transforms, n)
        for t in selected:
            x = t(x)
        return x


def get_tamper_pipeline(max_transforms=3):
    """Returns default tampering pipeline for training."""
    return TamperCompose(max_transforms=max_transforms)
