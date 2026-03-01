"""
SpikeGuard — Unseen Tampering Test Set Generator

Creates a held-out test set with DIFFERENT tampering transforms than training.
Used for honest robustness evaluation — no self-augmented test cheating.

These transforms are intentionally distinct from augmentations.py:
- Textured dust overlays (not solid rectangles)
- Angled tape patches with lighting
- IR glare with lens flare artifacts
- Partial fog/condensation
- Realistic camera-shake blur (not simple directional)
"""
import torch
import torch.nn.functional as F
import random
import math
import numpy as np


class TexturedDustOverlay:
    """
    Perlin-noise-like dust texture overlay, unlike the solid rectangles
    used in training augmentations.
    """

    def __init__(self, opacity_range=(0.2, 0.6)):
        self.opacity_range = opacity_range

    def __call__(self, x):
        B, C, H, W = x.shape
        result = x.clone()
        for i in range(B):
            opacity = random.uniform(*self.opacity_range)
            # Generate low-frequency noise for dust texture
            low_h, low_w = max(4, H // 8), max(4, W // 8)
            noise = torch.rand(1, low_h, low_w, device=x.device)
            dust = F.interpolate(noise.unsqueeze(0), size=(H, W),
                                 mode="bilinear", align_corners=False).squeeze(0)
            # Threshold for patchy appearance
            dust = (dust > random.uniform(0.3, 0.6)).float()
            # Dust color (tan/brown)
            dust_color = torch.tensor([0.6, 0.5, 0.35],
                                      device=x.device).view(3, 1, 1)
            result[i] = result[i] * (1 - opacity * dust) + dust_color * opacity * dust
        return result


class AngledTapePatch:
    """
    Simulates tape patches at random angles with lighting-aware blending.
    Unlike training's rectangular patches.
    """

    def __init__(self, num_strips_range=(1, 3)):
        self.num_strips_range = num_strips_range

    def __call__(self, x):
        B, C, H, W = x.shape
        result = x.clone()
        for i in range(B):
            n_strips = random.randint(*self.num_strips_range)
            for _ in range(n_strips):
                # Create angled strip via rotation
                thickness = random.randint(2, max(3, H // 8))
                angle = random.uniform(-45, 45)

                # Create horizontal strip, then rotate concept via coordinates
                yy = torch.arange(H, device=x.device).float()
                xx = torch.arange(W, device=x.device).float()
                grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")

                # Line equation: y*cos(θ) - x*sin(θ) = d
                theta = math.radians(angle)
                d = random.uniform(0, H)
                dist = torch.abs(grid_y * math.cos(theta) -
                                 grid_x * math.sin(theta) - d)
                tape_mask = (dist < thickness).float()

                # Semi-transparent tape color (gray/white)
                tape_color = torch.tensor([
                    random.uniform(0.6, 0.9),
                    random.uniform(0.6, 0.9),
                    random.uniform(0.6, 0.9),
                ], device=x.device).view(3, 1, 1)
                alpha = random.uniform(0.5, 0.85)

                result[i] = result[i] * (1 - alpha * tape_mask) + \
                            tape_color * alpha * tape_mask
        return result


class LensFlareGlare:
    """
    IR glare with lens flare artifacts (concentric rings + streak).
    Different from training's simple Gaussian brightness.
    """

    def __init__(self, intensity_range=(0.3, 1.5)):
        self.intensity_range = intensity_range

    def __call__(self, x):
        B, C, H, W = x.shape
        device = x.device
        result = x.clone()

        yy = torch.linspace(-1, 1, H, device=device)
        xx = torch.linspace(-1, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")

        for i in range(B):
            cy, cx = random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)
            intensity = random.uniform(*self.intensity_range)

            dist = torch.sqrt((grid_y - cy) ** 2 + (grid_x - cx) ** 2)

            # Main glow
            glow = intensity * torch.exp(-dist ** 2 / (2 * 0.2 ** 2))

            # Concentric ring artifacts
            ring = 0.3 * intensity * torch.cos(dist * 15) * \
                   torch.exp(-dist ** 2 / 0.5)
            ring = torch.clamp(ring, min=0)

            # Horizontal lens streak
            streak = 0.2 * intensity * \
                     torch.exp(-((grid_y - cy) ** 2) / 0.01) * \
                     torch.exp(-((grid_x - cx) ** 2) / 0.8)

            flare = glow + ring + streak
            result[i] = result[i] + flare.unsqueeze(0)

        return result


class FogCondensation:
    """
    Partial fog or condensation on the camera lens.
    Creates a non-uniform white haze — not used in training.
    """

    def __init__(self, density_range=(0.1, 0.5)):
        self.density_range = density_range

    def __call__(self, x):
        B, C, H, W = x.shape
        result = x.clone()
        for i in range(B):
            density = random.uniform(*self.density_range)
            # Low-freq fog pattern
            fog_h, fog_w = max(4, H // 4), max(4, W // 4)
            fog_noise = torch.rand(1, fog_h, fog_w, device=x.device)
            fog = F.interpolate(fog_noise.unsqueeze(0), size=(H, W),
                                mode="bilinear", align_corners=False).squeeze(0)
            fog = fog * density
            # Fog is white
            result[i] = result[i] * (1 - fog) + fog
        return result


class RealisticCameraShake:
    """
    Camera shake blur with random trajectory (not a simple 1D kernel).
    Simulates animal bumping the camera mount.
    """

    def __init__(self, magnitude_range=(1, 4)):
        self.magnitude_range = magnitude_range

    def __call__(self, x):
        B, C, H, W = x.shape
        k_size = 7
        kernel = torch.zeros(k_size, k_size, device=x.device)
        center = k_size // 2

        # Random walk trajectory
        py, px = center, center
        steps = random.randint(3, 6)
        mag = random.uniform(*self.magnitude_range)
        for _ in range(steps):
            kernel[int(py), int(px)] = 1.0
            py = max(0, min(k_size - 1, py + random.uniform(-mag, mag)))
            px = max(0, min(k_size - 1, px + random.uniform(-mag, mag)))
        kernel[int(py), int(px)] = 1.0

        kernel = kernel / (kernel.sum() + 1e-8)
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)

        padding = k_size // 2
        return F.conv2d(x, kernel, padding=padding, groups=C)


class UnseenTamperCompose:
    """
    Composes unseen attacks for the held-out test set.
    These are DIFFERENT transforms from the training pipeline.
    """

    def __init__(self, transforms=None, max_transforms=3):
        if transforms is None:
            transforms = [
                TexturedDustOverlay(),
                AngledTapePatch(),
                LensFlareGlare(),
                FogCondensation(),
                RealisticCameraShake(),
            ]
        self.transforms = transforms
        self.max_transforms = max_transforms

    def __call__(self, x):
        n = random.randint(1, min(self.max_transforms, len(self.transforms)))
        selected = random.sample(self.transforms, n)
        for t in selected:
            x = t(x)
        return x


def get_unseen_tamper_pipeline(max_transforms=2):
    """Returns unseen tampering pipeline for honest evaluation."""
    return UnseenTamperCompose(max_transforms=max_transforms)
