"""Image-conditioned noise prediction and action sampling."""

import torch
from diffusers import DDPMScheduler
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from torch import nn
from torch.nn import functional as F

from starter.data.diffusion import preprocess_image
from starter.interfaces import ACTION_DIM


class _ImageResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
        )
        self.skip = (
            nn.Identity() if in_channels == out_channels and stride == 1
            else nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
        )

    def forward(self, x):
        return F.silu(self.layers(x) + self.skip(x))


class _VisionEncoder(nn.Module):
    feature_dim = 128 * 4 + 32 * 2

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            _ImageResidual(32, 48, stride=2),
            _ImageResidual(48, 64, stride=2),
            _ImageResidual(64, 96),
            _ImageResidual(96, 128),
        )
        self.keypoints = nn.Conv2d(128, 32, 1)

    def forward(self, image):
        features = self.backbone(image)
        h, w = features.shape[-2:]
        heatmaps = self.keypoints(features).flatten(2).softmax(dim=-1).unflatten(-1, (h, w))
        x = (torch.arange(w, device=features.device, dtype=features.dtype) + 0.5) * (2 / w) - 1
        y = (torch.arange(h, device=features.device, dtype=features.dtype) + 0.5) * (2 / h) - 1
        points = torch.stack((heatmaps.sum(dim=-2) @ x, heatmaps.sum(dim=-1) @ y), dim=-1)
        pooled = F.adaptive_avg_pool2d(features, 2).flatten(1)
        return torch.cat((pooled, points.flatten(1)), dim=-1)


class _ConditionalResidual(nn.Module):
    def __init__(self, in_channels, out_channels, context_dim):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.film = nn.Linear(context_dim, 2 * out_channels)
        self.output = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
        )
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x, context):
        scale, bias = self.film(context).unsqueeze(-1).chunk(2, dim=1)
        h = self.norm(self.conv(x))
        return self.skip(x) + self.output(h * (1 + scale) + bias)


class _ActionUNet(nn.Module):
    def __init__(self, horizon, context_dim):
        super().__init__()
        self.position = nn.Parameter(torch.zeros(1, 96, horizon))
        self.input = nn.Conv1d(ACTION_DIM, 96, 1)
        self.down_blocks = nn.ModuleList(
            _ConditionalResidual(width, width, context_dim) for width in (96, 192)
        )
        self.downsample = nn.ModuleList(
            nn.Conv1d(a, b, 3, stride=2, padding=1) for a, b in ((96, 192), (192, 256))
        )
        self.middle = nn.ModuleList(_ConditionalResidual(256, 256, context_dim) for _ in range(2))
        self.up_blocks = nn.ModuleList(
            _ConditionalResidual(a + b, b, context_dim) for a, b in ((256, 192), (192, 96))
        )
        self.output = nn.Sequential(nn.GroupNorm(8, 96), nn.SiLU(), nn.Conv1d(96, ACTION_DIM, 1))

    def forward(self, noisy_actions, context):
        x = self.input(noisy_actions.transpose(1, 2)) + self.position
        skips = []
        for block, downsample in zip(self.down_blocks, self.downsample):
            x = block(x, context)
            skips.append(x)
            x = downsample(x)
        for block in self.middle:
            x = block(x, context)
        for block, skip in zip(self.up_blocks, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-1], mode="nearest")
            x = block(torch.cat((x, skip), dim=1), context)
        return self.output(x).transpose(1, 2)


class DiffusionPolicy(nn.Module):
    def __init__(self, horizon, image_size, inference_steps, scheduler_config, bounds):
        super().__init__()
        self.horizon = horizon
        self.image_size = image_size
        self.inference_steps = inference_steps
        self.scheduler = DDPMScheduler(**scheduler_config)
        for key, value in bounds.items():
            self.register_buffer(key, torch.as_tensor(value, dtype=torch.float32))

        self.vision = _VisionEncoder()
        self.proprio = nn.Sequential(nn.Linear(ACTION_DIM, 64), nn.SiLU())
        self.condition = nn.Sequential(
            nn.Linear(self.vision.feature_dim + 64, 256), nn.LayerNorm(256), nn.SiLU(),
        )
        self.timestep = nn.Sequential(
            Timesteps(128, flip_sin_to_cos=True, downscale_freq_shift=0),
            TimestepEmbedding(128, 256, out_dim=128),
        )
        self.denoiser = _ActionUNet(horizon, context_dim=256 + 128)

    def normalize(self, x, key):
        low, high = getattr(self, f"{key}_min"), getattr(self, f"{key}_max")
        half_range = (high - low) / 2
        scale = torch.where(half_range < 1e-6, 1.0, half_range)
        return (x - (high + low) / 2) / scale

    def denormalize_actions(self, sample):
        return self.action_min + (sample + 1) * (self.action_max - self.action_min) / 2

    def encode_observation(self, image, state):
        features = self.vision(image.float() / 255)
        proprio = self.proprio(self.normalize(state, "state"))
        return self.condition(torch.cat((features, proprio), dim=-1))

    def forward(self, noisy_actions, timesteps, condition):
        context = torch.cat((condition, self.timestep(timesteps)), dim=-1)
        return self.denoiser(noisy_actions, context)

    @torch.inference_mode()
    def sample_actions(self, images, states, *, num_inference_steps=None, generator=None):
        steps = self.inference_steps if num_inference_steps is None else num_inference_steps
        if not 2 <= steps <= self.scheduler.config.num_train_timesteps:
            raise ValueError("Sampling requires between 2 and num_train_timesteps denoising steps.")
        condition = self.encode_observation(images, states)
        device = states.device
        sample = torch.randn((len(states), self.horizon, ACTION_DIM), device=device, generator=generator)
        self.scheduler.set_timesteps(steps)
        for step in self.scheduler.timesteps.tolist():
            k = torch.full((len(states),), step, device=device, dtype=torch.long)
            noise = self(sample, k, condition)
            sample = self.scheduler.step(noise, step, sample, generator=generator).prev_sample
        return self.denormalize_actions(sample)

    @torch.inference_mode()
    def predict_action_chunk(self, rgb, state, *, num_inference_steps=None):
        """Return a denormalized [horizon, 16] action chunk for one observation."""
        device = self.state_min.device
        image = torch.as_tensor(preprocess_image(rgb, self.image_size), device=device)
        state = torch.as_tensor(state, device=device, dtype=torch.float32).reshape(1, ACTION_DIM)
        return self.sample_actions(image.unsqueeze(0), state, num_inference_steps=num_inference_steps)[0]


def load_policy(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    weights = checkpoint["model"]
    bounds = {
        f"{key}_{bound}": weights[f"{key}_{bound}"]
        for key in ("state", "action") for bound in ("min", "max")
    }
    model = DiffusionPolicy(**checkpoint["policy"], bounds=bounds)
    model.load_state_dict(weights)
    return model.to(device).eval()
