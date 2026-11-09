"""Image-conditioned noise prediction and action sampling."""

import torch
from diffusers import DDPMScheduler
from torch import nn

from starter.data.diffusion import preprocess_image
from starter.interfaces import ACTION_DIM


class DiffusionPolicy(nn.Module):
    def __init__(self, horizon, image_size, scheduler_config, bounds):
        super().__init__()
        self.horizon = horizon
        self.image_size = image_size
        self.scheduler = DDPMScheduler(**scheduler_config)
        for key, value in bounds.items():
            self.register_buffer(key, torch.as_tensor(value, dtype=torch.float32))

        self.vision = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.SiLU(),
        )
        self.condition = nn.Linear(128 + ACTION_DIM, 128)
        self.timestep = nn.Embedding(self.scheduler.config.num_train_timesteps, 128)
        self.position = nn.Parameter(torch.zeros(1, 128, horizon))
        self.input = nn.Conv1d(ACTION_DIM, 128, 1)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.GroupNorm(8, 128), nn.SiLU(),
                nn.Conv1d(128, 128, 3, dilation=dilation, padding=dilation),
            )
            for dilation in (1, 2, 4, 1)
        )
        self.output = nn.Conv1d(128, ACTION_DIM, 1)

    def normalize(self, x, key):
        low, high = getattr(self, f"{key}_min"), getattr(self, f"{key}_max")
        half_range = (high - low) / 2
        scale = torch.where(half_range < 1e-6, 1.0, half_range)
        return (x - (high + low) / 2) / scale

    def encode_observation(self, image, state):
        features = self.vision(image.float() / 255)
        return self.condition(torch.cat((features, self.normalize(state, "state")), dim=-1))

    def forward(self, noisy_actions, timesteps, condition):
        context = (condition + self.timestep(timesteps)).unsqueeze(-1)
        x = self.input(noisy_actions.transpose(1, 2)) + self.position
        for block in self.blocks:
            x = x + block(x + context)
        return self.output(x).transpose(1, 2)

    @torch.inference_mode()
    def predict_action_chunk(self, rgb, state, *, num_inference_steps=None):
        """Return a denormalized [horizon, 16] action chunk for one observation."""
        device = self.state_min.device
        image = torch.as_tensor(preprocess_image(rgb, self.image_size), device=device)
        state = torch.as_tensor(state, device=device, dtype=torch.float32).reshape(1, ACTION_DIM)
        condition = self.encode_observation(image.unsqueeze(0), state)
        sample = torch.randn((1, self.horizon, ACTION_DIM), device=device)
        steps = num_inference_steps or self.scheduler.config.num_train_timesteps
        self.scheduler.set_timesteps(steps)
        for step in self.scheduler.timesteps.tolist():
            k = torch.full((1,), step, device=device, dtype=torch.long)
            noise = self(sample, k, condition)
            sample = self.scheduler.step(noise, step, sample).prev_sample
        return (self.action_min + (sample[0] + 1) * (self.action_max - self.action_min) / 2)


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
