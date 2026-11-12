"""Conditional flow matching with a randomly initialized ViT and action DiT."""

import math
from collections import deque

import torch
from torch import nn
from torchvision.models.vision_transformer import VisionTransformer

from starter.data.diffusion import preprocess_image
from starter.interfaces import ACTION_DIM


def _time_embedding(t, width):
    frequencies = torch.exp(
        -math.log(10000) * torch.arange(width // 2, device=t.device) / (width // 2)
    )
    angles = t.float()[:, None] * 1000 * frequencies[None]
    return torch.cat((angles.cos(), angles.sin()), dim=-1)


def _modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class _DiTBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.norms = nn.ModuleList(
            nn.LayerNorm(width, elementwise_affine=False, eps=1e-6) for _ in range(3)
        )
        self.self_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * 4), nn.GELU(approximate="tanh"),
            nn.Dropout(dropout), nn.Linear(width * 4, width), nn.Dropout(dropout),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, width * 9))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x, memory, condition):
        shift1, scale1, gate1, shift2, scale2, gate2, shift3, scale3, gate3 = (
            self.modulation(condition).chunk(9, dim=-1)
        )
        query = _modulate(self.norms[0](x), shift1, scale1)
        x = x + gate1[:, None] * self.self_attention(query, query, query, need_weights=False)[0]
        query = _modulate(self.norms[1](x), shift2, scale2)
        x = x + gate2[:, None] * self.cross_attention(query, memory, memory, need_weights=False)[0]
        return x + gate3[:, None] * self.mlp(_modulate(self.norms[2](x), shift3, scale3))


class FlowMatchingPolicy(nn.Module):
    def __init__(
        self, horizon, image_size, inference_steps, bounds, n_obs_steps=2, action_steps=4,
        vision_dim=384, vision_depth=12, vision_heads=6, model_dim=512, depth=12,
        num_heads=8, dropout=0.1,
    ):
        super().__init__()
        sizes = (horizon, image_size, inference_steps, n_obs_steps, action_steps,
                 vision_dim, vision_depth, vision_heads, model_dim, depth, num_heads)
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in sizes):
            raise ValueError("Model sizes and inference_steps must be positive integers.")
        if image_size % 16 or action_steps > horizon:
            raise ValueError("image_size must be divisible by 16; action_steps must not exceed horizon.")
        if vision_dim % vision_heads or model_dim % num_heads or model_dim % 2:
            raise ValueError("Attention widths must be divisible by heads; model_dim must also be even.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        self.horizon = horizon
        self.image_size = image_size
        self.inference_steps = inference_steps
        self.n_obs_steps = n_obs_steps
        self.action_steps = action_steps
        self.model_dim = model_dim
        for key in ("state", "action"):
            low, high = (torch.as_tensor(bounds[f"{key}_{side}"], dtype=torch.float32).clone()
                         for side in ("min", "max"))
            if low.shape != (ACTION_DIM,) or high.shape != low.shape:
                raise ValueError("Normalization bounds must contain 16 values per vector.")
            if not (torch.isfinite(low).all() and torch.isfinite(high).all() and (high >= low).all()):
                raise ValueError("Normalization bounds must be finite and ordered.")
            self.register_buffer(f"{key}_min", low)
            self.register_buffer(f"{key}_max", high)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.vision = VisionTransformer(
            image_size=image_size, patch_size=16, num_layers=vision_depth, num_heads=vision_heads,
            hidden_dim=vision_dim, mlp_dim=vision_dim * 4, dropout=dropout,
            attention_dropout=dropout,
        )
        self.vision.heads = nn.Identity()
        self.vision_projection = nn.Linear(vision_dim, model_dim)
        self.state_encoder = nn.Sequential(
            nn.Linear(ACTION_DIM, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim),
        )
        patches = (image_size // 16) ** 2
        self.spatial_position = nn.Parameter(torch.empty(1, 1, patches, model_dim))
        self.observation_position = nn.Parameter(torch.empty(1, n_obs_steps, 1, model_dim))
        self.action_position = nn.Parameter(torch.empty(1, horizon, model_dim))
        for position in (self.spatial_position, self.observation_position, self.action_position):
            nn.init.normal_(position, std=0.02)
        self.memory_norm = nn.LayerNorm(model_dim)
        self.input = nn.Linear(ACTION_DIM, model_dim)
        self.time_encoder = nn.Sequential(
            nn.Linear(model_dim, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim),
        )
        self.blocks = nn.ModuleList(_DiTBlock(model_dim, num_heads, dropout) for _ in range(depth))
        self.output_norm = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
        self.output_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_dim, model_dim * 2))
        self.output = nn.Linear(model_dim, ACTION_DIM)
        nn.init.zeros_(self.output_modulation[-1].weight)
        nn.init.zeros_(self.output_modulation[-1].bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.reset()

    @staticmethod
    def canonicalize(values, *, strict=True):
        """Use unit quaternions in the w>=0 hemisphere and bounded grippers."""
        values = torch.as_tensor(values, dtype=torch.float32).clone()
        if values.shape[-1] != ACTION_DIM or not torch.isfinite(values).all():
            raise ValueError("Expected finite 16-D end-effector vectors.")
        for start in (3, 11):
            quat = values[..., start:start + 4]
            norm = quat.norm(dim=-1, keepdim=True)
            if strict and (norm < 1e-6).any():
                raise ValueError("End-effector quaternions must have nonzero norm.")
            identity = torch.zeros_like(quat)
            identity[..., 0] = 1
            quat = torch.where(norm < 1e-6, identity, quat / norm.clamp_min(1e-6))
            values[..., start:start + 4] = torch.where(quat[..., :1] < 0, -quat, quat)
        values[..., 7] = values[..., 7].clamp(0, 1)
        values[..., 15] = values[..., 15].clamp(0, 1)
        return values

    def normalize(self, x, key):
        low, high = getattr(self, f"{key}_min"), getattr(self, f"{key}_max")
        half_range = (high - low) / 2
        scale = torch.where(half_range < 1e-6, 1.0, half_range)
        return (x - (high + low) / 2) / scale

    def denormalize_actions(self, sample):
        half_range = (self.action_max - self.action_min) / 2
        actions = sample.clamp(-1, 1) * half_range + (self.action_max + self.action_min) / 2
        return self.canonicalize(actions, strict=False)

    def encode_visual(self, images):
        expected = (self.n_obs_steps, 3, self.image_size, self.image_size)
        if images.ndim != 5 or tuple(images.shape[1:]) != expected:
            raise ValueError(f"Expected images [B, {', '.join(map(str, expected))}].")
        image = images.flatten(0, 1).float() / 255
        image = (image - self.image_mean) / self.image_std
        tokens = self.vision.conv_proj(image).flatten(2).transpose(1, 2)
        cls = self.vision.class_token.expand(len(tokens), -1, -1)
        tokens = self.vision.encoder(torch.cat((cls, tokens), dim=1))[:, 1:]
        return tokens.reshape(len(images), self.n_obs_steps, -1, self.vision.hidden_dim)

    def condition_from_visual(self, visual, states):
        if tuple(states.shape) != (len(visual), self.n_obs_steps, ACTION_DIM):
            raise ValueError("Expected states [B, n_obs_steps, 16].")
        tokens = self.vision_projection(visual) + self.spatial_position
        state = self.state_encoder(self.normalize(states, "state"))
        memory = torch.cat((tokens, state[:, :, None]), dim=2) + self.observation_position
        return self.memory_norm(memory.flatten(1, 2)), state[:, -1]

    def encode_observation(self, images, states):
        return self.condition_from_visual(self.encode_visual(images), states)

    def forward(self, noisy_actions, timesteps, condition):
        memory, state = condition
        context = self.time_encoder(_time_embedding(timesteps, self.model_dim)) + state
        x = self.input(noisy_actions) + self.action_position
        for block in self.blocks:
            x = block(x, memory, context)
        shift, scale = self.output_modulation(context).chunk(2, dim=-1)
        return self.output(_modulate(self.output_norm(x), shift, scale))

    def loss(self, images, states, actions, valid_steps, *, generator=None):
        actions = self.normalize(actions, "action")
        noise = torch.randn(actions.shape, device=actions.device, generator=generator)
        time = torch.rand(len(actions), device=actions.device, generator=generator)
        interpolated = torch.lerp(noise, actions, time[:, None, None])
        velocity = self(interpolated, time, self.encode_observation(images, states))
        mask = torch.arange(self.horizon, device=actions.device)[None] < valid_steps[:, None]
        error = (velocity.float() - (actions - noise)).square().mean(dim=-1)
        return (error * mask).sum() / mask.sum()

    @torch.no_grad()
    def sample_normalized_actions(self, condition, *, num_inference_steps=None, generator=None):
        """Sample before clipping and pose projection, preserving the flow action for RL."""
        steps = self.inference_steps if num_inference_steps is None else num_inference_steps
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise ValueError("num_inference_steps must be a positive integer.")
        states = condition[1]
        sample = torch.randn((len(states), self.horizon, ACTION_DIM), device=states.device, generator=generator)
        dt = 1 / steps
        for index in range(steps):
            time = torch.full((len(states),), index * dt, device=states.device)
            velocity = self(sample, time, condition)
            midpoint = sample + velocity * (dt / 2)
            sample = sample + self(midpoint, time + dt / 2, condition) * dt
        return sample

    @torch.inference_mode()
    def sample_actions(self, images, states, *, num_inference_steps=None, generator=None):
        condition = self.encode_observation(images, states)
        sample = self.sample_normalized_actions(
            condition, num_inference_steps=num_inference_steps, generator=generator,
        )
        return self.denormalize_actions(sample)

    def reset(self):
        """Clear observation history at every episode boundary."""
        self._images = deque(maxlen=self.n_obs_steps)
        self._states = deque(maxlen=self.n_obs_steps)

    @torch.inference_mode()
    def update_observation(self, rgb, state):
        """Feed every environment step, including intermediate steps of an action chunk."""
        image = torch.from_numpy(preprocess_image(rgb, self.image_size))
        state = self.canonicalize(torch.as_tensor(state).reshape(ACTION_DIM)).cpu()
        if not self._images:
            for _ in range(self.n_obs_steps):
                self._images.append(image)
                self._states.append(state)
        else:
            self._images.append(image)
            self._states.append(state)

    @torch.inference_mode()
    def predict_action_chunk(self, rgb=None, state=None, *, num_inference_steps=None):
        """Return [horizon, 16] from history, optionally adding the current observation."""
        if self.training:
            raise RuntimeError("Call eval() before predicting actions.")
        if (rgb is None) != (state is None):
            raise ValueError("Provide both rgb and state, or use the existing observation history.")
        if rgb is not None:
            self.update_observation(rgb, state)
        if not self._images:
            raise ValueError("Feed an observation before requesting actions.")
        device = self.action_min.device
        images = torch.stack(tuple(self._images))[None].to(device)
        states = torch.stack(tuple(self._states))[None].to(device)
        return self.sample_actions(images, states, num_inference_steps=num_inference_steps)[0]


def load_policy(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    weights = checkpoint["model"]
    bounds = {f"{key}_{side}": weights[f"{key}_{side}"]
              for key in ("state", "action") for side in ("min", "max")}
    model = FlowMatchingPolicy(**checkpoint["policy"], bounds=bounds)
    model.load_state_dict(weights)
    return model.to(device).eval()
