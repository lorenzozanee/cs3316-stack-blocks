"""A frozen flow policy with a bounded, per-step Gaussian residual.

Sources: see SOURCES.md.
"""

from collections import deque
import math

import torch
from torch import nn
from torch.distributions import Normal

from starter.data.diffusion import preprocess_image
from starter.interfaces import ACTION_DIM


class ResidualActorCritic(nn.Module):
    def __init__(self, feature_dim, hidden_dim, initial_std):
        super().__init__()

        def mlp(output_dim):
            return nn.Sequential(
                nn.Linear(feature_dim, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, output_dim),
            )

        self.actor = mlp(ACTION_DIM)
        self.critic = mlp(1)
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), math.log(initial_std)))

    def distribution(self, features):
        return Normal(self.actor(features), self.log_std.clamp(-5, 1).exp())

    def value(self, features):
        return self.critic(features).squeeze(-1)

    def evaluate(self, features, latents):
        distribution = self.distribution(features)
        return (
            distribution.log_prob(latents).sum(-1),
            distribution.entropy().sum(-1),
            self.value(features),
        )


class ResidualPolicy(nn.Module):
    def __init__(
        self, base, hidden_dim=256, initial_std=0.2, position_scale=0.02,
        quaternion_scale=0.05, gripper_scale=0.1, noise_seed=0,
    ):
        super().__init__()
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim < 1:
            raise ValueError("hidden_dim must be a positive integer.")
        for name, value in (
            ("initial_std", initial_std), ("position_scale", position_scale),
            ("quaternion_scale", quaternion_scale), ("gripper_scale", gripper_scale),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        self.config = {
            "hidden_dim": hidden_dim, "initial_std": initial_std,
            "position_scale": position_scale, "quaternion_scale": quaternion_scale,
            "gripper_scale": gripper_scale, "noise_seed": noise_seed,
        }
        self.base = base.requires_grad_(False).eval()
        self.feature_dim = base.n_obs_steps * base.model_dim + 2 * ACTION_DIM + 1
        self.head = ResidualActorCritic(self.feature_dim, hidden_dim, initial_std)
        scales = [position_scale] * 3 + [quaternion_scale] * 4 + [gripper_scale]
        self.register_buffer("residual_scale", torch.tensor(scales * 2, dtype=torch.float32))
        generator = torch.Generator().manual_seed(noise_seed)
        self.register_buffer("base_noise", torch.randn(base.horizon, ACTION_DIM, generator=generator))
        self._controller = None
        self.to(base.action_min.device)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def compose(self, base_actions, latents):
        # PPO scores the stored Gaussian latent before this fixed action transform.
        actions = base_actions + latents.tanh() * self.residual_scale
        return self.base.canonicalize(actions, strict=False)

    def reset(self):
        self._controller = None

    @torch.no_grad()
    def update_observation(self, rgb, state):
        if self._controller is None:
            self._controller = ResidualController(self, 1)
        self._controller.observe([{"image": rgb, "proprio": state}], [0])

    @torch.no_grad()
    def predict_action_chunk(self, rgb=None, state=None):
        """Return one corrected action so every residual gets fresh feedback."""
        if self.training:
            raise RuntimeError("Call eval() before predicting actions.")
        if (rgb is None) != (state is None):
            raise ValueError("Provide both rgb and state, or use the existing observation history.")
        if rgb is not None:
            self.update_observation(rgb, state)
        if self._controller is None:
            raise ValueError("Feed an observation before requesting actions.")
        features, base_actions = self._controller.prepare([0])
        actions = self.compose(base_actions, self.head.distribution(features).mean)
        self._controller.advance([0])
        return actions


class ResidualController:
    """Independent histories and receding-horizon base plans for parallel scenes."""

    def __init__(self, policy, num_envs):
        self.policy = policy
        self.images = [deque(maxlen=policy.base.n_obs_steps) for _ in range(num_envs)]
        self.states = [deque(maxlen=policy.base.n_obs_steps) for _ in range(num_envs)]
        self.chunks = [None] * num_envs
        self.cursors = [0] * num_envs
        self.cached = [None] * num_envs

    @torch.no_grad()
    def observe(self, observations, indices):
        if len(observations) != len(indices):
            raise ValueError("Observations must align with active environment indices.")
        base = self.policy.base
        for index, observation in zip(indices, observations):
            image = torch.from_numpy(preprocess_image(observation["image"], base.image_size))
            state = base.canonicalize(torch.as_tensor(observation["proprio"]).reshape(ACTION_DIM)).cpu()
            repeats = 1 if self.images[index] else base.n_obs_steps
            for _ in range(repeats):
                self.images[index].append(image)
                self.states[index].append(state)
            self.cached[index] = None

    @torch.no_grad()
    def prepare(self, indices):
        base = self.policy.base
        device = base.action_min.device
        missing = [index for index in indices if self.cached[index] is None]
        if missing:
            if any(not self.images[index] for index in missing):
                raise ValueError("Feed an observation before requesting actions.")
            images = torch.stack([torch.stack(tuple(self.images[index])) for index in missing]).to(device)
            states = torch.stack([torch.stack(tuple(self.states[index])) for index in missing]).to(device)
            memory, encoded_state = base.encode_observation(images, states)
            replans = [slot for slot, index in enumerate(missing)
                       if self.chunks[index] is None or self.cursors[index] >= base.action_steps]
            if replans:
                chunks = base.sample_from_condition(
                    (memory[replans], encoded_state[replans]),
                    initial_noise=self.policy.base_noise[None].expand(len(replans), -1, -1),
                ).clone()
                for slot, chunk in zip(replans, chunks):
                    index = missing[slot]
                    self.chunks[index] = chunk
                    self.cursors[index] = 0
            visual = memory.reshape(len(missing), base.n_obs_steps, -1, base.model_dim).mean(2).flatten(1)
            for slot, index in enumerate(missing):
                action = self.chunks[index][self.cursors[index]]
                phase = states.new_tensor([self.cursors[index] / max(1, base.action_steps - 1)])
                features = torch.cat((
                    visual[slot], base.normalize(states[slot, -1], "state"),
                    base.normalize(action, "action"), phase,
                ))
                self.cached[index] = (features, action)
        return (
            torch.stack([self.cached[index][0] for index in indices]),
            torch.stack([self.cached[index][1] for index in indices]),
        )

    def advance(self, indices):
        for index in indices:
            self.cursors[index] += 1
            self.cached[index] = None
