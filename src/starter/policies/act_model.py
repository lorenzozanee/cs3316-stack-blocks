"""ACT DETR-VAE matching the XPolicyLab baseline, with 16-D ee actions.

Adapted from policy/ACT/detr/models/{detr_vae,backbone,position_encoding}.py.
Copyright Facebook, Inc. and the ACT authors; see the upstream LICENSE files.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.transforms.functional import normalize

from starter.config import require
from starter.interfaces import ACTION_DIM
from starter.policies.act_transformer import Encoder, EncoderLayer, Transformer


class PositionEmbeddingSine(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.features = features

    def forward(self, image: Tensor) -> Tensor:
        valid = torch.ones_like(image[0, [0]])
        y = valid.cumsum(1, dtype=torch.float32)
        x = valid.cumsum(2, dtype=torch.float32)
        y = y / (y[:, -1:, :] + 1e-6) * (2 * math.pi)
        x = x / (x[:, :, -1:] + 1e-6) * (2 * math.pi)
        dims = torch.arange(self.features, dtype=torch.float32, device=image.device)
        dims = 10000 ** (2 * (dims // 2) / self.features)
        x, y = x[..., None] / dims, y[..., None] / dims
        x = torch.stack([x[..., 0::2].sin(), x[..., 1::2].cos()], dim=4).flatten(3)
        y = torch.stack([y[..., 0::2].sin(), y[..., 1::2].cos()], dim=4).flatten(3)
        return torch.cat([y, x], dim=3).permute(0, 3, 1, 2)


class Backbone(nn.Module):
    def __init__(self, pretrained: bool):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        network = resnet18(weights=weights, norm_layer=FrozenBatchNorm2d)
        self.body = IntermediateLayerGetter(network, return_layers={"layer4": "0"})

    def forward(self, image: Tensor) -> Tensor:
        return self.body(image)["0"]


class BackboneWithPosition(nn.Sequential):
    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        features = self[0](image)
        return features, self[1](features).to(features.dtype)


def sinusoid_table(length: int, hidden: int) -> Tensor:
    positions = np.arange(length)[:, None]
    channels = np.arange(hidden)[None, :]
    angles = positions / np.power(10000, 2 * (channels // 2) / hidden)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return torch.tensor(angles, dtype=torch.float32).unsqueeze(0)


class ACTModel(nn.Module):
    def __init__(self, backbone: nn.Module, transformer: Transformer, encoder: Encoder, config: dict):
        super().__init__()
        hidden = int(require(config, "hidden_dim"))
        self.num_queries = int(require(config, "chunk_size"))
        self.camera_names = list(require(config, "camera_names"))
        self.transformer = transformer
        self.encoder = encoder
        self.action_head = nn.Linear(hidden, ACTION_DIM)
        self.is_pad_head = nn.Linear(hidden, 1)
        self.query_embed = nn.Embedding(self.num_queries, hidden)
        self.input_proj = nn.Conv2d(512, hidden, kernel_size=1)
        self.backbones = nn.ModuleList([backbone])
        self.input_proj_robot_state = nn.Linear(ACTION_DIM, hidden)
        self.latent_dim = 32
        self.cls_embed = nn.Embedding(1, hidden)
        self.encoder_action_proj = nn.Linear(ACTION_DIM, hidden)
        self.encoder_joint_proj = nn.Linear(ACTION_DIM, hidden)
        self.latent_proj = nn.Linear(hidden, self.latent_dim * 2)
        self.register_buffer("pos_table", sinusoid_table(self.num_queries + 2, hidden))
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden)
        self.additional_pos_embed = nn.Embedding(2, hidden)

    def forward(self, qpos: Tensor, image: Tensor, actions: Tensor | None = None,
                is_pad: Tensor | None = None):
        batch = qpos.shape[0]
        if actions is not None:
            action_embed = self.encoder_action_proj(actions)
            joint_embed = self.encoder_joint_proj(qpos).unsqueeze(1)
            cls = self.cls_embed.weight.unsqueeze(0).repeat(batch, 1, 1)
            sequence = torch.cat([cls, joint_embed, action_embed], dim=1).permute(1, 0, 2)
            prefix_mask = torch.full((batch, 2), False, device=qpos.device)
            padding = torch.cat([prefix_mask, is_pad], dim=1)
            positions = self.pos_table.clone().detach().permute(1, 0, 2)
            encoded = self.encoder(sequence, pos=positions, src_key_padding_mask=padding)[0]
            mu, logvar = self.latent_proj(encoded).chunk(2, dim=-1)
            std = logvar.div(2).exp()
            latent = mu + std * torch.empty_like(std).normal_()
        else:
            mu = logvar = None
            latent = torch.zeros((batch, self.latent_dim), dtype=torch.float32, device=qpos.device)
        latent = self.latent_out_proj(latent)
        features, positions = [], []
        for camera in range(len(self.camera_names)):
            feature, position = self.backbones[0](image[:, camera])
            features.append(self.input_proj(feature))
            positions.append(position)
        decoded = self.transformer(
            torch.cat(features, dim=3), self.query_embed.weight,
            torch.cat(positions, dim=3), latent, self.input_proj_robot_state(qpos),
            self.additional_pos_embed.weight,
        )
        # The pinned baseline selects the FIRST intermediate decoder output.
        # Keep all layers and their computation, including dropout RNG consumption.
        hidden = decoded[0]
        return self.action_head(hidden), self.is_pad_head(hidden), (mu, logvar)


class ACTPolicy(nn.Module):
    def __init__(self, config: dict, *, pretrained_backbone: bool = True):
        super().__init__()
        hidden = int(require(config, "hidden_dim"))
        heads = int(require(config, "nheads"))
        feedforward = int(require(config, "dim_feedforward"))
        dropout = float(require(config, "dropout"))
        encoders = int(require(config, "enc_layers"))
        decoders = int(require(config, "dec_layers"))
        if hidden <= 0 or hidden % 4 or heads <= 0 or hidden % heads:
            raise ValueError("hidden_dim must be divisible by 4 and nheads.")
        if encoders < 1 or decoders < 1 or int(require(config, "chunk_size")) < 1:
            raise ValueError("ACT layer counts and chunk_size must be positive.")
        # Preserve upstream construction order and initialization rules.
        backbone = BackboneWithPosition(Backbone(pretrained_backbone), PositionEmbeddingSine(hidden // 2))
        transformer = Transformer(hidden, heads, feedforward, dropout, encoders, decoders)
        encoder = Encoder(EncoderLayer(hidden, heads, feedforward, dropout), encoders)
        self.model = ACTModel(backbone, transformer, encoder, config)
        self.kl_weight = float(require(config, "kl_weight"))

    def forward(self, qpos: Tensor, image: Tensor, actions: Tensor | None = None,
                is_pad: Tensor | None = None):
        if qpos.ndim != 2 or qpos.shape[-1] != ACTION_DIM:
            raise ValueError("ACT qpos must have shape [batch, 16].")
        if image.ndim != 5 or image.shape[1:3] != (len(self.model.camera_names), 3):
            raise ValueError("ACT image must have shape [batch, cameras, 3, height, width].")
        image = normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if actions is None:
            return self.model(qpos, image)[0]
        chunk = self.model.num_queries
        if actions.ndim != 3 or actions.shape[-1] != ACTION_DIM or actions.shape[1] < chunk:
            raise ValueError("ACT actions must have shape [batch, at least chunk_size, 16].")
        if is_pad is None or is_pad.shape != actions.shape[:2] or is_pad.dtype != torch.bool:
            raise ValueError("ACT is_pad must be a boolean mask matching the action sequence.")
        actions, is_pad = actions[:, :chunk], is_pad[:, :chunk]
        predicted, _, (mu, logvar) = self.model(qpos, image, actions, is_pad)
        l1 = (F.l1_loss(actions, predicted, reduction="none") * ~is_pad.unsqueeze(-1)).mean()
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(1).mean()
        return {"l1": l1, "kl": kl, "loss": l1 + self.kl_weight * kl}

    def configure_optimizers(self, config: dict) -> torch.optim.AdamW:
        named = list(self.model.named_parameters())
        groups = [
            {"params": [p for name, p in named if "backbone" not in name and p.requires_grad]},
            {"params": [p for name, p in named if "backbone" in name and p.requires_grad],
             "lr": float(require(config, "lr_backbone"))},
        ]
        return torch.optim.AdamW(groups, lr=float(require(config, "lr")),
                                 weight_decay=float(require(config, "weight_decay")))
