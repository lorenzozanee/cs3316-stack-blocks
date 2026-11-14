"""Position-aware, post-norm Transformer from XPolicyLab ACT's DETR.

Adapted from policy/ACT/detr/models/transformer.py, Copyright Facebook, Inc.
Apache-2.0: see third_party/XPolicyLab/policy/ACT/detr/LICENSE.
"""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn


class EncoderLayer(nn.TransformerEncoderLayer):
    """Add positions to attention queries/keys, leaving values unchanged."""

    def forward(self, src: Tensor, *, pos: Tensor, src_key_padding_mask=None) -> Tensor:
        query = key = src + pos
        attended = self.self_attn(query, key, src, key_padding_mask=src_key_padding_mask)[0]
        src = self.norm1(src + self.dropout1(attended))
        feedforward = self.linear2(self.dropout(self.activation(self.linear1(src))))
        return self.norm2(src + self.dropout2(feedforward))


class DecoderLayer(nn.TransformerDecoderLayer):
    def forward(self, tgt: Tensor, memory: Tensor, *, pos: Tensor, query_pos: Tensor) -> Tensor:
        query = key = tgt + query_pos
        attended = self.self_attn(query, key, tgt)[0]
        tgt = self.norm1(tgt + self.dropout1(attended))
        attended = self.multihead_attn(tgt + query_pos, memory + pos, memory)[0]
        tgt = self.norm2(tgt + self.dropout2(attended))
        feedforward = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        return self.norm3(tgt + self.dropout3(feedforward))


class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, count: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(count)])

    def forward(self, src: Tensor, *, pos: Tensor, src_key_padding_mask=None) -> Tensor:
        for layer in self.layers:
            src = layer(src, pos=pos, src_key_padding_mask=src_key_padding_mask)
        return src


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, count: int, hidden_dim: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(count)])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, tgt: Tensor, memory: Tensor, *, pos: Tensor, query_pos: Tensor) -> Tensor:
        intermediate = []
        for layer in self.layers:
            tgt = layer(tgt, memory, pos=pos, query_pos=query_pos)
            intermediate.append(self.norm(tgt))
        intermediate[-1] = self.norm(tgt)
        return torch.stack(intermediate)


class Transformer(nn.Module):
    def __init__(self, hidden_dim: int, nheads: int, dim_feedforward: int,
                 dropout: float, enc_layers: int, dec_layers: int):
        super().__init__()
        encoder_layer = EncoderLayer(hidden_dim, nheads, dim_feedforward, dropout)
        self.encoder = Encoder(encoder_layer, enc_layers)
        decoder_layer = DecoderLayer(hidden_dim, nheads, dim_feedforward, dropout)
        self.decoder = Decoder(decoder_layer, dec_layers, hidden_dim)
        # Upstream resets the visual Transformer, but not the separate CVAE encoder.
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, src: Tensor, query_embed: Tensor, pos: Tensor,
                latent: Tensor, proprio: Tensor, additional_pos: Tensor) -> Tensor:
        batch = src.shape[0]
        src = src.flatten(2).permute(2, 0, 1)
        pos = pos.flatten(2).permute(2, 0, 1).repeat(1, batch, 1)
        pos = torch.cat([additional_pos.unsqueeze(1).repeat(1, batch, 1), pos], dim=0)
        src = torch.cat([torch.stack([latent, proprio], dim=0), src], dim=0)
        query_pos = query_embed.unsqueeze(1).repeat(1, batch, 1)
        memory = self.encoder(src, pos=pos)
        decoded = self.decoder(torch.zeros_like(query_pos), memory, pos=pos, query_pos=query_pos)
        return decoded.transpose(1, 2)
