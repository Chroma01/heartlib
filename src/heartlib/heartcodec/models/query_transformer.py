"""The pretrained vqv12 feature fusion and 12.5 Hz query encoder.

Module names intentionally follow the original HeartCodec checkpoint. Only the
inference path is retained: weight-normalized self-attention with partial RoPE,
Q/K LayerNorm, sigmoid GLU feed-forward layers, and LayerScale residuals.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm


class _RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer(
            "inv_freq", 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        )

    def forward(self, length):
        # The original computes rotations in FP32, even under autocast.
        with torch.autocast(device_type=self.inv_freq.device.type, enabled=False):
            positions = torch.arange(
                length, device=self.inv_freq.device, dtype=torch.float32
            )
            frequencies = torch.einsum("i,j->ij", positions, self.inv_freq.float())
            return torch.cat((frequencies, frequencies), dim=-1)


def _apply_rotary(x, frequencies):
    dtype = x.dtype
    with torch.autocast(device_type=x.device.type, enabled=False):
        x = x.float()
        width = frequencies.shape[-1]
        rotated, remainder = x[..., :width], x[..., width:]
        first, second = rotated.chunk(2, dim=-1)
        rotated = rotated * frequencies.cos() + torch.cat(
            (-second, first), dim=-1
        ) * frequencies.sin()
        return torch.cat((rotated, remainder), dim=-1).to(dtype)


class _LayerScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.full((dim,), 1e-2))

    def forward(self, x):
        return x * self.scale


class _SigmoidGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = weight_norm(nn.Linear(dim, 8 * dim))

    def forward(self, x):
        values, gate = self.proj(x).chunk(2, dim=-1)
        return values * gate.sigmoid()


class _FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ff = nn.Sequential(
            _SigmoidGLU(dim),
            nn.Identity(),
            weight_norm(nn.Linear(4 * dim, dim)),
            nn.Identity(),
        )

    def forward(self, x):
        return self.ff(x)


class _QueryAttention(nn.Module):
    def __init__(self, dim, head_dim):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.to_qkv = weight_norm(nn.Linear(dim, 3 * dim, bias=False))
        self.to_out = weight_norm(nn.Linear(dim, dim, bias=False))
        self.q_norm = nn.LayerNorm(head_dim)
        self.k_norm = nn.LayerNorm(head_dim)

    def forward(self, x, frequencies):
        batch, length, dim = x.shape
        q, k, v = (
            value.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
            for value in self.to_qkv(x).chunk(3, dim=-1)
        )
        q = _apply_rotary(self.q_norm(q), frequencies)
        k = _apply_rotary(self.k_norm(k), frequencies)
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return self.to_out(attended.transpose(1, 2).reshape(batch, length, dim))


class _QueryTransformerBlock(nn.Module):
    def __init__(self, dim, head_dim):
        super().__init__()
        self.self_attn = _QueryAttention(dim, head_dim)
        self.self_attn_scale = _LayerScale(dim)
        self.ff = _FeedForward(dim)
        self.ff_scale = _LayerScale(dim)
        self.rope = _RotaryEmbedding(max(head_dim // 2, 32))

    def forward(self, x):
        frequencies = self.rope(x.shape[1])
        x = x + self.self_attn_scale(self.self_attn(x, frequencies))
        return x + self.ff_scale(self.ff(x))


class HeartCodecQueryEncoder(nn.Module):
    """Fuse aligned 25 Hz features and extract one query per two frames.

    Inputs have shape ``[batch, frames, channels]``: 2048-channel acoustic and
    semantic MuEncoder features, and 1536-channel Whisper and WavLM features.
    The feature extractor must pad the frame count to a multiple of ``interval``.
    Output has shape ``[batch, frames // interval, dim]``.
    """

    def __init__(self, dim=512, num_layers=6, head_dim=128, interval=2):
        super().__init__()
        if head_dim < 32 or head_dim % 4 or dim % head_dim:
            raise ValueError("head_dim must be a multiple of 4, at least 32, and divide dim")
        if num_layers < 1 or interval < 1:
            raise ValueError("num_layers and interval must be positive")
        self.interval = interval
        self.cond_fusion_layer_semantic = nn.Linear(2048, dim)
        self.cond_fusion_layer_acoustic = nn.Linear(2048, dim)
        self.cond_fusion_layer_phone = nn.Linear(1536, dim)
        self.feature_proj = nn.Linear(3 * dim + 1536, dim)
        self.cls_token = nn.Parameter(torch.randn(1, dim))
        self.encoder_transformers = nn.Sequential(
            *(_QueryTransformerBlock(dim, head_dim) for _ in range(num_layers))
        )

    def forward(self, acoustic, semantic, whisper, wavlm):
        features = (acoustic, semantic, whisper, wavlm)
        for name, feature, channels in zip(
            ("acoustic", "semantic", "whisper", "wavlm"),
            features,
            (2048, 2048, 1536, 1536),
        ):
            if feature.ndim != 3 or feature.shape[-1] != channels:
                raise ValueError(f"{name} must have shape [batch, frames, {channels}]")
            if feature.shape[:2] != acoustic.shape[:2]:
                raise ValueError("All feature streams must have the same batch and frame count")
        batch, frames, _ = acoustic.shape
        if frames == 0 or frames % self.interval:
            raise ValueError("Feature frame count must be nonzero and divisible by interval")
        fused = self.feature_proj(
            torch.cat(
                (
                    self.cond_fusion_layer_acoustic(acoustic),
                    self.cond_fusion_layer_semantic(semantic),
                    whisper,
                    self.cond_fusion_layer_phone(wavlm),
                ),
                dim=-1,
            )
        )
        groups = fused.reshape(batch, frames // self.interval, self.interval, -1)
        queries = self.cls_token.reshape(1, 1, 1, -1).expand(
            batch, frames // self.interval, 1, -1
        )
        sequence = torch.cat((groups, queries), dim=2).flatten(1, 2)
        sequence = self.encoder_transformers(sequence)
        return sequence[:, self.interval :: self.interval + 1, :]
