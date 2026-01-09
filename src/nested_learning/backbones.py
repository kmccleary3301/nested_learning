from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AttentionConfig:
    dim: int
    heads: int
    dropout: float = 0.0
    use_flash: bool = True
    causal: bool = True
    qk_l2_norm: bool = False
    qk_norm_eps: float = 1e-6
    local_conv_window: int | None = None


class SelfAttention(nn.Module):
    def __init__(self, config: AttentionConfig):
        super().__init__()
        if config.dim % config.heads != 0:
            msg = f"dim must be divisible by heads (got dim={config.dim}, heads={config.heads})"
            raise ValueError(msg)
        self.config = config
        self.heads = config.heads
        self.head_dim = config.dim // config.heads
        self.qkv = nn.Linear(config.dim, config.dim * 3, bias=False)
        self.out_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.dim)
        self.local_conv: nn.Conv1d | None = None
        if config.local_conv_window is not None:
            window = int(config.local_conv_window)
            if window <= 0:
                raise ValueError("local_conv_window must be positive")
            self.local_conv = nn.Conv1d(
                config.dim,
                config.dim,
                kernel_size=window,
                groups=config.dim,
                padding=0,
                bias=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        residual = x
        attn_inp = x
        if self.local_conv is not None:
            kernel = self.local_conv.kernel_size[0]
            attn_inp = attn_inp.transpose(1, 2)
            # Causal depthwise conv: only attends to past tokens.
            attn_inp = F.pad(attn_inp, (kernel - 1, 0))
            attn_inp = self.local_conv(attn_inp).transpose(1, 2)
        q, k, v = self._compute_qkv(attn_inp)
        attn_output = self._scaled_dot_product_attn(q, k, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(x.size(0), x.size(1), -1)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        return self.norm(residual + attn_output)

    def _compute_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        shape = (x.size(0), x.size(1), self.heads, self.head_dim)
        q = q.view(*shape).transpose(1, 2)
        k = k.view(*shape).transpose(1, 2)
        v = v.view(*shape).transpose(1, 2)
        if self.config.qk_l2_norm:
            q = F.normalize(q, dim=-1, eps=self.config.qk_norm_eps)
            k = F.normalize(k, dim=-1, eps=self.config.qk_norm_eps)
        return q, k, v

    def _scaled_dot_product_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        dropout_p = self.config.dropout if self.training else 0.0
        device_type = q.device.type
        if (
            device_type == "cuda"
            and torch.cuda.is_available()
            and hasattr(torch.backends, "cuda")
            and hasattr(torch.backends.cuda, "sdp_kernel")
        ):
            with torch.backends.cuda.sdp_kernel(  # type: ignore[attr-defined]
                enable_flash=self.config.use_flash,
                enable_mem_efficient=True,
                enable_math=not self.config.use_flash,
            ):
                return F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=dropout_p,
                    is_causal=self.config.causal,
                )
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=self.config.causal,
        )
