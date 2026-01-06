import torch

from nested_learning.backbones import AttentionConfig, SelfAttention


def test_self_attention_qk_l2_norm_unit_vectors() -> None:
    attn = SelfAttention(AttentionConfig(dim=16, heads=4, qk_l2_norm=True, use_flash=False))
    x = torch.randn(2, 5, 16)
    q, k, _v = attn._compute_qkv(x)
    q_norm = q.norm(dim=-1)
    k_norm = k.norm(dim=-1)
    assert torch.allclose(q_norm, torch.ones_like(q_norm), atol=1e-4, rtol=1e-4)
    assert torch.allclose(k_norm, torch.ones_like(k_norm), atol=1e-4, rtol=1e-4)


def test_self_attention_local_conv_window_preserves_shape() -> None:
    attn = SelfAttention(AttentionConfig(dim=16, heads=4, local_conv_window=4, use_flash=False))
    assert attn.local_conv is not None
    assert attn.local_conv.kernel_size == (4,)
    x = torch.randn(2, 8, 16)
    out = attn(x)
    assert out.shape == x.shape
