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


def test_self_attention_local_conv_is_causal() -> None:
    torch.manual_seed(0)
    dim = 4
    attn = SelfAttention(
        AttentionConfig(dim=dim, heads=2, local_conv_window=4, use_flash=False, dropout=0.0)
    ).eval()
    assert attn.local_conv is not None
    with torch.no_grad():
        attn.local_conv.weight.fill_(1.0)
        eye = torch.eye(dim)
        attn.qkv.weight.zero_()
        attn.qkv.weight[:dim].copy_(eye)
        attn.qkv.weight[dim : 2 * dim].copy_(eye)
        attn.qkv.weight[2 * dim :].copy_(eye)
        attn.out_proj.weight.copy_(eye)
    x1 = torch.randn(1, 8, dim)
    x2 = x1.clone()
    x2[:, 4:, :] = torch.randn_like(x2[:, 4:, :])
    out1 = attn(x1)
    out2 = attn(x2)
    assert torch.allclose(out1[:, :4, :], out2[:, :4, :], atol=1e-5, rtol=1e-5)
