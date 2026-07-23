"""Shape and dtype test for dequant_matmul — runs without a GPU (CPU tensors)."""

import pytest

try:
    import torch  # type: ignore[import-untyped]
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_dequant_matmul_output_shape() -> None:
    """Verifies the kernel launch wrapper produces correct output shape."""
    from inference.kernels.dequant_matmul import dequant_matmul

    M, K, N, GROUP = 4, 64, 32, 32
    a        = torch.randn(M, K, dtype=torch.float16)
    w_packed = torch.randint(0, 255, (K // 2, N), dtype=torch.int8)
    scales   = torch.randn(K // GROUP, N, dtype=torch.float16)
    zeros    = torch.zeros(K // GROUP, N, dtype=torch.int8)

    out = dequant_matmul(a, w_packed, scales, zeros, group_size=GROUP)
    assert out.shape == (M, N), f"Expected ({M}, {N}), got {out.shape}"
    assert out.dtype == torch.float16