"""Shape and dtype test for dequant_matmul — requires CUDA (Triton is GPU-only)."""

import pytest

try:
    import torch  # type: ignore[import-untyped]
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
def test_dequant_matmul_output_shape() -> None:
    """Verifies the autotuned kernel produces correct output shape on GPU."""
    from inference.kernels.dequant_matmul import dequant_matmul

    M, K, N, GROUP = 4, 64, 32, 32
    a        = torch.randn(M, K, dtype=torch.float16).cuda()
    w_packed = torch.randint(0, 255, (K // 2, N), dtype=torch.int8).cuda()
    scales   = torch.randn(K // GROUP, N, dtype=torch.float16).cuda()
    zeros    = torch.zeros(K // GROUP, N, dtype=torch.int8).cuda()

    out = dequant_matmul(a, w_packed, scales, zeros, group_size=GROUP)
    assert out.shape == (M, N), f"Expected ({M}, {N}), got {out.shape}"
    assert out.dtype == torch.float16
