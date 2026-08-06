"""
Custom Triton kernel: 4-bit integer dequantize + matrix multiply.

WHY THIS EXISTS:
  Standard PyTorch cannot multiply float16 activations by int4 weights directly.
  auto-gptq''s default CUDA implementation works but is not tunable.
  This Triton kernel lets us control tile sizes (BLOCK_M/N/K) to maximise
  occupancy on the target GPU.

LAYOUT CONVENTIONS:
  W_packed : [K//2, N] int8  — two int4 values packed per byte (low nibble = col k,
                                                                   high nibble = col k+1)
  scales   : [K//group_size, N] float16 — per-group scale factors
  zeros    : [K//group_size, N] int8    — per-group zero-points
  A        : [M, K] float16            — activation matrix
  Out      : [M, N] float32            — accumulator (cast to fp16 before return)
"""

from __future__ import annotations

import triton                           # type: ignore[import-untyped]
import triton.language as tl            # type: ignore[import-untyped]
import torch


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N":  64, "BLOCK_K": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N":  64, "BLOCK_K": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N":  64, "BLOCK_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def dequant_matmul_kernel(
    # Pointers to tensors in GPU global memory
    A_ptr, W_packed_ptr, scales_ptr, zeros_ptr, Out_ptr,
    # Matrix dimensions
    M, N, K,
    # Quantisation group size (number of K elements sharing one scale/zero)
    group_size: tl.constexpr,
    # Tile sizes — tuned per GPU via triton.autotune in production
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Each Triton program handles one (BLOCK_M × BLOCK_N) output tile.

    BLOCK POINTER MATH:
      pid_m, pid_n  — which tile this program owns in the M and N dimensions
      row_start     — first row index this program processes (pid_m * BLOCK_M)
      col_start     — first col index (pid_n * BLOCK_N)

    MASK LOGIC:
      Near the right/bottom edges of the matrix the tile extends past the
      valid region. We pass a boolean mask to tl.load so out-of-bounds
      lanes read 0 instead of garbage memory.
    """
    pid_m = tl.program_id(0)   # which row-tile
    pid_n = tl.program_id(1)   # which col-tile

    # Row/col offset vectors for this tile (shape: [BLOCK_M] and [BLOCK_N])
    row_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # absolute row indices
    col_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # absolute col indices

    # fp32 accumulator for this output tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over the K dimension in BLOCK_K-sized chunks
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K] absolute k indices

        # ── Load activation tile A[row_offs, k_offs] ────────────────────────
        # Boundary mask: True where row < M AND k < K
        a_mask = (row_offs[:, None] < M) & (k_offs[None, :] < K)
        # Pointer arithmetic: A is row-major [M, K], stride K between rows
        a_ptrs = A_ptr + row_offs[:, None] * K + k_offs[None, :]
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float16)

        # ── Load packed int4 weights W_packed[k_offs//2, col_offs] ──────────
        # Two int4 values packed per byte along the K dim → address is k//2
        wp_row  = k_offs // 2                                     # [BLOCK_K]
        wp_mask = (wp_row[:, None] < K // 2) & (col_offs[None, :] < N)
        wp_ptrs = W_packed_ptr + wp_row[:, None] * N + col_offs[None, :]
        w_packed = tl.load(wp_ptrs, mask=wp_mask, other=0).to(tl.int8)

        # ── Unpack two int4 values from each byte ────────────────────────────
        # Low nibble  → weights for even k indices
        # High nibble → weights for odd  k indices
        w_lo = (w_packed & 0x0F).to(tl.float16)   # bits [3:0]
        w_hi = ((w_packed >> 4) & 0x0F).to(tl.float16)  # bits [7:4]

        # ── Load scale and zero-point for this K group ───────────────────────
        group_id = k_start // group_size
        s_ptrs = scales_ptr + group_id * N + col_offs
        z_ptrs = zeros_ptr  + group_id * N + col_offs
        s = tl.load(s_ptrs, mask=col_offs < N, other=1.0).to(tl.float16)
        z = tl.load(z_ptrs, mask=col_offs < N, other=0  ).to(tl.float16)

        # ── Dequantize: float_weight = (int4 - zero) * scale ────────────────
        w_lo_f = (w_lo - z[None, :]) * s[None, :]
        w_hi_f = (w_hi - z[None, :]) * s[None, :]

        # ── Accumulate: reconstruct full BLOCK_K × BLOCK_N contribution ─────
        # Interleave lo/hi to recover the original K ordering
        acc += tl.dot(a_tile[::2, :],  w_lo_f)   # even k rows × lo weights
        acc += tl.dot(a_tile[1::2, :], w_hi_f)   # odd  k rows × hi weights

    # ── Write output tile ────────────────────────────────────────────────────
    out_mask = (row_offs[:, None] < M) & (col_offs[None, :] < N)
    out_ptrs = Out_ptr + row_offs[:, None] * N + col_offs[None, :]
    tl.store(out_ptrs, acc.to(tl.float16), mask=out_mask)


def dequant_matmul(
    a: torch.Tensor,
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Python launch wrapper for the Triton kernel."""
    M, K = a.shape
    _, N = w_packed.shape[0] * 2, w_packed.shape[1]  # unpack K from packed dim

    out = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # Grid is a lambda so autotune can inject the winning BLOCK_M/N values
    grid = lambda meta: (  # noqa: E731
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    dequant_matmul_kernel[grid](
        a, w_packed, scales, zeros, out,
        M, N, K,
        group_size=group_size,
    )
    return out