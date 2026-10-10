"""
GPU tests for dequant_matmul — compared against the CPU reference, not just shape-checked.

Every assertion here is a numerical comparison with
``inference.kernels.reference.dequant_matmul_reference``, which implements the same
contract in plain Python and is tested on every machine by ``test_kernel_reference.py``.

The old version of this file asserted only that the output had the right shape, which
would pass for a kernel returning zeros. It also skipped on
``torch.cuda.is_available()`` inside a ``try: import torch`` — and torch is in neither
requirements file, so the skip fired because torch was *absent* on every machine that ever
ran the suite. Between that and CI excluding the file by name, a compile error in the
kernel went unnoticed for eighty chapters (learning.md Ch. 85).

The skip reason below distinguishes the two causes, so a skip says which one it is.
"""

from __future__ import annotations

import random

import pytest

from inference.kernels.reference import dequant_matmul_reference, pack_int4

try:
    import torch  # type: ignore[import-untyped]

    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_CUDA = False

if not HAS_TORCH:
    SKIP_REASON = "torch is not installed (it is in neither requirements file)"
elif not HAS_CUDA:
    SKIP_REASON = "torch is installed but reports no CUDA device"
else:
    SKIP_REASON = ""

needs_gpu = pytest.mark.skipif(bool(SKIP_REASON), reason=SKIP_REASON)

# fp16 storage with an fp32 accumulator against a float64 reference. The tolerance scales
# with K because every one of the K products contributes its own rounding.
RELATIVE_TOLERANCE = 2e-2
ABSOLUTE_TOLERANCE = 5e-2


def _problem(
    m: int, k: int, n: int, group: int, seed: int = 0
) -> tuple[list[list[float]], list[list[int]], list[list[float]], list[list[int]]]:
    """A deterministic (a, w_packed, scales, zeros) set in the reference's list form."""
    rng = random.Random(seed)
    groups = k // group

    a = [[rng.uniform(-1.0, 1.0) for _ in range(k)] for _ in range(m)]
    weights = [[rng.randint(0, 15) for _ in range(n)] for _ in range(k)]
    scales = [[rng.uniform(0.01, 0.2) for _ in range(n)] for _ in range(groups)]
    zeros = [[rng.randint(0, 15) for _ in range(n)] for _ in range(groups)]
    return a, pack_int4(weights), scales, zeros


def _run_kernel(
    a: list[list[float]],
    w_packed: list[list[int]],
    scales: list[list[float]],
    zeros: list[list[int]],
    group: int,
):
    from inference.kernels.dequant_matmul import dequant_matmul

    # Packed bytes are built as 0..255 but the kernel's tensor is int8, so values above
    # 127 have to arrive as their negative two's-complement form. The kernel masks before
    # shifting precisely so that this is lossless.
    signed = [[v - 256 if v > 127 else v for v in row] for row in w_packed]

    return dequant_matmul(
        torch.tensor(a, dtype=torch.float16, device="cuda"),
        torch.tensor(signed, dtype=torch.int8, device="cuda"),
        torch.tensor(scales, dtype=torch.float16, device="cuda"),
        torch.tensor(zeros, dtype=torch.int8, device="cuda"),
        group_size=group,
    )


def _assert_matches_reference(m: int, k: int, n: int, group: int, seed: int = 0) -> None:
    a, w_packed, scales, zeros = _problem(m, k, n, group, seed)
    expected = dequant_matmul_reference(a, w_packed, scales, zeros, group_size=group)
    actual = _run_kernel(a, w_packed, scales, zeros, group)

    assert actual.shape == (m, n), f"expected ({m}, {n}), got {tuple(actual.shape)}"
    assert actual.dtype == torch.float16

    got = actual.float().cpu().tolist()
    for row in range(m):
        for col in range(n):
            assert got[row][col] == pytest.approx(
                expected[row][col], rel=RELATIVE_TOLERANCE, abs=ABSOLUTE_TOLERANCE
            ), f"mismatch at ({row}, {col}) for M={m} K={k} N={n} group={group}"


# --- the shapes the autotuner actually has to handle --------------------------


@needs_gpu
def test_matches_the_reference_on_a_single_tile() -> None:
    _assert_matches_reference(m=4, k=64, n=32, group=32)


@needs_gpu
def test_matches_the_reference_across_several_tiles() -> None:
    # Larger than any BLOCK_M/BLOCK_N in the autotune table, so the grid is more than one
    # program in both dimensions.
    _assert_matches_reference(m=96, k=256, n=192, group=128)


@needs_gpu
def test_matches_the_reference_on_ragged_edges() -> None:
    # None of these divide the block sizes, which is what the tl.load masks are for.
    _assert_matches_reference(m=7, k=128, n=37, group=64)


@needs_gpu
def test_a_block_wider_than_a_group_still_uses_the_right_scales() -> None:
    """
    The autotuner offers BLOCK_K=64; a group_size of 16 means four groups inside one
    chunk. Reading a single group's scale per chunk — the original behaviour — gets three
    quarters of every tile wrong, and only a comparison against the reference catches it.
    """
    _assert_matches_reference(m=8, k=128, n=64, group=16)


@needs_gpu
def test_zero_points_are_honoured() -> None:
    # Weights all 4 with a zero-point of 4 dequantize to exactly zero, so the output must
    # be zero regardless of the activations. A kernel that skipped the subtraction, or
    # applied it after scaling, fails this without any tolerance argument.
    k, n, group = 64, 32, 32
    a = [[0.5] * k for _ in range(4)]
    packed = pack_int4([[4] * n for _ in range(k)])
    scales = [[0.1] * n for _ in range(k // group)]
    zeros = [[4] * n for _ in range(k // group)]

    actual = _run_kernel(a, packed, scales, zeros, group)
    assert actual.float().abs().max().item() == pytest.approx(0.0, abs=1e-3)
