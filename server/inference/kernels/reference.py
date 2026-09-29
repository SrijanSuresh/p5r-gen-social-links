"""
What the Triton kernel is supposed to compute, in plain Python.

The kernel needs CUDA. The contract it implements does not, and keeping the two apart is
what makes the contract testable: this module runs anywhere, in CI, with no dependency the
project does not already have — no numpy, no torch, no GPU.

It exists because `dequant_matmul_kernel` was wrong for eighty chapters behind a test that
only checked the shape of its output, was skipped whenever torch was absent (which is
always, here), and was excluded from CI by name. A reference implementation turns "low
nibble = col k, high nibble = col k+1" from a docstring into something that can disagree
with the kernel and say so. See learning.md Ch. 85.

Deliberately unoptimised. Triple-nested loops over Python lists are the point: there is
nothing here to be subtly clever about, so when the two disagree the kernel is wrong.

LAYOUT (mirrors dequant_matmul.py)
  a        : [M, K]              float activations, row-major
  w_packed : [K // 2, N]         one byte per (k-pair, column)
                                   bits 3..0 -> weight for even k = 2 * row
                                   bits 7..4 -> weight for odd  k = 2 * row + 1
  scales   : [K // group, N]     float, one per (group, column)
  zeros    : [K // group, N]     int,   one per (group, column)
  out      : [M, N]              float

DEQUANTIZE
  w[k][n] = (nibble(k, n) - zeros[k // group][n]) * scales[k // group][n]
"""

from __future__ import annotations

NIBBLE_MAX = 0x0F
"""Largest value an unsigned 4-bit weight can hold."""


class PackingError(ValueError):
    """A tensor's shape or values do not match the packed int4 layout."""


def pack_int4(weights: list[list[int]]) -> list[list[int]]:
    """
    Pack a ``[K, N]`` grid of 4-bit weights into ``[K // 2, N]`` bytes.

    Row ``k`` and row ``k + 1`` share a byte: the even row takes the low nibble and the odd
    row the high nibble. That pairing runs down K rather than across N because K is the
    reduction axis — the kernel walks it in ``BLOCK_K`` chunks and wants both halves of a
    byte inside the same chunk.

    Raises:
        PackingError: K is odd, rows are ragged, or a weight is outside 0..15.
    """
    if not weights:
        raise PackingError("no weights to pack")

    k_dim = len(weights)
    if k_dim % 2 != 0:
        raise PackingError(f"K must be even to pack two per byte, got {k_dim}")

    n_dim = len(weights[0])
    for k, row in enumerate(weights):
        if len(row) != n_dim:
            raise PackingError(f"row {k} has width {len(row)}, expected {n_dim}")
        for n, value in enumerate(row):
            if not 0 <= value <= NIBBLE_MAX:
                raise PackingError(
                    f"weight[{k}][{n}] = {value} does not fit in 4 bits (0..15)"
                )

    packed: list[list[int]] = []
    for pair in range(k_dim // 2):
        low  = weights[2 * pair]
        high = weights[2 * pair + 1]
        packed.append([low[n] | (high[n] << 4) for n in range(n_dim)])
    return packed


def unpack_int4(packed: list[list[int]], k_dim: int) -> list[list[int]]:
    """
    Recover the ``[K, N]`` grid that :func:`pack_int4` produced.

    ``k_dim`` is passed rather than inferred as ``2 * len(packed)`` so that the round trip
    is checked rather than assumed: the kernel derives K the same way, from the packed
    row count, and a mismatch there is the kind of thing that reads as plausible noise in
    an output tensor.

    Raises:
        PackingError: ``k_dim`` disagrees with the packed row count.
    """
    if k_dim != 2 * len(packed):
        raise PackingError(
            f"K={k_dim} does not match {len(packed)} packed rows (expected {2 * len(packed)})"
        )

    weights: list[list[int]] = []
    for row in packed:
        # A byte may arrive as a signed value; mask before shifting so the high nibble of
        # 0x9F is 9 rather than a sign-extended -7.
        weights.append([value & NIBBLE_MAX for value in row])
        weights.append([(value >> 4) & NIBBLE_MAX for value in row])
    return weights


def dequantize(
    packed: list[list[int]],
    scales: list[list[float]],
    zeros: list[list[int]],
    k_dim: int,
    group_size: int,
) -> list[list[float]]:
    """
    Expand packed int4 weights into the ``[K, N]`` float matrix the matmul consumes.

    Scales and zero-points are per ``(group, column)``, and a group is ``group_size``
    consecutive values of K. Indexing them by ``k // group_size`` per element — rather
    than once per tile — is the whole of the kernel's second defect: a ``BLOCK_K`` wider
    than ``group_size`` spans two groups and must not dequantize both with one scale.

    Raises:
        PackingError: K is not a multiple of ``group_size``, or a table is the wrong size.
    """
    if group_size <= 0:
        raise PackingError(f"group_size must be positive, got {group_size}")
    if k_dim % group_size != 0:
        raise PackingError(f"K={k_dim} is not a multiple of group_size={group_size}")

    groups = k_dim // group_size
    if len(scales) != groups or len(zeros) != groups:
        raise PackingError(
            f"expected {groups} rows of scales and zeros, got {len(scales)} and {len(zeros)}"
        )

    quantized = unpack_int4(packed, k_dim)
    n_dim = len(quantized[0])

    out: list[list[float]] = []
    for k in range(k_dim):
        group = k // group_size
        out.append([
            (quantized[k][n] - zeros[group][n]) * scales[group][n]
            for n in range(n_dim)
        ])
    return out


def dequant_matmul_reference(
    a: list[list[float]],
    w_packed: list[list[int]],
    scales: list[list[float]],
    zeros: list[list[int]],
    group_size: int = 128,
) -> list[list[float]]:
    """
    ``a @ dequantize(w_packed)`` — the answer the Triton kernel must reproduce.

    Accumulates in Python floats, which are doubles. The kernel accumulates in fp32 and
    stores fp16, so a GPU comparison against this needs a tolerance; what it does not need
    is an opinion about the packing order, the group indexing or the dequantize formula,
    all of which are fixed here.

    Raises:
        PackingError: the activation width disagrees with the packed weight height.
    """
    if not a:
        raise PackingError("no activations")

    m_dim = len(a)
    k_dim = len(a[0])
    for row_index, row in enumerate(a):
        if len(row) != k_dim:
            raise PackingError(f"activation row {row_index} has width {len(row)}, expected {k_dim}")

    if k_dim != 2 * len(w_packed):
        raise PackingError(
            f"activations have K={k_dim} but packed weights hold K={2 * len(w_packed)}"
        )

    weights = dequantize(w_packed, scales, zeros, k_dim, group_size)
    n_dim = len(weights[0])

    out: list[list[float]] = []
    for m in range(m_dim):
        row_a = a[m]
        out.append([
            sum(row_a[k] * weights[k][n] for k in range(k_dim))
            for n in range(n_dim)
        ])
    return out
