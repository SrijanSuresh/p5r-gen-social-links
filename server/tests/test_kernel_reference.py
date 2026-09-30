"""
The int4 packing contract, tested on the CPU (learning.md Ch. 85).

These run everywhere — no GPU, no torch, no numpy. That is the point: the Triton kernel
had a compile error for eighty chapters behind a shape-only test that skipped whenever
torch was missing and was excluded from CI by name. The semantics it implements are
testable without any of that, and this is where they are pinned.
"""

from __future__ import annotations

import pytest

from inference.kernels.reference import (
    PackingError,
    check_shapes,
    dequant_matmul_reference,
    dequantize,
    pack_int4,
    unpack_int4,
)


# --- the nibble order ---------------------------------------------------------


def test_even_k_takes_the_low_nibble_and_odd_k_the_high_one() -> None:
    # One column, two rows: the whole convention in a single byte.
    packed = pack_int4([[0x3], [0xA]])
    assert packed == [[0xA3]], "expected low=3 high=A, i.e. 0xA3"


def test_packing_round_trips() -> None:
    weights = [
        [0x0, 0x1, 0xF],
        [0xF, 0xE, 0x0],
        [0x7, 0x8, 0x9],
        [0x1, 0x2, 0x3],
    ]
    assert unpack_int4(pack_int4(weights), k_dim=4) == weights


def test_a_high_nibble_is_never_sign_extended() -> None:
    # 0x9F read as a signed byte is -97. Masking before the shift is what keeps the high
    # nibble at 9 instead of turning it into a negative weight.
    assert unpack_int4([[0x9F]], k_dim=2) == [[0xF], [0x9]]
    assert unpack_int4([[-0x61]], k_dim=2) == [[0xF], [0x9]]


def test_packing_rejects_what_cannot_be_packed() -> None:
    with pytest.raises(PackingError, match="even"):
        pack_int4([[1], [2], [3]])            # odd K
    with pytest.raises(PackingError, match="4 bits"):
        pack_int4([[16], [0]])                # does not fit a nibble
    with pytest.raises(PackingError, match="width"):
        pack_int4([[1, 2], [3]])              # ragged
    with pytest.raises(PackingError, match="packed rows"):
        unpack_int4([[0x12]], k_dim=4)        # K disagrees with the row count


# --- group indexing, which is the kernel's second defect ----------------------


def test_each_group_uses_its_own_scale() -> None:
    """
    Four values of K, two groups of two. If a single scale were applied to the whole
    reduction — which is what `group_id = k_start // group_size` does for a BLOCK_K wider
    than a group — rows 2 and 3 would come back scaled by 1.0 instead of 10.0.
    """
    packed = pack_int4([[1], [2], [3], [4]])
    scales = [[1.0], [10.0]]
    zeros = [[0], [0]]

    assert dequantize(packed, scales, zeros, k_dim=4, group_size=2) == [
        [1.0], [2.0], [30.0], [40.0],
    ]


def test_zero_points_are_subtracted_before_scaling() -> None:
    packed = pack_int4([[8], [10]])
    assert dequantize(packed, [[2.0]], [[8]], k_dim=2, group_size=2) == [[0.0], [4.0]]


def test_dequantize_rejects_a_k_that_does_not_divide_into_groups() -> None:
    packed = pack_int4([[1], [2], [3], [4]])
    with pytest.raises(PackingError, match="multiple of group_size"):
        dequantize(packed, [[1.0]], [[0]], k_dim=4, group_size=3)
    with pytest.raises(PackingError, match="positive"):
        dequantize(packed, [[1.0]], [[0]], k_dim=4, group_size=0)
    with pytest.raises(PackingError, match="rows of scales"):
        dequantize(packed, [[1.0]], [[0]], k_dim=4, group_size=2)   # needs 2 rows


# --- the matmul ---------------------------------------------------------------


def test_identity_weights_pass_the_activations_through() -> None:
    # dequantized weights become a 2x2 identity: scale 1, zero 0, nibbles 1 and 0.
    packed = pack_int4([[1, 0], [0, 1]])
    out = dequant_matmul_reference(
        a=[[3.0, 5.0], [7.0, 11.0]],
        w_packed=packed,
        scales=[[1.0, 1.0]],
        zeros=[[0, 0]],
        group_size=2,
    )
    assert out == [[3.0, 5.0], [7.0, 11.0]]


def test_a_worked_example_by_hand() -> None:
    # a is 1x2, weights are 2x2. Nibbles: k=0 -> [2, 3], k=1 -> [4, 5].
    # Scale 2.0, zero 1, so weights dequantize to [[2, 4], [6, 8]].
    # out = [1, 10] @ [[2, 4], [6, 8]] = [2 + 60, 4 + 80] = [62, 84].
    out = dequant_matmul_reference(
        a=[[1.0, 10.0]],
        w_packed=pack_int4([[2, 3], [4, 5]]),
        scales=[[2.0, 2.0]],
        zeros=[[1, 1]],
        group_size=2,
    )
    assert out == [[62.0, 84.0]]


def test_the_reduction_runs_down_k_not_across_n() -> None:
    """
    The kernel's first defect sliced M where it meant to slice K. A non-square problem
    catches that: an answer built by reducing over the wrong axis cannot even have the
    right shape here.
    """
    a = [[1.0, 2.0, 3.0, 4.0]]                      # M=1, K=4
    packed = pack_int4([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])   # K=4, N=3
    out = dequant_matmul_reference(
        a, packed, scales=[[1.0, 1.0, 1.0]], zeros=[[0, 0, 0]], group_size=4
    )

    assert len(out) == 1 and len(out[0]) == 3
    assert out == [[10.0, 10.0, 10.0]]              # 1+2+3+4 in every column


def test_matmul_rejects_mismatched_shapes() -> None:
    with pytest.raises(PackingError, match="packed weights hold"):
        dequant_matmul_reference(
            a=[[1.0, 2.0, 3.0, 4.0]],
            w_packed=pack_int4([[1], [1]]),         # only K=2
            scales=[[1.0]],
            zeros=[[0]],
            group_size=2,
        )
    with pytest.raises(PackingError, match="no activations"):
        dequant_matmul_reference([], [[0]], [[1.0]], [[0]], group_size=2)


# --- the shape contract, shared with the Triton wrapper -----------------------
#
# The wrapper defers to check_shapes so these rules hold for both implementations and can
# be tested without torch, which is in neither requirements file.


def test_check_shapes_accepts_a_consistent_problem() -> None:
    check_shapes(k_dim=128, packed_rows=64, group_size=32)


def test_check_shapes_rejects_a_k_mismatch() -> None:
    with pytest.raises(PackingError, match="packed weights hold K=8"):
        check_shapes(k_dim=16, packed_rows=4, group_size=8)


def test_check_shapes_rejects_a_group_size_that_does_not_divide_k() -> None:
    with pytest.raises(PackingError, match="multiple of group_size"):
        check_shapes(k_dim=128, packed_rows=64, group_size=30)


def test_check_shapes_rejects_a_nonpositive_group_size() -> None:
    with pytest.raises(PackingError, match="positive"):
        check_shapes(k_dim=128, packed_rows=64, group_size=0)
    with pytest.raises(PackingError, match="positive"):
        check_shapes(k_dim=128, packed_rows=64, group_size=-32)
