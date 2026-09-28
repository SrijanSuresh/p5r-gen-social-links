
## Chapter 85: The Kernel Nobody Could Run

The Triton kernel has been in this repo since Phase 1. It is the reason the project
description says "custom Triton kernels for 4-bit quantized inference". It has an
autotuner with five configs, forty lines of commentary on block pointer math, and a test.

It cannot execute. Reading it closely — which is apparently the first time that has
happened — turns up two defects, and the first one means the kernel does not even compile.

### Defect 1: the slice is on the wrong axis

The intent is to split the K dimension into even and odd halves, because two int4 weights
share a byte: low nibble for even k, high nibble for odd k. The code:

```python
acc += tl.dot(a_tile[::2, :],  w_lo_f)   # even k rows x lo weights
acc += tl.dot(a_tile[1::2, :], w_hi_f)   # odd  k rows x hi weights
```

`a_tile` is loaded as `[BLOCK_M, BLOCK_K]` — the first axis indexes **rows of the
activation matrix**, which is M, and the second indexes K. So `a_tile[::2, :]` takes every
other *output row*, not every other k. The comment says "even k rows" and the code slices
M.

The shapes then do not close:

```
a_tile[::2, :]  @  w_lo_f      =  [BLOCK_M//2, BLOCK_K] @ [BLOCK_K, BLOCK_N]
                               =  [BLOCK_M//2, BLOCK_N]

acc                            =  [BLOCK_M,    BLOCK_N]
```

Accumulating a half-height tile into a full-height one is not a subtle numerical issue, it
is a shape error at compile time. With `BLOCK_M=16, BLOCK_K=32` from the first autotune
config, it is `[8, 64]` into `[16, 64]`.

### The fix removes the slicing rather than repairing it

The even/odd interleave does not need a slice at all. `wp_row = k_offs // 2` already loads
the right byte for every k in the tile — duplicated across adjacent k, which is exactly
what is wanted, because k and k+1 live in the same byte. All that is missing is choosing
*which nibble* per k, and that is a select on the parity of k:

```python
k_is_even = (k_offs % 2) == 0
w_q = tl.where(k_is_even[:, None], w_lo, w_hi)   # [BLOCK_K, BLOCK_N]
```

One `tl.dot` over the whole tile instead of two over halves, no strided slicing — which
Triton does not support on tile values anyway — and the K ordering is reconstructed by
construction rather than by interleaving two partial products.

### Defect 2: one scale for a tile that may span two groups

```python
group_id = k_start // group_size
s_ptrs = scales_ptr + group_id * N + col_offs
```

This picks a single quantization group for the entire `BLOCK_K` chunk. That is only correct
when a chunk never crosses a group boundary — and the autotuner explicitly offers
`BLOCK_K=64`, while the test passes `group_size=32`. Under that combination every tile
spans two groups and dequantizes both halves with the first group's scale and zero-point.

Scales are per `(group, column)`, so the honest load is per-k:

```python
g = k_offs // group_size
s_ptrs = scales_ptr + g[:, None] * N + col_offs[None, :]   # [BLOCK_K, BLOCK_N]
```

which costs a wider load and makes the dequantize elementwise against the weight tile it
is scaling. The alternative — asserting `BLOCK_K <= group_size` in the wrapper — is cheaper
but encodes a constraint the autotuner is free to violate the next time somebody adds a
config.

### Why three layers of safety net caught none of it

1. **The test only checks the shape of the output.** `assert out.shape == (M, N)` would pass
   for a kernel that returned zeros. Nothing compares against a known-correct answer, so no
   arithmetic is under test at all.
2. **The test is skipped.** It is guarded on `torch.cuda.is_available()`, inside a
   `try: import torch` that sets the flag false on ImportError. `torch` is not in
   `requirements.txt` or `requirements-dev.txt` — so on every machine that has ever run
   this suite, the guard was false because torch was *absent*, not because CUDA was. The
   skip reads as "no GPU here" and means "this was never imported".
3. **CI ignores the file by name.** `pytest tests/ --ignore=tests/test_kernel.py`.

Each layer is individually defensible. Together they make a compile error invisible for
eighty chapters.

### What a real test looks like without a GPU

The kernel needs CUDA. The *contract* does not. A reference implementation in plain Python
can define what the kernel is supposed to compute — the nibble packing order, the group
indexing, the dequantize formula — and be tested on any machine, in CI, with no
dependencies the project does not already have.

That splits the problem in two. The reference pins the semantics and runs everywhere; the
GPU test becomes a numerical comparison against it, which is a real assertion instead of a
shape check, and which fails loudly on the exact bugs above the first time it is run on the
4060.

It also puts the packing convention somewhere executable. "Low nibble = col k, high nibble
= col k+1" is currently a docstring, and a docstring cannot disagree with the code.
