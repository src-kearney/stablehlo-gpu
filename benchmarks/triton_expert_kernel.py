#!/usr/bin/env python3
"""
benchmarks/triton_expert_kernel.py

Parameterized Triton kernel for one Mixtral MoE expert's FFN forward pass.

Computation per expert (SwiGLU):
    gate   = x @ w_gate              [T, D] × [D, F] → [T, F]
    up     = x @ w_up                [T, D] × [D, F] → [T, F]
    hidden = silu(gate) * up         [T, F]
    out    = hidden @ w_down         [T, F] × [F, D] → [T, D]

BLOCK_M is a tl.constexpr that varies by shape bucket, letting Triton emit a
distinct compiled kernel per bucket — matching the per-expert dispatch that the
remora ExpertOutliningPass produces in @main.

Usage (requires CUDA, tested on RTX 4080):
    python3 benchmarks/triton_expert_kernel.py
"""

import sys
import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Shape-bucket configs
# Keys are (lo, hi) half-open intervals: lo <= num_tokens < hi
# ---------------------------------------------------------------------------
BUCKET_CONFIGS: dict[tuple[int, int], dict] = {
    # Shared memory per launch = (num_stages-1)*(BLOCK_M*BLOCK_K + 2*BLOCK_K*BLOCK_N)*2 bytes
    # RTX 4090 limit: 101,376 bytes per block.
    (0,    50):  {"BLOCK_M":  16, "BLOCK_N":  64, "BLOCK_K": 32, "num_stages": 3},  #  3,072 B
    (50,  150):  {"BLOCK_M":  32, "BLOCK_N": 128, "BLOCK_K": 64, "num_stages": 3},  # 24,576 B
    (150, 300):  {"BLOCK_M":  64, "BLOCK_N": 128, "BLOCK_K": 64, "num_stages": 3},  # 81,920 B
    (300, 600):  {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "num_stages": 2},  # 81,920 B ← was 163,840 at ns=3
    (600, 9999): {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "num_stages": 2},  # 65,536 B
}

# num_warps paired to BLOCK_M — larger tiles benefit from more warps
_BLOCK_M_TO_NUM_WARPS = {16: 2, 32: 4, 64: 4, 128: 8, 256: 8}


def select_bucket(num_tokens: int) -> dict:
    """Return the bucket config for `num_tokens` (lo <= T < hi)."""
    for (lo, hi), cfg in BUCKET_CONFIGS.items():
        if lo <= num_tokens < hi:
            return cfg
    return list(BUCKET_CONFIGS.values())[-1]


# ---------------------------------------------------------------------------
# Triton kernel 1: fused gate + up projections + SwiGLU
#
#   Out[M, N] = silu(X[M, K] @ W_gate[K, N]) * (X[M, K] @ W_up[K, N])
#
# Both projections share the same LHS tile, halving HBM reads for X.
# Accumulators are float32; result is cast to float16 on store.
# ---------------------------------------------------------------------------
@triton.jit
def _gate_up_silu_kernel(
    X_ptr, Wg_ptr, Wu_ptr, Out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_gk, stride_gn,
    stride_uk, stride_un,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # row indices  [BLOCK_M]
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # col indices  [BLOCK_N]
    rk = tl.arange(0, BLOCK_K)                      # inner loop   [BLOCK_K]

    mask_m = rm < M
    mask_n = rn < N

    # Base pointers for the first K-chunk
    X_ptrs  = X_ptr  + rm[:, None] * stride_xm + rk[None, :] * stride_xk
    Wg_ptrs = Wg_ptr + rk[:, None] * stride_gk + rn[None, :] * stride_gn
    Wu_ptrs = Wu_ptr + rk[:, None] * stride_uk + rn[None, :] * stride_un

    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc   = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        mask_k = rk + k < K

        x  = tl.load(X_ptrs,  mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        wg = tl.load(Wg_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        wu = tl.load(Wu_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        gate_acc = tl.dot(x, wg, acc=gate_acc, out_dtype=tl.float32)
        up_acc   = tl.dot(x, wu, acc=up_acc,   out_dtype=tl.float32)

        X_ptrs  += BLOCK_K * stride_xk
        Wg_ptrs += BLOCK_K * stride_gk
        Wu_ptrs += BLOCK_K * stride_uk

    # SwiGLU: silu(gate) * up,  silu(x) = x * sigmoid(x)
    hidden = gate_acc * tl.sigmoid(gate_acc) * up_acc

    Out_ptrs = Out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on
    tl.store(Out_ptrs, hidden.to(tl.float16),
             mask=mask_m[:, None] & mask_n[None, :])


# ---------------------------------------------------------------------------
# Triton kernel 2: down projection
#
#   Out[M, N] = Hidden[M, K] @ W_down[K, N]
# ---------------------------------------------------------------------------
@triton.jit
def _down_kernel(
    H_ptr, Wd_ptr, Out_ptr,
    M, N, K,
    stride_hm, stride_hk,
    stride_dk, stride_dn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    mask_m = rm < M
    mask_n = rn < N

    H_ptrs  = H_ptr  + rm[:, None] * stride_hm + rk[None, :] * stride_hk
    Wd_ptrs = Wd_ptr + rk[:, None] * stride_dk + rn[None, :] * stride_dn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        mask_k = rk + k < K

        h  = tl.load(H_ptrs,  mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        wd = tl.load(Wd_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc = tl.dot(h, wd, acc=acc, out_dtype=tl.float32)

        H_ptrs  += BLOCK_K * stride_hk
        Wd_ptrs += BLOCK_K * stride_dk

    Out_ptrs = Out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on
    tl.store(Out_ptrs, acc.to(tl.float16),
             mask=mask_m[:, None] & mask_n[None, :])


# ---------------------------------------------------------------------------
# Python-level per-expert launch
# ---------------------------------------------------------------------------

def _expert_ffn_with_cfg(
    x:         torch.Tensor,   # [T, D] float16
    w_gate:    torch.Tensor,   # [D, F] float16  (w_gate[e] from outlined @main)
    w_up:      torch.Tensor,   # [D, F] float16
    w_down:    torch.Tensor,   # [F, D] float16
    BLOCK_M:   int,
    BLOCK_N:   int,
    BLOCK_K:   int,
    num_warps: int,
    num_stages: int = 3,
) -> torch.Tensor:             # [T, D] float16
    T, D = x.shape
    F    = w_gate.shape[1]
    assert w_gate.shape == (D, F)
    assert w_up.shape   == (D, F)
    assert w_down.shape == (F, D)

    # Gate + up + SwiGLU → hidden [T, F]
    hidden = torch.empty((T, F), device=x.device, dtype=x.dtype)
    grid_gu = (triton.cdiv(T, BLOCK_M), triton.cdiv(F, BLOCK_N))
    _gate_up_silu_kernel[grid_gu](
        x,      w_gate, w_up, hidden,
        T, F, D,
        x.stride(0),      x.stride(1),
        w_gate.stride(0), w_gate.stride(1),
        w_up.stride(0),   w_up.stride(1),
        hidden.stride(0), hidden.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )

    # Down projection → out [T, D]
    out = torch.empty((T, D), device=x.device, dtype=x.dtype)
    grid_d = (triton.cdiv(T, BLOCK_M), triton.cdiv(D, BLOCK_N))
    _down_kernel[grid_d](
        hidden, w_down, out,
        T, D, F,
        hidden.stride(0), hidden.stride(1),
        w_down.stride(0), w_down.stride(1),
        out.stride(0),    out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


# ---------------------------------------------------------------------------
# Dispatch table
#
# Keyed by the bucket tuple (lo, hi).  Each entry is a callable that
# launches the right pre-compiled kernel specialization.
# ---------------------------------------------------------------------------

def _make_dispatch_fn(BLOCK_M: int, BLOCK_N: int, BLOCK_K: int, num_stages: int = 3):
    """Closure capturing compile-time constants for one bucket."""
    nw = _BLOCK_M_TO_NUM_WARPS[BLOCK_M]

    def dispatch(x, w_gate, w_up, w_down):
        return _expert_ffn_with_cfg(x, w_gate, w_up, w_down,
                                     BLOCK_M, BLOCK_N, BLOCK_K, nw, num_stages)
    return dispatch


DISPATCH_TABLE: dict[tuple[int, int], callable] = {
    bucket: _make_dispatch_fn(**cfg)
    for bucket, cfg in BUCKET_CONFIGS.items()
}


def _get_dispatch_fn(num_tokens: int):
    """Return the pre-compiled dispatch function for `num_tokens`."""
    for (lo, hi), fn in DISPATCH_TABLE.items():
        if lo <= num_tokens < hi:
            return fn
    return list(DISPATCH_TABLE.values())[-1]


# ---------------------------------------------------------------------------
# run_remora_outlined
#
# Simulates what the compiled @main produced by ExpertOutliningPass does:
# tokens are already sliced per expert; we look up the bucket and dispatch.
# ---------------------------------------------------------------------------

def run_remora_outlined(
    expert_tokens: list[torch.Tensor],   # len=E, each [T_e, D] float16
    w_gate:        torch.Tensor,         # [E, D, F] float16
    w_up:          torch.Tensor,         # [E, D, F] float16
    w_down:        torch.Tensor,         # [E, F, D] float16
    min_tokens:    int  = 0,             # skip experts with fewer tokens than this
    streams:       list = None,          # pre-created CUDA streams, one per expert
) -> list[torch.Tensor]:                 # len=E, each [T_e, D] float16
    """
    Dispatch each expert's tokens to the Triton kernel compiled for its
    token-count bucket.

    min_tokens: experts with 0 < T < min_tokens are skipped; their output is
    zeros of shape [T, D] so the downstream cat + index_add_ remains valid.
    Experts with T == 0 are always skipped.

    streams: when provided, each expert's kernels are launched on a separate
    CUDA stream so they execute concurrently on the GPU.  Pass a list of
    pre-created torch.cuda.Stream objects (one per expert) created once at
    startup — stream creation itself is expensive.  A final synchronize()
    ensures all streams are complete before returning.
    """
    D       = w_gate.shape[1]   # hidden dim — correct output dim per token
    E       = len(expert_tokens)
    outputs = [None] * E

    for e, x in enumerate(expert_tokens):
        T = x.shape[0]
        if T == 0 or T < min_tokens:
            outputs[e] = torch.zeros(T, D, dtype=x.dtype, device=x.device)
            continue
        if streams is not None:
            with torch.cuda.stream(streams[e]):
                fn = _get_dispatch_fn(T)
                outputs[e] = fn(x, w_gate[e], w_up[e], w_down[e])
        else:
            fn = _get_dispatch_fn(T)
            outputs[e] = fn(x, w_gate[e], w_up[e], w_down[e])

    if streams is not None:
        torch.cuda.synchronize()

    return outputs


# ---------------------------------------------------------------------------
# Kernel warmup — forces JIT compilation of every bucket at import time
# so the first real call is latency-free.
# ---------------------------------------------------------------------------

def warmup_all_buckets(
    hidden_dim:  int  = 4096,
    inter_dim:   int  = 14336,
    device:      str  = "cuda",
    use_streams: bool = False,
) -> None:
    """
    Drive one tiny forward pass per bucket to trigger Triton JIT compilation.
    Uses the minimum representable token count for each bucket.

    use_streams: if True, each bucket's warmup pass runs on a dedicated CUDA
    stream, ensuring the compiled kernels are cached for non-default streams.
    """
    print("Warming up Triton kernels for all buckets...")
    dtype        = torch.float16
    bucket_items = list(BUCKET_CONFIGS.items())
    streams      = (
        [torch.cuda.Stream() for _ in bucket_items] if use_streams
        else [None] * len(bucket_items)
    )

    for i, ((lo, hi), cfg) in enumerate(bucket_items):
        T      = max(lo, 1)
        x      = torch.zeros(T, hidden_dim, dtype=dtype, device=device)
        w_gate = torch.zeros(hidden_dim, inter_dim, dtype=dtype, device=device)
        w_up   = torch.zeros(hidden_dim, inter_dim, dtype=dtype, device=device)
        w_down = torch.zeros(inter_dim, hidden_dim, dtype=dtype, device=device)
        fn     = DISPATCH_TABLE[(lo, hi)]
        if streams[i] is not None:
            with torch.cuda.stream(streams[i]):
                fn(x, w_gate, w_up, w_down)
        else:
            fn(x, w_gate, w_up, w_down)
        torch.cuda.synchronize()
        stream_tag = f"  stream {i}" if use_streams else ""
        print(f"  bucket {lo:4d}–{hi:4d}  BLOCK_M={cfg['BLOCK_M']:3d}  compiled ✓{stream_tag}")
    print()


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------

def _pytorch_expert_ffn_ref(
    x:      torch.Tensor,  # [T, D]
    w_gate: torch.Tensor,  # [D, F]
    w_up:   torch.Tensor,  # [D, F]
    w_down: torch.Tensor,  # [F, D]
) -> torch.Tensor:         # [T, D]
    """Naive per-expert FFN in PyTorch (float32 accumulation)."""
    x_f  = x.float()
    gate = x_f @ w_gate.float()
    up   = x_f @ w_up.float()
    hidden = torch.nn.functional.silu(gate) * up
    return (hidden @ w_down.float()).to(x.dtype)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("No CUDA device found — this benchmark requires a GPU.")
        sys.exit(1)

    device = "cuda"
    dtype  = torch.float16

    # Mixtral-8x7B dimensions
    E      = 8
    HIDDEN = 4096
    INTER  = 14336

    # Skewed distribution: exercises buckets 16, 32, 64, 128 (all but 256)
    TOKENS_PER_EXPERT = [300, 200, 100, 50, 20, 10, 5, 2]

    print("=" * 68)
    print("remora  Triton expert kernel  —  correctness check")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Mixtral dims: hidden={HIDDEN}, intermediate={INTER}, experts={E}")
    print(f"Token distribution: {TOKENS_PER_EXPERT}")
    print("=" * 68)
    print()

    # Pre-compile all bucket kernels
    warmup_all_buckets(hidden_dim=HIDDEN, inter_dim=INTER, device=device)

    # Random weights (small scale avoids fp16 overflow in intermediate)
    torch.manual_seed(42)
    scale = 0.02
    w_gate = (torch.randn(E, HIDDEN, INTER, dtype=dtype, device=device) * scale)
    w_up   = (torch.randn(E, HIDDEN, INTER, dtype=dtype, device=device) * scale)
    w_down = (torch.randn(E, INTER,  HIDDEN, dtype=dtype, device=device) * scale)

    # Per-expert token tensors (simulating post-dispatch from @main)
    expert_tokens = [
        torch.randn(T, HIDDEN, dtype=dtype, device=device) * 0.1
        for T in TOKENS_PER_EXPERT
    ]

    # ---- Triton path -------------------------------------------------------
    torch.cuda.synchronize()
    triton_outs = run_remora_outlined(expert_tokens, w_gate, w_up, w_down)
    torch.cuda.synchronize()

    # ---- PyTorch reference -------------------------------------------------
    ref_outs = [
        _pytorch_expert_ffn_ref(
            expert_tokens[e], w_gate[e], w_up[e], w_down[e]
        )
        for e in range(E)
    ]

    # ---- Compare -----------------------------------------------------------
    print(f"{'expert':>6}  {'tokens':>6}  {'bucket BLOCK_M':>14}  "
          f"{'max |err|':>10}  {'mean |err|':>10}  {'status':>6}")
    print("-" * 68)

    all_pass = True
    atol = 1e-3

    for e in range(E):
        T   = TOKENS_PER_EXPERT[e]
        cfg = select_bucket(T)
        bm  = cfg["BLOCK_M"]
        tri = triton_outs[e].float()
        ref = ref_outs[e].float()

        max_err  = (tri - ref).abs().max().item()
        mean_err = (tri - ref).abs().mean().item()
        ok       = max_err < atol
        status   = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False

        print(f"  {e:4d}  {T:6d}  {bm:14d}  {max_err:10.2e}  {mean_err:10.2e}  {status:>6}")

    print()
    if all_pass:
        print(f"ALL PASS  (atol={atol})")
    else:
        print(f"FAILED — some experts exceeded atol={atol}")
        sys.exit(1)
