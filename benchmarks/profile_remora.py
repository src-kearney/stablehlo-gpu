#!/usr/bin/env python3
"""
benchmarks/profile_remora.py

Profiles remora_forward at zipf_1.0 routing to find where time is going.
Uses torch.profiler with record_function annotations on each logical phase:
  - output_init       : torch.zeros for output accumulator
  - mask_creation     : ids_k == e  (one call per expert per top-k slot)
  - any_check         : masks[e].any()
  - gather            : hidden_states[masks[e]]
  - run_remora_outlined: all Triton kernel dispatches (both _gate_up_silu + _down)
  - scatter           : output[masks[e]] += wts_k * expert_out

Output
------
  stdout                           — top-20 ops by self CUDA time, phase summary,
                                     wall-time vs GPU-kernel-time breakdown
  benchmarks/results/remora_profile.json — Chrome trace (open in chrome://tracing)

Usage:
    python3 benchmarks/profile_remora.py
"""

import os
import sys
import time

import numpy as np
import torch
import torch.profiler

# ---------------------------------------------------------------------------
# Sibling-module imports
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from triton_expert_kernel import run_remora_outlined, warmup_all_buckets

# ---------------------------------------------------------------------------
# Dimensions — identical to sweep_skew.py
# ---------------------------------------------------------------------------
HIDDEN_DIM       = 4096
INTERMEDIATE_DIM = 14336
NUM_EXPERTS      = 8
TOP_K            = 2
NUM_TOKENS       = 512
DTYPE            = torch.float16

WARMUP_ITERS  = 10
PROFILE_ITERS =  5

# ---------------------------------------------------------------------------
# zipf_1.0 routing distribution
# ---------------------------------------------------------------------------

def _make_zipf_1(device: str):
    s           = 1.0
    num_tokens  = NUM_TOKENS
    num_experts = NUM_EXPERTS
    top_k       = TOP_K
    dtype       = DTYPE

    ranks = np.arange(1, num_experts + 1, dtype=np.float64)
    p     = 1.0 / (ranks ** s)
    p    /= p.sum()
    probs   = torch.tensor(p, dtype=torch.float32, device=device)
    probs   = probs.unsqueeze(0).expand(num_tokens, -1).contiguous()
    ids     = torch.multinomial(probs, num_samples=top_k, replacement=False).to(torch.int32)
    sel     = torch.gather(probs, 1, ids.long())
    weights = (sel / sel.sum(dim=-1, keepdim=True)).to(dtype)
    return ids.contiguous(), weights.contiguous()


# ---------------------------------------------------------------------------
# Annotated remora_forward
# Each logical phase is wrapped in record_function so it appears labelled
# in both the Chrome trace and the profiler key_averages table.
# ---------------------------------------------------------------------------

def remora_forward_annotated(
    hidden_states: torch.Tensor,   # [T, D]
    w_gate:        torch.Tensor,   # [E, D, F]
    w_up:          torch.Tensor,   # [E, D, F]
    w_down:        torch.Tensor,   # [E, F, D]
    topk_ids:      torch.Tensor,   # [T, top_k] int32
    topk_weights:  torch.Tensor,   # [T, top_k] fp16
) -> torch.Tensor:
    T, D = hidden_states.shape
    E    = w_gate.shape[0]

    with torch.profiler.record_function("output_init"):
        output = torch.zeros(T, D, dtype=hidden_states.dtype, device=hidden_states.device)

    for k in range(TOP_K):
        ids_k = topk_ids[:, k].long()
        wts_k = topk_weights[:, k]

        # --- mask creation --------------------------------------------------
        masks = []
        for e in range(E):
            with torch.profiler.record_function("mask_creation"):
                masks.append(ids_k == e)

        # --- any() + gather -------------------------------------------------
        expert_tokens = []
        for e in range(E):
            with torch.profiler.record_function("any_check"):
                has_tokens = masks[e].any().item()
            with torch.profiler.record_function("gather"):
                expert_tokens.append(
                    hidden_states[masks[e]] if has_tokens
                    else hidden_states.new_empty(0, D)
                )

        # --- Triton kernel dispatch -----------------------------------------
        with torch.profiler.record_function("run_remora_outlined"):
            expert_outs = run_remora_outlined(expert_tokens, w_gate, w_up, w_down)

        # --- scatter-add ----------------------------------------------------
        for e in range(E):
            with torch.profiler.record_function("any_check"):
                has_tokens = masks[e].any().item()
            with torch.profiler.record_function("scatter"):
                if has_tokens:
                    output[masks[e]] += wts_k[masks[e], None] * expert_outs[e]

    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not torch.cuda.is_available():
        print("ERROR: no CUDA device found.", file=sys.stderr)
        sys.exit(1)

    device = "cuda"
    gpu    = torch.cuda.get_device_name(0)

    print("=" * 72)
    print("remora  profile_remora  —  zipf_1.0 routing")
    print(f"GPU:          {gpu}")
    print(f"Mixtral dims: hidden={HIDDEN_DIM}, intermediate={INTERMEDIATE_DIM}, "
          f"experts={NUM_EXPERTS}, top_k={TOP_K}")
    print(f"Batch size:   {NUM_TOKENS} tokens")
    print(f"Profile:      {WARMUP_ITERS} warmup + {PROFILE_ITERS} profiled iters")
    print("=" * 72)
    print()

    # ---- Weights -----------------------------------------------------------
    torch.manual_seed(0)
    scale  = 0.02
    w_gate = (torch.randn(NUM_EXPERTS, HIDDEN_DIM, INTERMEDIATE_DIM,
                          dtype=DTYPE, device=device) * scale).contiguous()
    w_up   = (torch.randn(NUM_EXPERTS, HIDDEN_DIM, INTERMEDIATE_DIM,
                          dtype=DTYPE, device=device) * scale).contiguous()
    w_down = (torch.randn(NUM_EXPERTS, INTERMEDIATE_DIM, HIDDEN_DIM,
                          dtype=DTYPE, device=device) * scale).contiguous()

    topk_ids, topk_weights = _make_zipf_1(device)
    hidden_states = (
        torch.randn(NUM_TOKENS, HIDDEN_DIM, dtype=DTYPE, device=device) * 0.1
    ).contiguous()

    tpe = topk_ids.flatten().bincount(minlength=NUM_EXPERTS).cpu().tolist()
    print(f"Routing (zipf_1.0): tokens_per_expert = {tpe}")
    print()

    # ---- Pre-compile Triton buckets ----------------------------------------
    warmup_all_buckets(hidden_dim=HIDDEN_DIM, inter_dim=INTERMEDIATE_DIM, device=device)

    # ---- Warmup (un-profiled) ----------------------------------------------
    print(f"Warming up remora_forward ({WARMUP_ITERS} iters)...")
    for _ in range(WARMUP_ITERS):
        remora_forward_annotated(
            hidden_states, w_gate, w_up, w_down, topk_ids, topk_weights
        )
    torch.cuda.synchronize()
    print()

    # ---- Wall-time baseline (un-profiled, 50 iters) ------------------------
    t0 = time.perf_counter()
    for _ in range(50):
        remora_forward_annotated(
            hidden_states, w_gate, w_up, w_down, topk_ids, topk_weights
        )
    torch.cuda.synchronize()
    wall_ms_per_iter = (time.perf_counter() - t0) / 50 * 1e3
    print(f"Wall time (50-iter mean, un-profiled): {wall_ms_per_iter:.3f} ms/iter")
    print()

    # ---- Profiled run ------------------------------------------------------
    out_dir    = os.path.join(_HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    trace_path = os.path.join(out_dir, "remora_profile.json")

    print(f"Profiling {PROFILE_ITERS} iterations...")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=False,
        profile_memory=False,
    ) as prof:
        for _ in range(PROFILE_ITERS):
            remora_forward_annotated(
                hidden_states, w_gate, w_up, w_down, topk_ids, topk_weights
            )
            torch.cuda.synchronize()

    prof.export_chrome_trace(trace_path)
    print(f"Chrome trace saved to {os.path.relpath(trace_path)}")
    print()

    # ---- Top-20 by self CUDA time ------------------------------------------
    print("=" * 72)
    print("Top 20 ops by self CUDA time  (across all profiled iters)")
    print("=" * 72)
    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=20,
        )
    )

    # ---- Phase summary table -----------------------------------------------
    PHASES = [
        "output_init",
        "mask_creation",
        "any_check",
        "gather",
        "run_remora_outlined",
        "scatter",
    ]

    avgs = {e.key: e for e in prof.key_averages()}

    print("=" * 72)
    print("Phase summary  (CPU time = dispatch + sync overhead; "
          "CUDA time = GPU kernel execution)")
    print("=" * 72)
    print(
        f"{'phase':<22}  {'calls':>6}  "
        f"{'cpu_total_ms':>13}  {'cuda_total_ms':>14}  "
        f"{'cpu_avg_us':>11}  {'cuda_avg_us':>12}"
    )
    print("-" * 84)

    total_cuda_ms = 0.0
    total_cpu_ms  = 0.0

    for phase in PHASES:
        if phase not in avgs:
            print(f"  {phase:<20}  (not found in profile)")
            continue
        e          = avgs[phase]
        cpu_tot_ms = e.self_cpu_time_total  / 1e3   # µs → ms
        cuda_tot_ms= e.self_cuda_time_total / 1e3
        calls      = e.count
        cpu_avg_us = e.self_cpu_time_total  / max(calls, 1)
        cuda_avg_us= e.self_cuda_time_total / max(calls, 1)
        total_cpu_ms  += cpu_tot_ms
        total_cuda_ms += cuda_tot_ms
        print(
            f"  {phase:<20}  {calls:>6}  "
            f"{cpu_tot_ms:>12.3f}ms  {cuda_tot_ms:>13.3f}ms  "
            f"{cpu_avg_us:>10.1f}µs  {cuda_avg_us:>11.1f}µs"
        )

    print("-" * 84)
    # Sum all CUDA kernel time (not just annotated phases) for the true GPU total
    all_cuda_ms = sum(
        e.self_cuda_time_total for e in prof.key_averages()
    ) / 1e3
    all_cpu_ms  = sum(
        e.self_cpu_time_total  for e in prof.key_averages()
    ) / 1e3

    print()
    print("=" * 72)
    print("Wall time vs GPU kernel time breakdown")
    print("=" * 72)
    profiled_wall_ms = wall_ms_per_iter * PROFILE_ITERS
    print(f"  Wall time ({PROFILE_ITERS} iters):         {profiled_wall_ms:.3f} ms")
    print(f"  Total CPU time (profiler):   {all_cpu_ms:.3f} ms")
    print(f"  Total CUDA kernel time:      {all_cuda_ms:.3f} ms")
    print(f"  Per-iter wall time:          {wall_ms_per_iter:.3f} ms")
    print(f"  Per-iter CUDA kernel time:   {all_cuda_ms / PROFILE_ITERS:.3f} ms")
    print(
        f"  Python/launch overhead est:  "
        f"{max(0, wall_ms_per_iter - all_cuda_ms / PROFILE_ITERS):.3f} ms/iter"
    )
    print()
    print(f"Chrome trace: open {os.path.relpath(trace_path)} in chrome://tracing")


if __name__ == "__main__":
    main()
