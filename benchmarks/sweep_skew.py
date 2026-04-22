#!/usr/bin/env python3
"""
benchmarks/sweep_skew.py

Main experiment for the Remora paper.
Compares vLLM fused_moe (monolithic kernel) against Remora per-expert
Triton dispatch across six routing skew distributions at Mixtral-8x7B dims.

Systems
-------
vLLM    — fused_moe: one monolithic Triton kernel with capacity-factor padding.
          Every expert's slot is filled to the max token count in the batch,
          so skewed distributions pay the cost of the heaviest expert for all.

Remora  — run_remora_outlined: per-expert Triton kernels with shape-bucket
          dispatch.  Only actual tokens are processed per expert; idle experts
          are skipped entirely.

Routing distributions: uniform, zipf_0.5, zipf_1.0, zipf_1.5, zipf_2.0, single.
Mixtral-8x7B dims: hidden=4096, intermediate=14336, experts=8, top_k=2, batch=512.

Output
------
  stdout                       — formatted comparison table
  benchmarks/results/sweep_skew.json
  benchmarks/results/sweep_skew.csv
"""

import csv
import json
import os
import sys

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Sibling-module imports
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from triton_expert_kernel import run_remora_outlined, warmup_all_buckets

try:
    import vllm.model_executor.layers.fused_moe.fused_moe as _fused_moe_mod
    if hasattr(_fused_moe_mod, "fused_moe"):
        fused_moe = _fused_moe_mod.fused_moe
    elif hasattr(_fused_moe_mod, "fused_experts"):
        fused_moe = _fused_moe_mod.fused_experts
    else:
        raise ImportError(
            "neither fused_moe nor fused_experts found in vllm fused_moe module"
        )
except ImportError as e:
    print(
        f"ERROR: could not import fused_moe from vLLM: {e}\n"
        "Install vLLM:  pip install vllm",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Mixtral-8x7B dimensions
# ---------------------------------------------------------------------------
HIDDEN_DIM       = 4096
INTERMEDIATE_DIM = 14336
NUM_EXPERTS      = 8
TOP_K            = 2
NUM_TOKENS       = 512
DTYPE            = torch.float16

WARMUP_ITERS = 5
TIMED_ITERS  = 50

# ---------------------------------------------------------------------------
# Routing distributions  (identical logic to baseline_vllm.py)
# ---------------------------------------------------------------------------

def _make_uniform(num_tokens, num_experts, top_k, dtype, device):
    probs   = torch.ones(num_tokens, num_experts, dtype=torch.float32, device=device)
    ids     = torch.multinomial(probs, num_samples=top_k, replacement=False).to(torch.int32)
    weights = torch.full((num_tokens, top_k), 1.0 / top_k, dtype=dtype, device=device)
    return ids, weights


def _make_zipf(s, num_tokens, num_experts, top_k, dtype, device):
    ranks = np.arange(1, num_experts + 1, dtype=np.float64)
    p     = 1.0 / (ranks ** s)
    p    /= p.sum()
    probs   = torch.tensor(p, dtype=torch.float32, device=device)
    probs   = probs.unsqueeze(0).expand(num_tokens, -1).contiguous()
    ids     = torch.multinomial(probs, num_samples=top_k, replacement=False).to(torch.int32)
    sel     = torch.gather(probs, 1, ids.long())
    weights = (sel / sel.sum(dim=-1, keepdim=True)).to(dtype)
    return ids, weights


def _make_single(num_tokens, num_experts, top_k, dtype, device):
    assert top_k == 2, "single distribution requires top_k=2"
    ids = torch.stack([
        torch.zeros(num_tokens, dtype=torch.int32, device=device),
        torch.ones( num_tokens, dtype=torch.int32, device=device),
    ], dim=1)
    weights = torch.stack([
        torch.ones( num_tokens, dtype=dtype, device=device),
        torch.zeros(num_tokens, dtype=dtype, device=device),
    ], dim=1)
    return ids, weights


DISTRIBUTIONS = {
    "uniform":  lambda nt, ne, tk, dt, dv: _make_uniform(nt, ne, tk, dt, dv),
    "zipf_0.5": lambda nt, ne, tk, dt, dv: _make_zipf(0.5, nt, ne, tk, dt, dv),
    "zipf_1.0": lambda nt, ne, tk, dt, dv: _make_zipf(1.0, nt, ne, tk, dt, dv),
    "zipf_1.5": lambda nt, ne, tk, dt, dv: _make_zipf(1.5, nt, ne, tk, dt, dv),
    "zipf_2.0": lambda nt, ne, tk, dt, dv: _make_zipf(2.0, nt, ne, tk, dt, dv),
    "single":   lambda nt, ne, tk, dt, dv: _make_single(nt, ne, tk, dt, dv),
}

# ---------------------------------------------------------------------------
# Remora forward pass
#
# Mirrors what the ExpertOutliningPass-compiled @main does at runtime:
#   for each top-k slot k:
#     gather per-expert token subsets
#     dispatch to bucket-selected Triton kernels
#     scatter back, weighted
# ---------------------------------------------------------------------------

def remora_forward(
    hidden_states: torch.Tensor,   # [T, D]  float16
    w_gate:        torch.Tensor,   # [E, D, F]  float16
    w_up:          torch.Tensor,   # [E, D, F]  float16
    w_down:        torch.Tensor,   # [E, F, D]  float16
    topk_ids:      torch.Tensor,   # [T, top_k]  int32
    topk_weights:  torch.Tensor,   # [T, top_k]  float16
    streams:       list = None,    # pre-created CUDA streams (one per expert)
) -> torch.Tensor:                 # [T, D]  float16
    T, D   = hidden_states.shape
    E      = w_gate.shape[0]
    output = torch.zeros(T, D, dtype=hidden_states.dtype, device=hidden_states.device)

    for k in range(TOP_K):
        ids_k = topk_ids[:, k].long()   # [T]
        wts_k = topk_weights[:, k]      # [T]  fp16

        # Sort tokens by expert id — one argsort, then one contiguous gather each.
        sort_order    = torch.argsort(ids_k, stable=True)          # [T]
        ids_sorted    = ids_k[sort_order]                           # [T]
        x_sorted      = hidden_states[sort_order]                   # [T, D]
        wts_sorted    = wts_k[sort_order]                           # [T]

        # Split into contiguous per-expert slices — pure metadata, no CUDA op.
        counts        = torch.bincount(ids_sorted, minlength=E)     # [E]
        expert_tokens = list(x_sorted.split(counts.tolist()))       # list[T_e, D]

        # Dispatch all experts — optionally concurrent on separate CUDA streams.
        expert_outs = run_remora_outlined(
            expert_tokens, w_gate, w_up, w_down, streams=streams
        )

        # Scatter back — one cat + one index_add_ replaces E masked index_put calls.
        out_sorted = torch.cat(expert_outs, dim=0)                  # [T, D]
        output.index_add_(0, sort_order, out_sorted * wts_sorted[:, None])

    return output


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def _timed(fn, warmup: int, timed: int) -> float:
    """Warmup, then measure median latency in ms using CUDA Events."""
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev   = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    latencies: list[float] = []
    for _ in range(timed):
        start_ev.record()
        fn()
        end_ev.record()
        torch.cuda.synchronize()
        latencies.append(start_ev.elapsed_time(end_ev))

    return float(np.median(latencies))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not torch.cuda.is_available():
        print("ERROR: no CUDA device found.", file=sys.stderr)
        sys.exit(1)

    device = "cuda"
    gpu    = torch.cuda.get_device_name(0)

    print("=" * 84)
    print("remora  sweep_skew  —  vLLM fused_moe vs Remora per-expert dispatch")
    print(f"GPU:          {gpu}")
    print(f"Mixtral dims: hidden={HIDDEN_DIM}, intermediate={INTERMEDIATE_DIM}, "
          f"experts={NUM_EXPERTS}, top_k={TOP_K}")
    print(f"Batch size:   {NUM_TOKENS} tokens")
    print(f"Timing:       {WARMUP_ITERS} warmup + {TIMED_ITERS} timed iters, median")
    print("=" * 84)
    print()

    # ---- Weights -----------------------------------------------------------
    # vLLM:   w1 [E, 2*F, H], w2 [E, H, F]  (gate+up concatenated, transposed)
    # Remora: w_gate [E, D, F], w_up [E, D, F], w_down [E, F, D]
    # Allocated separately — both do the same compute, benchmarking is throughput only.
    torch.manual_seed(0)
    scale = 0.02

    w1_vllm = (torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE_DIM, HIDDEN_DIM,
                           dtype=DTYPE, device=device) * scale).contiguous()
    w2_vllm = (torch.randn(NUM_EXPERTS, HIDDEN_DIM, INTERMEDIATE_DIM,
                           dtype=DTYPE, device=device) * scale).contiguous()

    w_gate  = (torch.randn(NUM_EXPERTS, HIDDEN_DIM, INTERMEDIATE_DIM,
                           dtype=DTYPE, device=device) * scale).contiguous()
    w_up    = (torch.randn(NUM_EXPERTS, HIDDEN_DIM, INTERMEDIATE_DIM,
                           dtype=DTYPE, device=device) * scale).contiguous()
    w_down  = (torch.randn(NUM_EXPERTS, INTERMEDIATE_DIM, HIDDEN_DIM,
                           dtype=DTYPE, device=device) * scale).contiguous()

    # ---- Pre-compile all Triton bucket kernels -----------------------------
    warmup_all_buckets(hidden_dim=HIDDEN_DIM, inter_dim=INTERMEDIATE_DIM,
                       device=device)

    # ---- Header ------------------------------------------------------------
    col = 10
    header = (
        f"{'distribution':<{col}}  {'vllm_ms':>8}  {'remora_ms':>9}  "
        f"{'speedup':>7}  {'vllm_tok/s':>11}  {'remora_tok/s':>12}  tokens_per_expert"
    )
    print(header)
    print("-" * len(header))

    # ---- Sweep -------------------------------------------------------------
    results: dict[str, dict] = {}

    for dist_name, dist_fn in DISTRIBUTIONS.items():
        topk_ids, topk_weights = dist_fn(
            NUM_TOKENS, NUM_EXPERTS, TOP_K, DTYPE, device
        )
        topk_ids     = topk_ids.contiguous()
        topk_weights = topk_weights.contiguous()

        hidden_states = (
            torch.randn(NUM_TOKENS, HIDDEN_DIM, dtype=DTYPE, device=device) * 0.1
        ).contiguous()

        # Token count per expert (across all top-k slots)
        tpe = topk_ids.flatten().bincount(minlength=NUM_EXPERTS).cpu().tolist()

        # vLLM — monolithic kernel
        vllm_ms = _timed(
            lambda: fused_moe(
                hidden_states, w1_vllm, w2_vllm, topk_weights, topk_ids, inplace=False
            ),
            WARMUP_ITERS, TIMED_ITERS,
        )

        # Remora — per-expert bucket dispatch, sequential
        remora_ms = _timed(
            lambda: remora_forward(
                hidden_states, w_gate, w_up, w_down, topk_ids, topk_weights,
            ),
            WARMUP_ITERS, TIMED_ITERS,
        )

        speedup    = vllm_ms / remora_ms
        vllm_tps   = NUM_TOKENS / (vllm_ms   * 1e-3)
        remora_tps = NUM_TOKENS / (remora_ms  * 1e-3)

        results[dist_name] = {
            "vllm_ms":           round(vllm_ms,   4),
            "remora_ms":         round(remora_ms,  4),
            "speedup":           round(speedup,    3),
            "vllm_tok_s":        round(vllm_tps,   1),
            "remora_tok_s":      round(remora_tps, 1),
            "tokens_per_expert": tpe,
        }

        tpe_str = " ".join(f"{c:4d}" for c in tpe)
        print(
            f"{dist_name:<{col}}  "
            f"{vllm_ms:>7.3f}ms  "
            f"{remora_ms:>8.3f}ms  "
            f"{speedup:>6.2f}x  "
            f"{vllm_tps:>10,.0f}  "
            f"{remora_tps:>11,.0f}  "
            f"[{tpe_str}]"
        )

    print()

    # ---- Save JSON ---------------------------------------------------------
    out_dir = os.path.join(_HERE, "results")
    os.makedirs(out_dir, exist_ok=True)

    meta = {
        "gpu":              gpu,
        "hidden_dim":       HIDDEN_DIM,
        "intermediate_dim": INTERMEDIATE_DIM,
        "num_experts":      NUM_EXPERTS,
        "top_k":            TOP_K,
        "batch_size":       NUM_TOKENS,
        "dtype":            str(DTYPE),
        "warmup_iters":     WARMUP_ITERS,
        "timed_iters":      TIMED_ITERS,
    }
    json_path = os.path.join(out_dir, "sweep_skew.json")
    with open(json_path, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    print(f"JSON saved to {os.path.relpath(json_path)}")

    # ---- Save CSV ----------------------------------------------------------
    csv_path  = os.path.join(out_dir, "sweep_skew.csv")
    fieldnames = (
        ["distribution", "vllm_ms", "remora_ms", "speedup", "vllm_tok_s", "remora_tok_s"]
        + [f"e{i}" for i in range(NUM_EXPERTS)]
    )
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for dist_name, r in results.items():
            row: dict = {
                "distribution": dist_name,
                "vllm_ms":      r["vllm_ms"],
                "remora_ms":    r["remora_ms"],
                "speedup":      r["speedup"],
                "vllm_tok_s":   r["vllm_tok_s"],
                "remora_tok_s": r["remora_tok_s"],
            }
            for i, c in enumerate(r["tokens_per_expert"]):
                row[f"e{i}"] = c
            writer.writerow(row)
    print(f"CSV  saved to {os.path.relpath(csv_path)}")


if __name__ == "__main__":
    main()
