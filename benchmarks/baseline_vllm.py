#!/usr/bin/env python3
"""
benchmarks/baseline_vllm.py

Measures vLLM's fused_moe throughput at Mixtral-8x7B dimensions across six
routing distributions.  This is the production baseline against which the
remora per-expert Triton dispatch is compared.

vLLM's fused_moe runs all 8 experts in one monolithic Triton kernel with
capacity-factor padding — every expert's slot is filled to the maximum token
count in that batch, regardless of the actual routing skew.  That means skewed
distributions pay the cost of the largest-loaded expert for every expert.

Weight shapes follow vLLM convention:
    w1: [E, 2*F, H]  — gate + up weights concatenated on dim 0 (transposed)
    w2: [E, H,   F]  — down weight (transposed)

where H = hidden_dim, F = intermediate_dim.

Usage (requires CUDA + vLLM installed):
    python3 benchmarks/baseline_vllm.py

Output:
    stdout  — formatted table
    benchmarks/results/baseline_vllm.json
"""

import json
import os
import sys

import numpy as np
import torch

# ---------------------------------------------------------------------------
# vLLM import — warn clearly if not available
# ---------------------------------------------------------------------------
try:
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_moe
except ImportError:
    print(
        "ERROR: could not import fused_moe from vLLM.\n"
        "Install vLLM:  pip install vllm",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Mixtral-8x7B dimensions
# ---------------------------------------------------------------------------
HIDDEN_DIM      = 4096
INTERMEDIATE_DIM = 14336
NUM_EXPERTS     = 8
TOP_K           = 2
NUM_TOKENS      = 512      # batch size for the main experiment
DTYPE           = torch.float16

WARMUP_ITERS  = 5
TIMED_ITERS   = 50

# ---------------------------------------------------------------------------
# Routing distributions
# ---------------------------------------------------------------------------

def _make_uniform(num_tokens: int, num_experts: int, top_k: int,
                  dtype: torch.dtype, device: str):
    """Uniform: each expert receives ~num_tokens/num_experts tokens."""
    probs = torch.ones(num_tokens, num_experts, dtype=torch.float32, device=device)
    ids   = torch.multinomial(probs, num_samples=top_k, replacement=False).to(torch.int32)
    # Equal weight to each selected expert
    weights = torch.full((num_tokens, top_k), 1.0 / top_k, dtype=dtype, device=device)
    return ids, weights


def _make_zipf(s: float, num_tokens: int, num_experts: int, top_k: int,
               dtype: torch.dtype, device: str):
    """
    Zipf s: p_e ∝ 1/e^s  for e = 1..num_experts.
    Sample top_k experts per token without replacement; normalize selected
    probabilities to get per-token weights that sum to 1.
    """
    ranks = np.arange(1, num_experts + 1, dtype=np.float64)
    p     = 1.0 / (ranks ** s)
    p    /= p.sum()

    # Expand to [num_tokens, num_experts] for multinomial sampling
    probs = torch.tensor(p, dtype=torch.float32, device=device)
    probs = probs.unsqueeze(0).expand(num_tokens, -1).contiguous()

    ids   = torch.multinomial(probs, num_samples=top_k, replacement=False).to(torch.int32)

    # Weight = normalized probability of each selected expert
    sel   = torch.gather(probs, 1, ids.long())          # [T, top_k]
    weights = (sel / sel.sum(dim=-1, keepdim=True)).to(dtype)
    return ids, weights


def _make_single(num_tokens: int, num_experts: int, top_k: int,
                 dtype: torch.dtype, device: str):
    """
    Single: all tokens routed to expert 0 (weight 1.0).
    The secondary selection is expert 1 with weight 0.0 so vLLM still sees
    valid top_k=2 routing, but functionally all computation is on expert 0.
    """
    assert top_k == 2, "single distribution requires top_k=2"
    ids = torch.stack([
        torch.zeros(num_tokens, dtype=torch.int32, device=device),
        torch.ones( num_tokens, dtype=torch.int32, device=device),
    ], dim=1)   # [T, 2]
    weights = torch.stack([
        torch.ones( num_tokens, dtype=dtype, device=device),
        torch.zeros(num_tokens, dtype=dtype, device=device),
    ], dim=1)   # [T, 2]
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
# Timing helper
# ---------------------------------------------------------------------------

def benchmark_fused_moe(
    hidden_states: torch.Tensor,   # [T, H]
    w1:            torch.Tensor,   # [E, 2F, H]
    w2:            torch.Tensor,   # [E, H,  F]
    topk_weights:  torch.Tensor,   # [T, top_k]  fp16, rows sum to 1
    topk_ids:      torch.Tensor,   # [T, top_k]  int32
    warmup:        int = WARMUP_ITERS,
    timed:         int = TIMED_ITERS,
) -> dict:
    """
    Returns {latency_ms (median), tokens_per_sec, tokens_per_expert}.
    Uses torch.cuda.Event for sub-millisecond precision.
    """
    num_tokens  = hidden_states.shape[0]
    num_experts = w1.shape[0]

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev   = torch.cuda.Event(enable_timing=True)

    # Warmup — ensures kernels are compiled and caches are warm
    for _ in range(warmup):
        fused_moe(hidden_states, w1, w2, topk_weights, topk_ids, inplace=False)
    torch.cuda.synchronize()

    # Timed loop — one Event pair per iteration for fine-grained measurement
    latencies_ms: list[float] = []
    for _ in range(timed):
        start_ev.record()
        fused_moe(hidden_states, w1, w2, topk_weights, topk_ids, inplace=False)
        end_ev.record()
        torch.cuda.synchronize()
        latencies_ms.append(start_ev.elapsed_time(end_ev))

    median_ms     = float(np.median(latencies_ms))
    tokens_per_sec = num_tokens / (median_ms * 1e-3)

    # Token count per expert (all top_k slots — reflects vLLM's actual work)
    tokens_per_expert = (
        topk_ids.flatten()
        .bincount(minlength=num_experts)
        .cpu()
        .tolist()
    )

    return {
        "latency_ms":       round(median_ms, 4),
        "tokens_per_sec":   round(tokens_per_sec, 1),
        "tokens_per_expert": tokens_per_expert,
        "batch_size":       num_tokens,
        "num_experts":      num_experts,
        "hidden_dim":       hidden_states.shape[1],
    }


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
    print("remora  vLLM fused_moe baseline")
    print(f"GPU:          {gpu}")
    print(f"Mixtral dims: hidden={HIDDEN_DIM}, intermediate={INTERMEDIATE_DIM}, "
          f"experts={NUM_EXPERTS}, top_k={TOP_K}")
    print(f"Batch size:   {NUM_TOKENS} tokens")
    print(f"Timing:       {WARMUP_ITERS} warmup + {TIMED_ITERS} timed iters, median")
    print("=" * 72)
    print()

    # ---- Weights ----------------------------------------------------------
    # vLLM w1: [E, 2*F, H]  gate and up projections concatenated (rows = out)
    # vLLM w2: [E, H,   F]  down projection
    torch.manual_seed(0)
    scale = 0.02
    w1 = (torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE_DIM, HIDDEN_DIM,
                      dtype=DTYPE, device=device) * scale).contiguous()
    w2 = (torch.randn(NUM_EXPERTS, HIDDEN_DIM, INTERMEDIATE_DIM,
                      dtype=DTYPE, device=device) * scale).contiguous()

    # ---- Sweep distributions ----------------------------------------------
    results: dict[str, dict] = {}
    col_w = 10   # column width for distribution name

    header = (
        f"{'distribution':<{col_w}}  {'latency_ms':>11}  "
        f"{'tokens/sec':>12}  {'tokens_per_expert'}"
    )
    print(header)
    print("-" * len(header))

    for dist_name, dist_fn in DISTRIBUTIONS.items():
        topk_ids, topk_weights = dist_fn(
            NUM_TOKENS, NUM_EXPERTS, TOP_K, DTYPE, device
        )
        topk_ids     = topk_ids.contiguous()
        topk_weights = topk_weights.contiguous()

        hidden_states = (
            torch.randn(NUM_TOKENS, HIDDEN_DIM, dtype=DTYPE, device=device) * 0.1
        ).contiguous()

        result = benchmark_fused_moe(
            hidden_states, w1, w2, topk_weights, topk_ids,
        )
        results[dist_name] = result

        tpe_str = " ".join(f"{c:4d}" for c in result["tokens_per_expert"])
        print(
            f"{dist_name:<{col_w}}  "
            f"{result['latency_ms']:>10.3f}ms  "
            f"{result['tokens_per_sec']:>11,.0f}  "
            f"[{tpe_str}]"
        )

    print()

    # ---- Save JSON --------------------------------------------------------
    out_dir  = os.path.join(os.path.dirname(__file__), "results")
    out_path = os.path.join(out_dir, "baseline_vllm.json")
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
    payload = {"meta": meta, "results": results}

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Results saved to {os.path.relpath(out_path)}")


if __name__ == "__main__":
    main()
