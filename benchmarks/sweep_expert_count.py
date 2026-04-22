#!/usr/bin/env python3
"""
benchmarks/sweep_expert_count.py

Sweeps num_experts ∈ {8, 16, 32} across routing distributions, measuring
how many experts receive zero tokens at each expert count and how that
correlates with Remora's speedup over vLLM.

Key metric: zero-token expert count.  Remora's advantage comes from skipping
those experts entirely; vLLM pads all slots to the max token count regardless.

Dimensions: hidden=4096, intermediate=14336, top_k=2, batch=512, float16.
VRAM guard: configurations requiring more than 20 GB of weight memory are
skipped with a warning.

Output
------
  stdout                              — per-expert-count distribution table,
                                        zero-expert summary, and final table
  benchmarks/results/sweep_expert_count.json
  benchmarks/results/sweep_expert_count.csv
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

from sweep_skew import remora_forward, fused_moe, DTYPE, WARMUP_ITERS, TIMED_ITERS
from triton_expert_kernel import warmup_all_buckets

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HIDDEN_DIM       = 4096
INTERMEDIATE_DIM = 14336
TOP_K            = 2
NUM_TOKENS       = 512

EXPERT_COUNTS    = [8, 16, 32]
VRAM_LIMIT_BYTES = 20 * 1024 ** 3   # 20 GB

WIN_REMORA = 1.05
WIN_VLLM   = 0.95

# ---------------------------------------------------------------------------
# Routing distributions (parameterised by num_experts)
# ---------------------------------------------------------------------------

def _make_uniform(num_tokens, num_experts, top_k, dtype, device):
    probs   = torch.ones(num_tokens, num_experts, dtype=torch.float32, device=device)
    ids     = torch.multinomial(probs, num_samples=top_k, replacement=False).to(torch.int32)
    weights = torch.full((num_tokens, top_k), 1.0 / top_k, dtype=dtype, device=device)
    return ids.contiguous(), weights.contiguous()


def _make_zipf(s, num_tokens, num_experts, top_k, dtype, device):
    ranks = np.arange(1, num_experts + 1, dtype=np.float64)
    p     = 1.0 / (ranks ** s)
    p    /= p.sum()
    probs   = torch.tensor(p, dtype=torch.float32, device=device)
    probs   = probs.unsqueeze(0).expand(num_tokens, -1).contiguous()
    ids     = torch.multinomial(probs, num_samples=top_k, replacement=False).to(torch.int32)
    sel     = torch.gather(probs, 1, ids.long())
    weights = (sel / sel.sum(dim=-1, keepdim=True)).to(dtype)
    return ids.contiguous(), weights.contiguous()


def _make_single(num_tokens, num_experts, top_k, dtype, device):
    assert top_k == 2
    ids = torch.stack([
        torch.zeros(num_tokens, dtype=torch.int32, device=device),
        torch.ones( num_tokens, dtype=torch.int32, device=device),
    ], dim=1)
    weights = torch.stack([
        torch.ones( num_tokens, dtype=dtype, device=device),
        torch.zeros(num_tokens, dtype=dtype, device=device),
    ], dim=1)
    return ids.contiguous(), weights.contiguous()


DIST_FNS: dict[str, callable] = {
    "uniform":  lambda nt, ne, tk, dt, dv: _make_uniform(nt, ne, tk, dt, dv),
    "zipf_0.5": lambda nt, ne, tk, dt, dv: _make_zipf(0.5, nt, ne, tk, dt, dv),
    "zipf_1.0": lambda nt, ne, tk, dt, dv: _make_zipf(1.0, nt, ne, tk, dt, dv),
    "zipf_1.5": lambda nt, ne, tk, dt, dv: _make_zipf(1.5, nt, ne, tk, dt, dv),
    "zipf_2.0": lambda nt, ne, tk, dt, dv: _make_zipf(2.0, nt, ne, tk, dt, dv),
    "single":   lambda nt, ne, tk, dt, dv: _make_single(nt, ne, tk, dt, dv),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vram_estimate_bytes(num_experts: int) -> int:
    """
    Rough fp16 byte count for all five weight tensors:
      w_gate [E,D,F] + w_up [E,D,F] + w_down [E,F,D]   → 3 × E×D×F
      w1_vllm [E,2F,D]                                  → 2 × E×D×F
      w2_vllm [E,D,F]                                   → 1 × E×D×F
    Total = 6 × E×D×F × 2 bytes
    """
    return num_experts * HIDDEN_DIM * INTERMEDIATE_DIM * 6 * 2


def _timed(fn, warmup: int, timed: int) -> float:
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev   = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    lats: list[float] = []
    for _ in range(timed):
        start_ev.record()
        fn()
        end_ev.record()
        torch.cuda.synchronize()
        lats.append(start_ev.elapsed_time(end_ev))
    return float(np.median(lats))


def _winner(speedup: float) -> str:
    if speedup > WIN_REMORA:
        return "remora"
    if speedup < WIN_VLLM:
        return "vllm"
    return "~tie"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not torch.cuda.is_available():
        print("ERROR: no CUDA device found.", file=sys.stderr)
        sys.exit(1)

    device = "cuda"
    gpu    = torch.cuda.get_device_name(0)

    print("=" * 76)
    print("remora  sweep_expert_count  —  vLLM vs Remora across expert counts")
    print(f"GPU:           {gpu}")
    print(f"Mixtral dims:  hidden={HIDDEN_DIM}, intermediate={INTERMEDIATE_DIM}, top_k={TOP_K}")
    print(f"Batch size:    {NUM_TOKENS} tokens")
    print(f"Expert counts: {EXPERT_COUNTS}")
    print(f"Timing:        {WARMUP_ITERS} warmup + {TIMED_ITERS} timed iters, median")
    print("=" * 76)
    print()

    # Pre-compile Triton buckets once (shared across all expert counts).
    warmup_all_buckets(hidden_dim=HIDDEN_DIM, inter_dim=INTERMEDIATE_DIM,
                       device=device)

    all_results:  dict[str, dict] = {}
    summary_rows: list[dict]      = []

    for E in EXPERT_COUNTS:
        needed_gb = _vram_estimate_bytes(E) / 1024 ** 3
        if _vram_estimate_bytes(E) > VRAM_LIMIT_BYTES:
            print(
                f"\nWARNING: skipping experts={E} — estimated weight VRAM "
                f"{needed_gb:.1f} GB exceeds 20 GB limit"
            )
            continue

        print(f"\n{'=' * 60}")
        print(f"experts = {E}   (estimated weight VRAM: {needed_gb:.1f} GB)")
        print(f"{'=' * 60}")

        # Allocate weights for this expert count
        torch.manual_seed(0)
        scale   = 0.02
        w1_vllm = (torch.randn(E, 2 * INTERMEDIATE_DIM, HIDDEN_DIM,
                               dtype=DTYPE, device=device) * scale).contiguous()
        w2_vllm = (torch.randn(E, HIDDEN_DIM, INTERMEDIATE_DIM,
                               dtype=DTYPE, device=device) * scale).contiguous()
        w_gate  = (torch.randn(E, HIDDEN_DIM, INTERMEDIATE_DIM,
                               dtype=DTYPE, device=device) * scale).contiguous()
        w_up    = (torch.randn(E, HIDDEN_DIM, INTERMEDIATE_DIM,
                               dtype=DTYPE, device=device) * scale).contiguous()
        w_down  = (torch.randn(E, INTERMEDIATE_DIM, HIDDEN_DIM,
                               dtype=DTYPE, device=device) * scale).contiguous()

        print(
            f"  {'distribution':<12}  {'zero_experts':>14}  "
            f"{'vllm_ms':>8}  {'remora_ms':>9}  {'speedup':>7}  {'winner':<7}"
        )
        print("  " + "-" * 68)

        all_results[str(E)] = {}

        for dist_name, dist_fn in DIST_FNS.items():
            topk_ids, topk_weights = dist_fn(NUM_TOKENS, E, TOP_K, DTYPE, device)
            hidden_states = (
                torch.randn(NUM_TOKENS, HIDDEN_DIM, dtype=DTYPE, device=device) * 0.1
            ).contiguous()

            tpe          = topk_ids.flatten().bincount(minlength=E).cpu().tolist()
            zero_experts = sum(1 for c in tpe if c == 0)

            vllm_ms = _timed(
                lambda: fused_moe(
                    hidden_states, w1_vllm, w2_vllm, topk_weights, topk_ids, inplace=False
                ),
                WARMUP_ITERS, TIMED_ITERS,
            )
            remora_ms = _timed(
                lambda: remora_forward(
                    hidden_states, w_gate, w_up, w_down, topk_ids, topk_weights,
                ),
                WARMUP_ITERS, TIMED_ITERS,
            )

            speedup    = vllm_ms / remora_ms
            vllm_tps   = NUM_TOKENS / (vllm_ms   * 1e-3)
            remora_tps = NUM_TOKENS / (remora_ms  * 1e-3)
            win        = _winner(speedup)

            all_results[str(E)][dist_name] = {
                "num_experts":       E,
                "distribution":      dist_name,
                "zero_experts":      zero_experts,
                "tokens_per_expert": tpe,
                "vllm_ms":           round(vllm_ms,    4),
                "remora_ms":         round(remora_ms,   4),
                "vllm_tok_s":        round(vllm_tps,    1),
                "remora_tok_s":      round(remora_tps,  1),
                "speedup":           round(speedup,     3),
                "winner":            win,
            }
            summary_rows.append({
                "num_experts":  E,
                "distribution": dist_name,
                "zero_experts": zero_experts,
                "speedup":      round(speedup, 3),
                "winner":       win,
            })

            print(
                f"  {dist_name:<12}  {zero_experts:>4} of {E:<2}          "
                f"{vllm_ms:>7.3f}ms  {remora_ms:>8.3f}ms  "
                f"{speedup:>6.2f}x  {win:<7}"
            )

        # Free before next expert count to stay within VRAM budget
        del w1_vllm, w2_vllm, w_gate, w_up, w_down
        torch.cuda.empty_cache()

    # ---- Zero-expert detail ------------------------------------------------
    print()
    print("=" * 76)
    print("Zero-expert detail — how many experts are idle per config")
    print("=" * 76)
    for row in summary_rows:
        if row["distribution"] in ("zipf_2.0", "single"):
            E   = row["num_experts"]
            z   = row["zero_experts"]
            s   = row["speedup"]
            w   = row["winner"]
            d   = row["distribution"]
            print(
                f"  experts={E:<2},  {d:<10}  "
                f"{z:>2} of {E:<2} experts receive 0 tokens  "
                f"→ speedup {s:.2f}x  ({w})"
            )

    # ---- Final summary table -----------------------------------------------
    print()
    print("=" * 76)
    print("Summary table")
    print("=" * 76)
    print(
        f"  {'num_experts':<12}  {'distribution':<12}  "
        f"{'zero_experts':>12}  {'speedup':>8}  {'winner':<7}"
    )
    print("  " + "-" * 60)
    for row in summary_rows:
        print(
            f"  {row['num_experts']:<12}  {row['distribution']:<12}  "
            f"{row['zero_experts']:>12}  {row['speedup']:>7.2f}x  {row['winner']:<7}"
        )

    # ---- Save JSON ---------------------------------------------------------
    out_dir = os.path.join(_HERE, "results")
    os.makedirs(out_dir, exist_ok=True)

    meta = {
        "gpu":              gpu,
        "hidden_dim":       HIDDEN_DIM,
        "intermediate_dim": INTERMEDIATE_DIM,
        "top_k":            TOP_K,
        "batch_size":       NUM_TOKENS,
        "dtype":            str(DTYPE),
        "warmup_iters":     WARMUP_ITERS,
        "timed_iters":      TIMED_ITERS,
        "expert_counts":    EXPERT_COUNTS,
        "vram_limit_gb":    20,
    }
    json_path = os.path.join(out_dir, "sweep_expert_count.json")
    with open(json_path, "w") as f:
        json.dump({"meta": meta, "results": all_results}, f, indent=2)
    print(f"\nJSON saved to {os.path.relpath(json_path)}")

    # ---- Save CSV ----------------------------------------------------------
    csv_path   = os.path.join(out_dir, "sweep_expert_count.csv")
    fieldnames = [
        "num_experts", "distribution", "zero_experts",
        "vllm_ms", "remora_ms", "speedup",
        "vllm_tok_s", "remora_tok_s", "winner",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for E_str, dists in all_results.items():
            for r in dists.values():
                writer.writerow({k: r[k] for k in fieldnames})
    print(f"CSV  saved to {os.path.relpath(csv_path)}")


if __name__ == "__main__":
    main()
