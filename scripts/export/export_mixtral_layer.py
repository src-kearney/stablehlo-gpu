#!/usr/bin/env python3
"""
Export a single Mixtral-8x7B MoE layer to StableHLO.

Usage:
    python3 scripts/export/export_mixtral_layer.py

Output:
    mlir/stablehlo/mixtral_moe_layer.mlir

Writes one sparse MoE FFN block (router + 8 experts + SwiGLU) at Mixtral-8x7B
dimensions using random weights. Goal: confirm the expert dim appears as batch
dim 0 in every dot_general, same pattern as simple_moe.mlir — before writing
compiler/passes/ExpertOutlining.cpp.
"""
import os
import re
import sys
import numpy as np

# ---------------------------------------------------------------------------
# Mixtral-8x7B MoE dimensions
# ---------------------------------------------------------------------------
HIDDEN       = 4096
INTER        = 14336
NUM_EXPERTS  = 8
TOP_K        = 2
SEQ_LEN      = 512
T            = SEQ_LEN   # flat token batch: batch=1 * seq=512

REPO_ROOT = os.path.join(os.path.dirname(__file__), "../..")
OUTPUT_PATH = os.path.join(REPO_ROOT, "mlir/stablehlo/mixtral_moe_layer.mlir")

import jax
import jax.numpy as jnp


def silu(x):
    return x * jax.nn.sigmoid(x)


def mixtral_moe(tokens, router_w, w_gate, w_up, w_down):
    """
    tokens:   [T, D]      — flat token batch (batch * seq)
    router_w: [D, E]      — router projection
    w_gate:   [E, D, F]   — SwiGLU gate weight per expert
    w_up:     [E, D, F]   — SwiGLU up-projection per expert
    w_down:   [E, F, D]   — SwiGLU down-projection per expert

    Top-2 routing: each token selects 2 experts, outputs are weighted and summed.
    The k=2 loop unrolls statically at trace time — JAX sees two independent
    scatter/expert/gather chains.
    """
    # Router logits: [T, E]
    logits = tokens @ router_w

    # Top-2 selection: indices [T, 2], raw weights [T, 2]
    top2_weights_raw, top2_indices = jax.lax.top_k(logits, k=TOP_K)
    top2_weights = jax.nn.softmax(top2_weights_raw, axis=-1)  # normalize

    output = jnp.zeros_like(tokens)

    for k in range(TOP_K):
        # One-hot dispatch for slot k: [T, E]
        dispatch_k = jax.nn.one_hot(top2_indices[:, k], num_classes=NUM_EXPERTS)
        weight_k   = top2_weights[:, k : k + 1]   # [T, 1]

        # Scatter tokens to expert slots: [E, T, D]
        dispatched = jnp.einsum("te,td->etd", dispatch_k, tokens)

        # Expert FFN — SwiGLU
        gate       = jnp.einsum("etd,edf->etf", dispatched, w_gate)  # [E, T, F]
        up         = jnp.einsum("etd,edf->etf", dispatched, w_up)    # [E, T, F]
        hidden     = silu(gate) * up                                   # [E, T, F]
        expert_out = jnp.einsum("etf,efd->etd", hidden, w_down)       # [E, T, D]

        # Gather back and weight: [T, D]
        gathered = jnp.einsum("te,etd->td", dispatch_k, expert_out)
        output   = output + weight_k * gathered

    return output


# ---------------------------------------------------------------------------
# Export at full Mixtral dims
# ---------------------------------------------------------------------------
print(f"Exporting Mixtral MoE layer: T={T}, D={HIDDEN}, F={INTER}, E={NUM_EXPERTS}, top_k={TOP_K}")
print("(random weights, float32 — IR shape only)\n")

exported = jax.export.export(jax.jit(mixtral_moe))(
    jax.ShapeDtypeStruct((T,           HIDDEN),        jnp.float32),  # tokens
    jax.ShapeDtypeStruct((HIDDEN,      NUM_EXPERTS),   jnp.float32),  # router_w
    jax.ShapeDtypeStruct((NUM_EXPERTS, HIDDEN, INTER), jnp.float32),  # w_gate
    jax.ShapeDtypeStruct((NUM_EXPERTS, HIDDEN, INTER), jnp.float32),  # w_up
    jax.ShapeDtypeStruct((NUM_EXPERTS, INTER,  HIDDEN),jnp.float32),  # w_down
)

mlir_text = str(exported.mlir_module())
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    f.write(mlir_text)
print(f"Wrote {OUTPUT_PATH}  ({len(mlir_text):,} bytes)\n")


# ---------------------------------------------------------------------------
# IR diagnostic: find all dot_general ops and inspect batch dims
# ---------------------------------------------------------------------------
print("=" * 70)
print("dot_general analysis")
print("=" * 70)

# Match lines like:
#   %N = stablehlo.dot_general %a, %b, batching_dims = [0] x [0], ...
#   : (tensor<8x512x4096xf32>, tensor<8x4096x14336xf32>) -> tensor<8x512x14336xf32>
# The type annotation may be on the same line or the next — join lines first.
single_line = mlir_text.replace("\n", " ")
dot_re = re.compile(
    r"stablehlo\.dot_general\s+[^:]+:\s*"
    r"\(([^)]+)\)\s*->\s*([^\s,;]+)"
)
dots = dot_re.findall(single_line)

if not dots:
    print("No stablehlo.dot_general found — pattern may differ from simple_moe.")
else:
    print(f"Found {len(dots)} dot_general op(s):\n")
    for i, (operands, result) in enumerate(dots):
        types = [t.strip() for t in operands.split(",")]
        print(f"  [{i}] lhs: {types[0]}")
        print(f"       rhs: {types[1] if len(types) > 1 else '?'}")
        print(f"       out: {result}")
        # Check for expert batch dim: first dim of result should be NUM_EXPERTS
        m = re.search(r"tensor<(\d+)x", result)
        if m:
            first_dim = int(m.group(1))
            tag = f"  ← expert batch dim (size {NUM_EXPERTS})" if first_dim == NUM_EXPERTS else ""
            print(f"       first_dim={first_dim}{tag}")
        print()

# Check for per-expert function symbols — there should be none.
expert_syms = re.findall(r"@expert_\w+", mlir_text)
if expert_syms:
    print(f"WARNING: found per-expert symbols: {set(expert_syms)}")
    print("This is unexpected — expert boundaries were NOT erased.\n")
else:
    print("No @expert_N symbols found (expected — expert boundaries are erased).")
    print("This is what ExpertOutlining.cpp must recover.\n")

# Count func.func definitions.
funcs = re.findall(r"func\.func\s+(?:public|private)?\s*@(\w+)", mlir_text)
print(f"func.func symbols: {funcs}\n")


# ---------------------------------------------------------------------------
# Correctness check — mini dims to avoid materializing 4.5 GB of weights
# ---------------------------------------------------------------------------
print("=" * 70)
print("Correctness check (mini dims: T=4, D=16, F=32, E=8, top_k=2)")
print("=" * 70)

T_m, D_m, F_m, E_m = 4, 16, 32, NUM_EXPERTS
rng = np.random.default_rng(42)

tokens_np  = rng.standard_normal((T_m, D_m)).astype(np.float32)
router_np  = rng.standard_normal((D_m, E_m)).astype(np.float32)
w_gate_np  = rng.standard_normal((E_m, D_m, F_m)).astype(np.float32)
w_up_np    = rng.standard_normal((E_m, D_m, F_m)).astype(np.float32)
w_down_np  = rng.standard_normal((E_m, F_m, D_m)).astype(np.float32)


def numpy_silu(x):
    return x / (1.0 + np.exp(-x))


def numpy_moe_reference(tokens, router_w, w_gate, w_up, w_down):
    """Explicit expert loop — reference implementation for correctness."""
    T = tokens.shape[0]
    E = w_gate.shape[0]

    logits = tokens @ router_w                       # [T, E]

    # Top-2 indices via argsort
    top2_idx = np.argsort(-logits, axis=-1)[:, :2]  # [T, 2]
    top2_raw = logits[np.arange(T)[:, None], top2_idx]  # [T, 2]

    # Softmax over the two selected logits
    top2_raw -= top2_raw.max(axis=-1, keepdims=True)
    exp = np.exp(top2_raw)
    top2_w = exp / exp.sum(axis=-1, keepdims=True)  # [T, 2]

    output = np.zeros_like(tokens)

    for k in range(2):
        expert_idx = top2_idx[:, k]   # [T] — which expert each token chose
        weight_k   = top2_w[:, k]     # [T]

        for e in range(E):
            mask = expert_idx == e
            if not mask.any():
                continue
            x = tokens[mask]                          # [n, D]
            g = numpy_silu(x @ w_gate[e])            # [n, F]
            u = x @ w_up[e]                          # [n, F]
            h = g * u
            out = h @ w_down[e]                      # [n, D]
            output[mask] += weight_k[mask, None] * out

    return output


# JAX result at mini dims
jax_out = jax.jit(mixtral_moe)(
    jnp.array(tokens_np),
    jnp.array(router_np),
    jnp.array(w_gate_np),
    jnp.array(w_up_np),
    jnp.array(w_down_np),
)
jax_out_np = np.array(jax_out)

# NumPy reference
ref_out = numpy_moe_reference(tokens_np, router_np, w_gate_np, w_up_np, w_down_np)

max_err = np.abs(jax_out_np - ref_out).max()
mean_err = np.abs(jax_out_np - ref_out).mean()
print(f"max |JAX - numpy|  = {max_err:.2e}")
print(f"mean |JAX - numpy| = {mean_err:.2e}")

# float32 accumulation across expert loops produces O(1e-5) rounding error.
atol = 1e-4
if max_err < atol:
    print(f"PASS  (atol={atol})")
else:
    print(f"FAIL  (atol={atol}) — check routing or SwiGLU implementation")
    sys.exit(1)
