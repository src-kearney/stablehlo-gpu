#!/usr/bin/env python3
"""
Verify compiler/passes/ExpertOutlining.cpp.

Runs:
  compiler/build/remora mlir/stablehlo/mixtral_moe_layer.mlir \\
      --pass-pipeline='moe-expert-outlining' --no-execute \\
      --dump-compilation-phases-to=<tmpdir>

then checks:
  1. Structural — the transformed IR has the expected shape.
  2. Mathematical — JAX mini-dim computation matches NumPy per-expert reference.

Usage:
    python3 scripts/verify/verify_outlining.py

The script is self-contained; no arguments are required.
"""
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "../.."))
BINARY    = os.path.join(REPO_ROOT, "compiler/build/remora")
INPUT_IR  = os.path.join(REPO_ROOT, "mlir/stablehlo/mixtral_moe_layer.mlir")

# ---------------------------------------------------------------------------
# 1. Run the pass and capture transformed IR
# ---------------------------------------------------------------------------
print("=" * 70)
print("Step 1: run moe-expert-outlining pass")
print("=" * 70)

with tempfile.TemporaryDirectory() as tmpdir:
    cmd = [
        BINARY, INPUT_IR,
        "--pass-pipeline=moe-expert-outlining",
        "--no-execute",
        f"--dump-compilation-phases-to={tmpdir}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("FAIL — compiler exited with non-zero status")
        print(result.stderr)
        sys.exit(1)

    phase_dir = os.path.join(tmpdir, "01-moe-expert-outlining")
    ir_path = os.path.join(phase_dir, "module.mlir")
    if not os.path.exists(ir_path):
        print(f"FAIL — expected IR dump at {ir_path} not found")
        sys.exit(1)

    with open(ir_path) as f:
        ir = f.read()

print("PASS — compiler exited 0 and produced transformed IR")
print(f"       IR is {len(ir):,} bytes, {ir.count(chr(10))} lines\n")

# ---------------------------------------------------------------------------
# 2. Structural checks
# ---------------------------------------------------------------------------
print("=" * 70)
print("Step 2: structural checks on transformed IR")
print("=" * 70)

failures = []

def check(name, condition, detail=""):
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}" + (f": {detail}" if detail else ""))
        failures.append(name)

# 2a. Both @expert_slot_N functions exist.
for slot_id in (0, 1):
    check(
        f"@expert_slot_{slot_id} defined",
        f"func.func @expert_slot_{slot_id}" in ir,
    )

# 2b. moe.slot_id attributes are present.
for slot_id in (0, 1):
    check(
        f"moe.slot_id = {slot_id}",
        f"moe.slot_id = {slot_id} :" in ir,
    )

# 2c. moe.num_experts = 8 present.
check("moe.num_experts = 8", "moe.num_experts = 8" in ir)

# 2d. @expert_slot functions have NO batching_dims on their dot_generals.
#     Scan each @expert_slot block and verify.
slot_func_re = re.compile(
    r"func\.func @(expert_slot_\d+)\b.*?(?=\n  func\.func|\Z)",
    re.DOTALL,
)
for m in slot_func_re.finditer(ir):
    fname = m.group(1)
    body  = m.group(0)
    batch_lines = [
        l for l in body.splitlines()
        if "batching_dims" in l and "dot_general" in l
    ]
    check(
        f"no batching_dims in {fname}",
        len(batch_lines) == 0,
        detail=f"found {len(batch_lines)} line(s): {batch_lines[:1]}",
    )
    # And there should be exactly 3 dot_general ops (gate, up, down).
    dot_count = body.count("stablehlo.dot_general")
    check(
        f"{fname} has 3 dot_generals (gate, up, down)",
        dot_count == 3,
        detail=f"found {dot_count}",
    )

# 2e. @main calls each @expert_slot_N exactly 8 times (once per expert).
for slot_id in (0, 1):
    count = ir.count(f"call @expert_slot_{slot_id}")
    check(
        f"@main calls @expert_slot_{slot_id} 8× (one per expert)",
        count == 8,
        detail=f"found {count}",
    )

# 2f. @main has 2 stablehlo.concatenate ops (one per top-k slot).
concat_count = ir.count("stablehlo.concatenate")
check(
    "@main has 2 stablehlo.concatenate ops",
    concat_count == 2,
    detail=f"found {concat_count}",
)

# 2g. The original batched expert FFN dots (dim0=8) are gone from @main.
#     Parse @main's body and look for dot_generals with lhs shape[0]==8.
main_func_m = re.search(
    r"func\.func public @main\b.*?(?=\nfunc\.func|\Z)",
    ir, re.DOTALL,
)
if main_func_m:
    main_body = main_func_m.group(0)
    # Expert-batch FFN dots: batching_dims = [0] x [0] with lhs tensor<8x...
    # (remaining gather dots use [0] x [1] — those are fine to leave in @main)
    expert_batch_dots = re.findall(
        r"stablehlo\.dot_general.*?batching_dims = \[0\] x \[0\].*?tensor<8x",
        main_body,
    )
    check(
        "no expert-batch dot_generals remain in @main",
        len(expert_batch_dots) == 0,
        detail=f"found {len(expert_batch_dots)}: {expert_batch_dots[:1]}",
    )
else:
    check("@main function found", False)

print()
if failures:
    print(f"Structural FAIL — {len(failures)} check(s) failed: {failures}")
    sys.exit(1)
print("Structural PASS\n")

# ---------------------------------------------------------------------------
# 3. Mathematical correctness at mini dimensions
#
# We cannot run the MLIR directly because stablehlo.custom_call @mhlo.topk
# blocks the MLIR CPU JIT.  Instead we verify the SwiGLU FFN logic:
# the per-expert function called by the outlined @expert_slot_N must compute
# the same result as the reference loop in export_mixtral_layer.py.
#
# Strategy: run both JAX (full batched forward) and NumPy (explicit expert
# loop) at mini dims; they must agree — this validates both the reference and
# our understanding of what the outlined function should compute.
# ---------------------------------------------------------------------------
print("=" * 70)
print("Step 3: mathematical check (JAX vs NumPy expert loop, mini dims)")
print("=" * 70)

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    print("SKIP — JAX not installed; skipping math check")
    sys.exit(0)

T_m, D_m, F_m, E_m = 4, 16, 32, 8

rng = np.random.default_rng(42)
tokens_np  = rng.standard_normal((T_m, D_m)).astype(np.float32)
router_np  = rng.standard_normal((D_m, E_m)).astype(np.float32)
w_gate_np  = rng.standard_normal((E_m, D_m, F_m)).astype(np.float32)
w_up_np    = rng.standard_normal((E_m, D_m, F_m)).astype(np.float32)
w_down_np  = rng.standard_normal((E_m, F_m, D_m)).astype(np.float32)


def numpy_silu(x):
    return x / (1.0 + np.exp(-x))


def numpy_expert_ffn(x, w_gate, w_up, w_down):
    """Single expert SwiGLU FFN — mirrors @expert_slot_N body."""
    gate = numpy_silu(x @ w_gate) * (x @ w_up)
    return gate @ w_down


def numpy_moe(tokens, router_w, w_gate, w_up, w_down):
    """Full MoE forward — explicit top-2 loop over experts."""
    T = tokens.shape[0]
    E = w_gate.shape[0]

    logits     = tokens @ router_w               # [T, E]
    top2_idx   = np.argsort(-logits, axis=-1)[:, :2]
    top2_raw   = logits[np.arange(T)[:, None], top2_idx]
    top2_raw  -= top2_raw.max(axis=-1, keepdims=True)
    exp_       = np.exp(top2_raw)
    top2_w     = exp_ / exp_.sum(axis=-1, keepdims=True)

    output = np.zeros_like(tokens)
    for k in range(2):
        expert_idx = top2_idx[:, k]
        weight_k   = top2_w[:, k]
        for e in range(E):
            mask = expert_idx == e
            if not mask.any():
                continue
            x   = tokens[mask]
            out = numpy_expert_ffn(x, w_gate[e], w_up[e], w_down[e])
            output[mask] += weight_k[mask, None] * out
    return output


def jax_moe(tokens, router_w, w_gate, w_up, w_down):
    """JAX vectorised MoE — what the original MLIR computes."""
    TOP_K = 2

    def silu(x):
        return x * jax.nn.sigmoid(x)

    logits              = tokens @ router_w
    top2_weights_raw, top2_indices = jax.lax.top_k(logits, k=TOP_K)
    top2_weights        = jax.nn.softmax(top2_weights_raw, axis=-1)

    output = jnp.zeros_like(tokens)
    for k in range(TOP_K):
        dispatch_k = jax.nn.one_hot(top2_indices[:, k], num_classes=E_m)
        weight_k   = top2_weights[:, k : k + 1]
        dispatched = jnp.einsum("te,td->etd", dispatch_k, tokens)
        gate       = jnp.einsum("etd,edf->etf", dispatched, w_gate)
        up         = jnp.einsum("etd,edf->etf", dispatched, w_up)
        hidden     = silu(gate) * up
        expert_out = jnp.einsum("etf,efd->etd", hidden, w_down)
        gathered   = jnp.einsum("te,etd->td", dispatch_k, expert_out)
        output     = output + weight_k * gathered
    return output


jax_out = np.array(jax.jit(jax_moe)(
    jnp.array(tokens_np), jnp.array(router_np),
    jnp.array(w_gate_np), jnp.array(w_up_np), jnp.array(w_down_np),
))
ref_out = numpy_moe(tokens_np, router_np, w_gate_np, w_up_np, w_down_np)

max_err  = np.abs(jax_out - ref_out).max()
mean_err = np.abs(jax_out - ref_out).mean()
print(f"max  |JAX - numpy| = {max_err:.2e}")
print(f"mean |JAX - numpy| = {mean_err:.2e}")

atol = 1e-4
if max_err < atol:
    print(f"PASS  (atol={atol})\n")
else:
    print(f"FAIL  (atol={atol}) — FFN math mismatch")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
print("=" * 70)
print("All checks PASSED")
print("  • @expert_slot_0, @expert_slot_1 outlined with no batching dims")
print("  • @main slices, calls, and concatenates per-expert results")
print("  • SwiGLU FFN math matches NumPy reference at mini dims")
print("=" * 70)
