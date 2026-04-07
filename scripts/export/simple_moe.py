#!/usr/bin/env python3
"""
Export a tiny 2-expert MoE to StableHLO and print the IR.

Usage:
    pip install jax jaxlib
    python3 py/simple_moe.py

Then inspect the IR — key questions to answer before writing the annotation pass:
    - Are experts separate func.func ops, or one batched dot_general?
    - How is dispatch expressed: stablehlo.gather? one_hot + einsum? while loop?
    - What does argmax lower to?

Then try running the pipeline:
    compiler/build/remora mlir/stablehlo/simple_moe.mlir --emit-ptx
    (expect failures — the point is to see what breaks)
"""
import jax
import jax.numpy as jnp


def silu(x):
    return x * jax.nn.sigmoid(x)


def moe_layer(tokens, router_w, w_gate, w_up, w_down):
    """
    tokens:   [T, D]    — T tokens, each D-dimensional
    router_w: [D, E]    — router projection (learned)
    w_gate:   [E, D, F] — SwiGLU gate weight per expert
    w_up:     [E, D, F] — SwiGLU up-projection weight per expert
    w_down:   [E, F, D] — SwiGLU down-projection weight per expert

    Top-1 routing with SwiGLU expert FFN — matches Mixtral/DeepSeek architecture.
    Each expert computes: down(silu(gate(x)) * up(x))
    """
    T, D = tokens.shape
    E = w_gate.shape[0]

    # --- routing ---
    logits = tokens @ router_w                            # [T, E]
    expert_idx = jnp.argmax(logits, axis=-1)              # [T]
    dispatch = jax.nn.one_hot(expert_idx, num_classes=E)  # [T, E]

    # --- dispatch: scatter tokens to expert slots ---
    dispatched = jnp.einsum('te,td->etd', dispatch, tokens)  # [E, T, D]

    # --- expert FFN: SwiGLU (matches Mixtral/DeepSeek) ---
    # gate path:  [E, T, D] @ [E, D, F] → [E, T, F]
    gate = jnp.einsum('etd,edf->etf', dispatched, w_gate)
    # up path:    [E, T, D] @ [E, D, F] → [E, T, F]
    up   = jnp.einsum('etd,edf->etf', dispatched, w_up)
    # fuse:       silu(gate) * up         → [E, T, F]
    hidden = silu(gate) * up
    # down path:  [E, T, F] @ [E, F, D] → [E, T, D]
    expert_out = jnp.einsum('etf,efd->etd', hidden, w_down)

    # --- gather: weighted sum back per token ---
    output = jnp.einsum('te,etd->td', dispatch, expert_out)  # [T, D]
    return output


# Tiny but realistic proportions: T=8 tokens, D=32 hidden, E=2 experts, F=64 FFN dim
T, D, E, F = 8, 32, 2, 64

exported = jax.export.export(jax.jit(moe_layer))(
    jax.ShapeDtypeStruct((T, D), jnp.float32),    # tokens
    jax.ShapeDtypeStruct((D, E), jnp.float32),    # router weights
    jax.ShapeDtypeStruct((E, D, F), jnp.float32), # w_gate
    jax.ShapeDtypeStruct((E, D, F), jnp.float32), # w_up
    jax.ShapeDtypeStruct((E, F, D), jnp.float32), # w_down
)

output_path = "mlir/stablehlo/simple_moe.mlir"
with open(output_path, "w") as f:
    f.write(str(exported.mlir_module()))
print(f"wrote {output_path}")
