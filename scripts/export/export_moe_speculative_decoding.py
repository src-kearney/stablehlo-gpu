"""
Minimal MoE transformer for speculative decoding (draft model).

Architecture:
  - 2 layers
  - hidden_dim = 256
  - 2 attention heads (head_dim = 128)
  - 4 experts per MoE FFN layer, top-1 routing
  - expert_dim = 512 (2x hidden)

Exports a forward pass on dummy token input (batch=1, seq_len=64) to
StableHLO and saves the result to mlir/stablehlo/export_moe_speculative_decoding.mlir.

Usage:
    python py/export_moe_speculative_decoding.py [--output mlir/stablehlo/export_moe_speculative_decoding.mlir]
"""

import argparse
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

VOCAB_SIZE  = 32000
HIDDEN_DIM  = 256
NUM_HEADS   = 2
HEAD_DIM    = HIDDEN_DIM // NUM_HEADS   # 128
NUM_LAYERS  = 2
NUM_EXPERTS = 4
EXPERT_DIM  = HIDDEN_DIM * 2           # 512  (FFN expansion factor = 2)
TOP_K       = 1


# ---------------------------------------------------------------------------
# Parameter tree
# ---------------------------------------------------------------------------

class AttentionParams(NamedTuple):
    wq: jax.Array   # [hidden, hidden]
    wk: jax.Array   # [hidden, hidden]
    wv: jax.Array   # [hidden, hidden]
    wo: jax.Array   # [hidden, hidden]


class MoEParams(NamedTuple):
    router: jax.Array           # [hidden, num_experts]
    w_up:   jax.Array           # [num_experts, hidden, expert_dim]
    w_down: jax.Array           # [num_experts, expert_dim, hidden]


class LayerParams(NamedTuple):
    attn:      AttentionParams
    moe:       MoEParams
    ln_attn:   jax.Array        # [hidden]  — scale for pre-attn layernorm
    ln_moe:    jax.Array        # [hidden]  — scale for pre-moe layernorm


class ModelParams(NamedTuple):
    embed:     jax.Array        # [vocab, hidden]
    layers:    list[LayerParams]
    ln_final:  jax.Array        # [hidden]
    lm_head:   jax.Array        # [hidden, vocab]


def init_params(key: jax.Array) -> ModelParams:
    """Xavier-uniform init — keeps activations in a reasonable range."""
    def w(key, *shape):
        lim = np.sqrt(6.0 / (shape[-2] + shape[-1])) if len(shape) >= 2 else 0.02
        return jax.random.uniform(key, shape, minval=-lim, maxval=lim)

    def split(key, n):
        return jax.random.split(key, n)

    keys = split(key, 20)
    ki = 0

    embed   = w(keys[ki], VOCAB_SIZE, HIDDEN_DIM); ki += 1
    ln_fin  = jnp.ones(HIDDEN_DIM)
    lm_head = w(keys[ki], HIDDEN_DIM, VOCAB_SIZE); ki += 1

    layers = []
    for _ in range(NUM_LAYERS):
        wq = w(keys[ki],   HIDDEN_DIM, HIDDEN_DIM); ki += 1
        wk = w(keys[ki],   HIDDEN_DIM, HIDDEN_DIM); ki += 1
        wv = w(keys[ki],   HIDDEN_DIM, HIDDEN_DIM); ki += 1
        wo = w(keys[ki],   HIDDEN_DIM, HIDDEN_DIM); ki += 1
        attn = AttentionParams(wq=wq, wk=wk, wv=wv, wo=wo)

        router = w(keys[ki], HIDDEN_DIM, NUM_EXPERTS);               ki += 1
        w_up   = w(keys[ki], NUM_EXPERTS, HIDDEN_DIM, EXPERT_DIM);   ki += 1
        w_down = w(keys[ki], NUM_EXPERTS, EXPERT_DIM, HIDDEN_DIM);   ki += 1
        moe = MoEParams(router=router, w_up=w_up, w_down=w_down)
        layers.append(LayerParams(
            attn=attn,
            moe=moe,
            ln_attn=jnp.ones(HIDDEN_DIM),
            ln_moe =jnp.ones(HIDDEN_DIM),
        ))

    return ModelParams(embed=embed, layers=layers, ln_final=ln_fin, lm_head=lm_head)


# ---------------------------------------------------------------------------
# Model ops
# ---------------------------------------------------------------------------

def rms_norm(x: jax.Array, scale: jax.Array, eps: float = 1e-6) -> jax.Array:
    """RMSNorm over last axis."""
    rms = jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
    return scale * x / rms


def attention(params: AttentionParams, x: jax.Array) -> jax.Array:
    """Multi-head self-attention (causal mask, no KV cache for export)."""
    B, T, _ = x.shape

    q = x @ params.wq                          # [B, T, hidden]
    k = x @ params.wk
    v = x @ params.wv

    # Reshape to [B, T, heads, head_dim] then transpose to [B, heads, T, head_dim]
    def split_heads(t):
        return t.reshape(B, T, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)

    q, k, v = split_heads(q), split_heads(k), split_heads(v)

    scale  = 1.0 / jnp.sqrt(HEAD_DIM).astype(jnp.float32)
    logits = jnp.einsum("bhid,bhjd->bhij", q, k) * scale  # [B, heads, T, T]

    # Causal mask: upper triangle = -inf
    mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
    logits = jnp.where(mask[None, None, :, :], logits, jnp.finfo(jnp.float32).min)

    weights = jax.nn.softmax(logits, axis=-1)              # [B, heads, T, T]
    out = jnp.einsum("bhij,bhjd->bhid", weights, v)        # [B, heads, T, head_dim]

    # Merge heads → [B, T, hidden]
    out = out.transpose(0, 2, 1, 3).reshape(B, T, HIDDEN_DIM)
    return out @ params.wo


def moe_ffn(params: MoEParams, x: jax.Array) -> jax.Array:
    """Top-1 MoE FFN. Each token is routed to exactly one expert."""
    B, T, H = x.shape
    x_flat = x.reshape(B * T, H)                           # [N, hidden]

    # Router: softmax scores, pick top-1 expert per token
    logits    = x_flat @ params.router                     # [N, num_experts]
    scores    = jax.nn.softmax(logits, axis=-1)            # [N, num_experts]
    expert_id = jnp.argmax(scores, axis=-1)                # [N]
    gate      = scores[jnp.arange(B * T), expert_id]      # [N]  — routing weight

    # Dispatch: for each expert, gather its tokens and run the FFN.
    # We unroll over NUM_EXPERTS (small constant) to keep this XLA-friendly
    # without dynamic shapes.
    out = jnp.zeros_like(x_flat)
    for e in range(NUM_EXPERTS):
        mask = (expert_id == e).astype(jnp.float32)[:, None]   # [N, 1]
        h = jax.nn.gelu(x_flat @ params.w_up[e])              # [N, expert_dim]
        h = h @ params.w_down[e]                               # [N, hidden]
        out = out + mask * h

    # Apply per-token routing weight
    out = out * gate[:, None]
    return out.reshape(B, T, H)


def transformer_layer(params: LayerParams, x: jax.Array) -> jax.Array:
    # Pre-norm attention
    x = x + attention(params.attn, rms_norm(x, params.ln_attn))
    # Pre-norm MoE FFN
    x = x + moe_ffn(params.moe, rms_norm(x, params.ln_moe))
    return x


def forward(params: ModelParams, token_ids: jax.Array) -> jax.Array:
    """
    Args:
        token_ids: int32 [batch, seq_len]
    Returns:
        logits: float32 [batch, seq_len, vocab_size]
    """
    x = params.embed[token_ids]                             # [B, T, hidden]
    for layer_params in params.layers:
        x = transformer_layer(layer_params, x)
    x = rms_norm(x, params.ln_final)
    return x @ params.lm_head                              # [B, T, vocab]


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def main(output_path: str) -> None:
    print("Initializing MoE draft model parameters ...")
    key    = jax.random.PRNGKey(0)
    params = init_params(key)

    # Bind params into the closure so export only traces over token_ids.
    def inference(token_ids: jax.Array) -> jax.Array:
        return forward(params, token_ids)

    batch, seq = 1, 64
    abstract_ids = jax.ShapeDtypeStruct((batch, seq), jnp.int32)

    print(f"Exporting forward pass (batch={batch}, seq_len={seq}) to StableHLO ...")
    exported = jax.export.export(jax.jit(inference))(abstract_ids)

    with open(output_path, "w") as f:
        f.write(exported.mlir_module())

    import os
    size_kb = os.path.getsize(output_path) / 1024
    print(f"Saved StableHLO to {output_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="mlir/stablehlo/export_moe_speculative_decoding.mlir",
                        help="Path for the StableHLO MLIR output")
    args = parser.parse_args()
    main(args.output)
