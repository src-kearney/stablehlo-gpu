import jax
import jax.numpy as jnp
from jax.export import export

# Simple attention layer
# Learn more here: https://jalammar.github.io/illustrated-transformer/
def attention_projection(x, w):
    return jnp.matmul(x, w)

# (1, 512, 768) — 1 sequence in the batch, 512 tokens, each token represented as a 768-dimensional vector
# Sequence of 512 word embeddings
x = jax.ShapeDtypeStruct((1, 512, 768), jnp.float32)
# (768, 768) - weight matrix, linear transformation in 768-dimensional space
w = jax.ShapeDtypeStruct((768, 768), jnp.float32)

exported = export(jax.jit(attention_projection))(x, w)
print(exported.mlir_module())
