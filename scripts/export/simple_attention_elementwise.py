import jax
import jax.numpy as jnp
from jax.export import export

# Elementwise ops applied after attention projection
# relu(x + bias) - adds a bias to each token embedding, then clamps negatives to 0
# Learn more here: https://jalammar.github.io/illustrated-transformer/
def elementwise(x, bias):
    return jax.nn.relu(x + bias)

# (1, 512, 768) — 1 sequence in the batch, 512 tokens, each token represented as a 768-dimensional vector
# Sequence of 512 word embeddings
x = jax.ShapeDtypeStruct((1, 512, 768), jnp.float32)
# (768,) bias vector, one value per hidden dimension
bias = jax.ShapeDtypeStruct((768,), jnp.float32)

exported = export(jax.jit(elementwise))(x, bias)
print(exported.mlir_module())
