"""
Verify projection kernel against JAX reference.
Uses the same inputs as runProjection() in compiler/main.cpp:
  x = all 1.0,    shape (1, 512, 768)
  w = all 1/768,  shape (768, 768)
Expected: matmul(x, w) = 1.0 everywhere
"""
import jax
import jax.numpy as jnp

def attention_projection(x, w):
    return jnp.matmul(x, w)

x = jnp.ones((1, 512, 768), dtype=jnp.float32)
w = jnp.full((768, 768), 1.0 / 768, dtype=jnp.float32)

result = jax.jit(attention_projection)(x, w)

print(f"result[0][0][0] = {result[0, 0, 0]:.6e} (expected 1.0)")
print(f"result[0][0][1] = {result[0, 0, 1]:.6e} (expected 1.0)")
