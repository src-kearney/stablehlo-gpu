"""
Verify elementwise kernel against JAX reference.
Uses the same inputs as runElementwise() in compiler/main.cpp:
  x    = all 1.0,  shape (1, 512, 768)
  bias = all -0.5, shape (768,)
Expected: relu(x + bias) = 0.5 everywhere
"""
import jax
import jax.numpy as jnp

def elementwise(x, bias):
    return jax.nn.relu(x + bias)

x    = jnp.ones((1, 512, 768), dtype=jnp.float32)
bias = jnp.full((768,), -0.5, dtype=jnp.float32)

result = jax.jit(elementwise)(x, bias)

print(f"result[0][0][0] = {result[0, 0, 0]:.6e} (expected 0.5)")
print(f"result[0][0][1] = {result[0, 0, 1]:.6e} (expected 0.5)")
