import jax
import jax.numpy as jnp
from jax import random

# Überprüfen, ob JAX die GPU verwendet
gpu_available = jax.device_count('gpu') > 0

if gpu_available:
    print("JAX verwendet die GPU.")
else:
    print("JAX verwendet die CPU.")

from jax import grad, jit, vmap
from jax import random

key = random.PRNGKey(0)
x = random.normal(key, (10,))
print(x)

size = 3000
x = random.normal(key, (size, size), dtype=jnp.float32)
print(x)



