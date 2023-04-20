
import jax
import jax.numpy as jnp
import flax.linen as nn


def get_sinusoidal_embeddings(
        timesteps: jax.Array,
        embedding_dim: int,
        freq_shift: float = 1,
        min_timescale: float = 1,
        max_timescale: float = 1.0e4,
        flip_sin_to_cos: bool = False,
        scale: float = 1.0,
        dtype: jnp.dtype = jnp.float32
) -> jax.Array:
    assert timesteps.ndim == 1, "Timesteps should be a 1d-array"
    assert embedding_dim % 2 == 0, f"Embedding dimension {embedding_dim} should be even"
    num_timescales = dtype(embedding_dim // 2)
    log_timescale_increment = jnp.log(max_timescale / min_timescale) / (num_timescales - freq_shift)
    inv_timescales = min_timescale * jnp.exp(jnp.arange(num_timescales, dtype = dtype) * -log_timescale_increment)
    emb = jnp.expand_dims(timesteps, 1) * jnp.expand_dims(inv_timescales, 0)

    # scale embeddings
    scaled_time = scale * emb

    if flip_sin_to_cos:
        signal = jnp.concatenate([jnp.cos(scaled_time), jnp.sin(scaled_time)], axis = 1)
    else:
        signal = jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis = 1)
    signal = jnp.reshape(signal, [jnp.shape(timesteps)[0], embedding_dim])
    return signal


class TimestepEmbedding(nn.Module):
    time_embed_dim: int = 32
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, temb: jax.Array) -> jax.Array:
        temb = nn.Dense(self.time_embed_dim, dtype = self.dtype, name = "linear_1")(temb)
        temb = nn.silu(temb)
        temb = nn.Dense(self.time_embed_dim, dtype = self.dtype, name = "linear_2")(temb)
        return temb


class Timesteps(nn.Module):
    dim: int = 32
    flip_sin_to_cos: bool = False
    freq_shift: float = 1
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, timesteps: jax.Array) -> jax.Array:
        return get_sinusoidal_embeddings(
                timesteps = timesteps,
                embedding_dim = self.dim,
                flip_sin_to_cos = self.flip_sin_to_cos,
                freq_shift = self.freq_shift,
                dtype = self.dtype
        )

