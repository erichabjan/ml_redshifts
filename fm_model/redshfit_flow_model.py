"""Conditional flow-matching architecture for galaxy redshift estimation.

The model has three parts:

1. ``ImageEncoder``   - a CNN over the postage stamps, ``(B, H, W, C)`` for any
   number of channels ``C``.
2. ``CatalogEncoder`` - an MLP over the catalog features, ``(B, F)`` for any
   number of features ``F``.
3. ``VelocityHead``   - the flow-matching head. It sees the fused image+catalog
   embedding and the flow time, and predicts the velocity of a scalar redshift
   variable.

The flow runs in *standardized* redshift space, ``y = (z - z_mean) / z_std``,
so the target sits on the same scale as the ``N(0, 1)`` base distribution.
Use :meth:`RedshiftFlowModel.standardize` / :meth:`unstandardize` to convert.

Three entry points are exposed so the encoder can be run once and reused across
every step of an ODE solve, rather than re-running the CNN at each step:

    variables = model.init(key, images=..., features=..., y_t=..., t=...)

    cond = model.apply(variables, images, features, training=False,
                       method=RedshiftFlowModel.encode)
    v    = model.apply(variables, y_t, t, cond, training=False,
                       method=RedshiftFlowModel.velocity)

    # or, in one call (re-runs the encoder):
    v    = model.apply(variables, images, features, y_t, t, training=False)

``velocity`` matches the ``velocity_fn(params, y_t, t, condition)`` signature
used by the flow-matching loss in ``training_structure.py``.

Normalization is GroupNorm rather than BatchNorm, so the model carries no
mutable ``batch_stats`` collection and ``apply`` stays a pure function of
``params``. When ``dropout_rate > 0`` and ``training=True``, ``apply`` needs a
dropout key: ``rngs={"dropout": key}``. Evaluation and ODE sampling run with
``training=False`` and need no rngs.
"""

import math
from typing import Optional, Sequence

import jax.numpy as jnp
from flax import linen as nn


def _groups(features: int, max_groups: int = 32) -> int:
    """Largest group count <= max_groups that divides the channel count."""
    return math.gcd(features, max_groups)


def sinusoidal_embedding(t: jnp.ndarray, dim: int, max_period: float = 1e4) -> jnp.ndarray:
    """Standard sinusoidal embedding of the flow time.

    t: (B,) in [0, 1]  ->  (B, dim)
    """
    half = dim // 2
    freqs = jnp.exp(
        -math.log(max_period) * jnp.arange(half, dtype=jnp.float32) / max(half - 1, 1)
    )
    args = t[:, None].astype(jnp.float32) * freqs[None, :]
    emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
    if dim % 2:
        emb = jnp.concatenate([emb, jnp.zeros((emb.shape[0], 1), emb.dtype)], axis=-1)
    return emb


class ImageEncoder(nn.Module):
    """CNN over the postage stamps, producing one embedding vector per galaxy.

    Accepts ``(B, H, W, C)`` for any ``C`` -- the channel count is inferred by
    the first convolution, so the same architecture takes 3 bands, the 9
    channels written by ``data_process.py``, or those 9 plus mask channels.

    Each stage is a stride-1 conv followed by a stride-2 conv, then global
    average pooling, matching the downsampling pattern of the existing CNNs.
    """

    features: Sequence[int] = (24, 48, 96)
    embed_dim: int = 128
    dropout_rate: float = 0.15
    ### Channel (Dropout2d-style) dropout inside the conv stack. Plain dropout
    ### after global average pooling barely regularizes a CNN, because it only
    ### sees an already-pooled vector.
    spatial_dropout_rate: float = 0.1
    use_norm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        if x.ndim != 4:
            raise ValueError(f"images must be (B, H, W, C), got shape {x.shape}")

        def conv(x, feats, stride):
            x = nn.Conv(
                feats,
                kernel_size=(3, 3),
                strides=(stride, stride),
                padding="SAME",
                kernel_init=nn.initializers.xavier_uniform(),
            )(x)
            if self.use_norm:
                x = nn.GroupNorm(num_groups=_groups(feats))(x)
            return nn.gelu(x)

        ### The channels span several orders of magnitude even after the
        ### per-channel scaling applied in data_process.py, so normalize first.
        if self.use_norm:
            x = nn.GroupNorm(num_groups=_groups(x.shape[-1]))(x)

        for feats in self.features:
            x = conv(x, feats, stride=1)
            x = conv(x, feats, stride=2)
            ### broadcast over H and W so whole feature maps drop out together
            x = nn.Dropout(
                self.spatial_dropout_rate,
                broadcast_dims=(1, 2),
                deterministic=not training,
            )(x)

        ### Global average pooling keeps the head independent of stamp size
        x = jnp.mean(x, axis=(1, 2))

        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        x = nn.Dense(self.embed_dim, kernel_init=nn.initializers.xavier_uniform())(x)
        return nn.gelu(x)


class CatalogEncoder(nn.Module):
    """MLP over the catalog features, producing one embedding vector per galaxy.

    Accepts ``(B, F)`` for any ``F``; the input width is inferred by the first
    dense layer.
    """

    features: Sequence[int] = (64, 64)
    embed_dim: int = 128
    dropout_rate: float = 0.15
    use_norm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        if x.ndim != 2:
            raise ValueError(f"features must be (B, F), got shape {x.shape}")

        for feats in self.features:
            x = nn.Dense(feats, kernel_init=nn.initializers.xavier_uniform())(x)
            if self.use_norm:
                x = nn.LayerNorm()(x)
            x = nn.gelu(x)
            x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)

        x = nn.Dense(self.embed_dim, kernel_init=nn.initializers.xavier_uniform())(x)
        return nn.gelu(x)


class AdaLNBlock(nn.Module):
    """Residual MLP block modulated by the conditioning vector (adaLN-Zero).

    The conditioning enters as a per-block scale/shift on the normalized
    activations plus an output gate. The gate is zero-initialized, so the block
    starts as the identity and the network begins training as a well-behaved
    shallow model regardless of depth.
    """

    width: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, h: jnp.ndarray, cond: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        ### scale, shift and gate for this block
        mod = nn.Dense(
            3 * self.width,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(nn.gelu(cond))
        scale, shift, gate = jnp.split(mod, 3, axis=-1)

        r = nn.LayerNorm(use_scale=False, use_bias=False)(h)
        r = r * (1.0 + scale) + shift

        r = nn.Dense(self.width, kernel_init=nn.initializers.xavier_uniform())(r)
        r = nn.gelu(r)
        r = nn.Dropout(self.dropout_rate, deterministic=not training)(r)
        r = nn.Dense(self.width, kernel_init=nn.initializers.xavier_uniform())(r)

        return h + gate * r


class VelocityHead(nn.Module):
    """Predicts d y / d t for the scalar redshift variable.

    Takes the current point on the probability path ``y_t``, the flow time
    ``t``, and the fused conditioning vector, and returns a velocity of the
    same shape as ``y_t``.
    """

    width: int = 112
    depth: int = 3
    time_embed_dim: int = 64
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        y_t: jnp.ndarray,
        t: jnp.ndarray,
        cond: jnp.ndarray,
        training: bool = True,
    ) -> jnp.ndarray:
        ### Flow time may arrive as (B,) or (B, 1)
        t_flat = t.reshape(-1)
        t_emb = sinusoidal_embedding(t_flat, self.time_embed_dim)
        t_emb = nn.Dense(self.width, kernel_init=nn.initializers.xavier_uniform())(t_emb)
        t_emb = nn.gelu(t_emb)
        t_emb = nn.Dense(self.width, kernel_init=nn.initializers.xavier_uniform())(t_emb)

        ### Time and the galaxy embedding jointly modulate every block
        c = nn.Dense(self.width, kernel_init=nn.initializers.xavier_uniform())(cond)
        c = c + t_emb

        h = nn.Dense(self.width, kernel_init=nn.initializers.xavier_uniform())(y_t)
        for _ in range(self.depth):
            h = AdaLNBlock(width=self.width, dropout_rate=self.dropout_rate)(
                h, c, training=training
            )

        h = nn.LayerNorm()(h)
        ### Zero-initialized output: the field starts at zero velocity
        return nn.Dense(
            y_t.shape[-1],
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(h)


class RedshiftFlowModel(nn.Module):
    """CNN + MLP encoder fused into a conditional flow-matching head.

    The image and catalog branches are independently optional: pass ``None``
    for either one to run an ablation with only the other. At least one must be
    provided.

    Attributes:
        z_mean, z_std: standardization constants for the redshift. The flow
            operates on ``y = (z - z_mean) / z_std``.
    """

    ### Image branch
    cnn_features: Sequence[int] = (24, 48, 96)
    ### Catalog branch
    mlp_features: Sequence[int] = (64, 64)
    ### Fused conditioning vector. Keeping this small matters: a wide condition
    ### uniquely identifies each of the ~7.8k training galaxies and lets the
    ### velocity head behave as a lookup table.
    embed_dim: int = 128
    ### Flow-matching head. The adaLN modulation costs 3*width per block, so
    ### this is where parameters accumulate fastest for what is only a scalar
    ### -> scalar map; keep it narrow.
    velocity_width: int = 112
    velocity_depth: int = 3
    time_embed_dim: int = 64

    dropout_rate: float = 0.15
    spatial_dropout_rate: float = 0.1
    head_dropout_rate: float = 0.1
    use_norm: bool = True

    ### Redshift standardization; set these from the training split
    z_mean: float = 0.0
    z_std: float = 1.0

    def setup(self):
        self.image_encoder = ImageEncoder(
            features=self.cnn_features,
            embed_dim=self.embed_dim,
            dropout_rate=self.dropout_rate,
            spatial_dropout_rate=self.spatial_dropout_rate,
            use_norm=self.use_norm,
        )
        self.catalog_encoder = CatalogEncoder(
            features=self.mlp_features,
            embed_dim=self.embed_dim,
            dropout_rate=self.dropout_rate,
            use_norm=self.use_norm,
        )
        self.fusion = nn.Dense(
            self.embed_dim, kernel_init=nn.initializers.xavier_uniform()
        )
        self.fusion_norm = nn.LayerNorm()
        self.velocity_head = VelocityHead(
            width=self.velocity_width,
            depth=self.velocity_depth,
            time_embed_dim=self.time_embed_dim,
            dropout_rate=self.head_dropout_rate,
        )

    def standardize(self, z: jnp.ndarray) -> jnp.ndarray:
        """Redshift -> the space the flow operates in."""
        return (z - self.z_mean) / self.z_std

    def unstandardize(self, y: jnp.ndarray) -> jnp.ndarray:
        """The space the flow operates in -> redshift."""
        return y * self.z_std + self.z_mean

    def encode(
        self,
        images: Optional[jnp.ndarray] = None,
        features: Optional[jnp.ndarray] = None,
        training: bool = True,
    ) -> jnp.ndarray:
        """Fuse the image and catalog branches into one conditioning vector.

        Returns ``(B, embed_dim)``. Run this once per batch and reuse the
        result across ODE steps -- the CNN is the expensive part.
        """
        if images is None and features is None:
            raise ValueError("at least one of images or features must be provided")

        parts = []
        if images is not None:
            parts.append(self.image_encoder(images, training=training))
        if features is not None:
            parts.append(self.catalog_encoder(features, training=training))

        h = parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=-1)
        return self.fusion_norm(self.fusion(h))

    def velocity(
        self,
        y_t: jnp.ndarray,
        t: jnp.ndarray,
        cond: jnp.ndarray,
        training: bool = True,
    ) -> jnp.ndarray:
        """Velocity at ``(y_t, t)`` given a precomputed conditioning vector.

        y_t: (B, 1) point on the probability path, in standardized space
        t:   (B,) or (B, 1) flow time in [0, 1]
        cond: (B, embed_dim) from :meth:`encode`
        """
        if y_t.ndim == 1:
            y_t = y_t[:, None]
        return self.velocity_head(y_t, t, cond, training=training)

    def __call__(
        self,
        images: Optional[jnp.ndarray],
        features: Optional[jnp.ndarray],
        y_t: jnp.ndarray,
        t: jnp.ndarray,
        training: bool = True,
    ) -> jnp.ndarray:
        """Encode and predict the velocity in one pass.

        Convenient for ``init`` and for the training loss, where the encoder is
        evaluated once per step anyway.
        """
        cond = self.encode(images, features, training=training)
        return self.velocity(y_t, t, cond, training=training)
