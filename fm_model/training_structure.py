from typing import Any, Dict, Iterator, Sequence, Tuple, Optional
import time
import h5py
import numpy as np

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

import os
import pickle
from flax import serialization

import wandb


### wandb writes its run data to <WANDB_DIR>/wandb/, so this is the parent of
### the wandb folder itself, not the folder. Keeping it out of the model
### directory stops run logs from mixing in with saved weights.
WANDB_DIR = '/home/habjan.e/SuperBIT_code/Redshift_ml/Sandbox_notebooks'

### Range the smeared training target is allowed to fall in, in redshift units.
### Wider than the labels on purpose -- see the note in train_model. The floor
### is slightly negative so objects censored at the BPZ grid floor have room to
### extrapolate below it; peculiar velocities make small blueshifts physical at
### these redshifts, and predictions can be clipped at analysis time.
TARGET_Z_LO = -0.02
TARGET_Z_HI = 3.0

### Edges of the LoVoCCS BPZ redshift grid. A label sitting on one of these is
### a limit ("BPZ ran out of grid"), not a measurement, so its target is drawn
### one-sided beyond the edge instead of centred on it. DESI spec-z have no
### grid, so censoring must never be applied to the test split.
CENSOR_Z_LO = 0.01
CENSOR_Z_HI = 1.49

### Floor on the smearing width for objects whose error bar crosses an edge.
### The reported ZERR on a saturated fit is not trustworthy -- a quarter of the
### ceiling objects claim +-0.04 -- and without a floor those stay pinned to
### the edge and never extrapolate.
CENSOR_SIGMA_FLOOR = 0.05


### Catalog columns fed to the MLP branch.
###
### The HDF5 files carry 14 columns, but only these six are independent. The
### other eight are exact analytic functions of them, verified numerically on
### the source catalogs:
###   R_{b,g,u}_prepsf == 0.600000 * R_{b,g,u}
###   m_x              == 30 - 2.5*log10(FLUX_AUTO_x)   (zero point 30)
###   color_bg         == m_b - m_g,   color_ub == m_u - m_b
### Including them would triple the weight on size/brightness relative to
### color and add three constant-ratio duplicate inputs.
MLP_FEATURES: Tuple[str, ...] = ("m_b", "m_g", "m_u", "R_b", "R_g", "R_u")


def preload_hdf5_to_memory(
    file_path: str,
    feature_names: Sequence[str] = MLP_FEATURES,
    normalized: bool = True,
    include_mask: bool = True,
) -> Dict[str, Any]:
    """
    Load one split written by data_process.py fully into memory.

    The files are flat and column-oriented (one row per galaxy), so every
    dataset reads in a single slice.

    Returns, for n galaxies:
      images      (n, 51, 51, 9) float32  the nine VIGNET channels, in the
                                          order given by metadata["image_channels"]
      image_mask  (n, 51, 51, 3) float32  1.0 = good pixel, 0.0 = masked, for
                                          the three raw VIGNETs (see below)
      features    (n, F)         float32  the requested catalog columns
      z, z_err    (n, 1)         float32  training target and its uncertainty
      z_lovoccs / z_desi and their _err   both estimates, NaN where unavailable
      ra, dec, field, group_id            provenance; split on group_id, never
                                          on row index, because Abell2384a and
                                          Abell2384b share ~1100 objects
      z_mean, z_std              float    for the model's standardized space
      metadata                   dict     the file attributes

    Only one mask exists in the files. The three raw VIGNET_{b,g,u} channels
    carry a -1e30 SExtractor sentinel on ~1% of pixels, which data_process.py
    zeroed while recording where it was; the _im and _wt channels have no
    sentinel and so have no mask. image_mask is stored as bool on disk and is
    converted to 0.0/1.0 float32 here.

    normalized=True loads images_norm / features_norm (per-channel median-MAD
    scaling and per-feature z-scoring, both fit on the training split);
    normalized=False loads the raw values.
    """
    print(f"\nPreloading {file_path} into memory...")
    start = time.time()

    with h5py.File(file_path, "r") as f:
        metadata = {k: f.attrs[k] for k in f.attrs.keys()}

        image_key = "images_norm" if normalized else "images"
        feature_key = "features_norm" if normalized else "features"

        images = np.asarray(f[image_key][:], dtype=np.float32)
        n_samples = images.shape[0]

        ### Pick the requested catalog columns out of the 14 stored ones
        stored = [str(c) for c in metadata["feature_names"]]
        missing = [c for c in feature_names if c not in stored]
        if missing:
            raise KeyError(
                f"{file_path} has no catalog columns {missing}; available: {stored}"
            )
        cols = [stored.index(c) for c in feature_names]
        features = np.asarray(f[feature_key][:, cols], dtype=np.float32)

        ### Masks are 0/1 floats so they can be concatenated onto the image
        ### stack or used as a multiplicative weight without a cast
        if include_mask and "image_mask" in f:
            image_mask = np.asarray(f["image_mask"][:], dtype=np.float32)
            mask_channels = [str(c) for c in metadata.get("image_mask_channels", [])]
        else:
            image_mask = None
            mask_channels = []

        def column(name):
            return np.asarray(f[name][:], dtype=np.float32).reshape(n_samples, 1)

        z = column("z")
        z_err = column("z_err")
        redshifts = {
            name: column(name)
            for name in ("z_lovoccs", "z_lovoccs_err", "z_desi", "z_desi_err")
            if name in f
        }

        ra = np.asarray(f["ra"][:], dtype=np.float64)
        dec = np.asarray(f["dec"][:], dtype=np.float64)
        group_id = np.asarray(f["group_id"][:], dtype=np.int32)
        field = np.array([s.decode() if isinstance(s, bytes) else str(s)
                          for s in f["field"][:]])

        extras = {}
        for name in ("spectype", "at_edge", "zwarn"):
            if name in f:
                v = f[name][:]
                extras[name] = np.array(
                    [s.decode() if isinstance(s, bytes) else s for s in v]
                )

    elapsed = time.time() - start
    size_b = images.nbytes + features.nbytes + z.nbytes + z_err.nbytes
    if image_mask is not None:
        size_b += image_mask.nbytes

    print(f"✓ Loaded {n_samples} galaxies in {elapsed:.2f}s ({size_b / 1e9:.2f} GB)")
    print(f"  images   {images.shape} from '{image_key}'")
    if image_mask is not None:
        print(f"  mask     {image_mask.shape} for {mask_channels}, "
              f"{100 * (1.0 - float(image_mask.mean())):.2f}% masked")
    print(f"  features {features.shape} from '{feature_key}': {list(feature_names)}")
    print(f"  z        {metadata['redshift_column']} "
          f"min={float(z.min()):.4f} mean={float(z.mean()):.4f} "
          f"max={float(z.max()):.4f}, z_err median={float(np.median(z_err)):.5f}")

    out = dict(
        images=images,
        image_mask=image_mask,
        features=features,
        z=z,
        z_err=z_err,
        ra=ra,
        dec=dec,
        field=field,
        group_id=group_id,
        feature_names=list(feature_names),
        image_channels=[str(c) for c in metadata["image_channels"]],
        mask_channels=mask_channels,
        ### The model's flow runs on (z - z_mean) / z_std; take these from the
        ### training split and reuse them for validation and test
        z_mean=float(z.mean()),
        z_std=float(z.std()),
        metadata=metadata,
    )
    out.update(redshifts)
    out.update(extras)
    return out



### ---------------------------------------------------------------- batching


def n_rows(data: Dict[str, Any]) -> int:
    """Number of galaxies in a split, independent of which branches are used."""
    return data["z"].shape[0]


def random_dihedral(images: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply an independent random dihedral transform to each stamp.

    The eight symmetries of the square (four rotations x optional flip) are an
    exact augmentation here: galaxy orientation carries no redshift
    information, and the 51x51 stamps have an odd size so rotating about the
    centre pixel needs no interpolation and loses nothing.

    Operates on the whole (B, H, W, C) stack at once, so the mask channels stay
    aligned with the vignettes they describe. The catalog features are
    rotation-invariant scalars and are left alone.
    """
    out = np.empty_like(images)
    which = rng.integers(0, 8, size=len(images))
    for t in range(8):
        sel = which == t
        if not sel.any():
            continue
        k, flip = divmod(t, 2)
        block = np.rot90(images[sel], k=k, axes=(1, 2))
        if flip:
            block = block[:, :, ::-1, :]
        out[sel] = block
    return out


def make_batch(
    data: Dict[str, Any],
    bidx: np.ndarray,
    include_mask: bool = True,
    augment: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Any, ...]:
    """Assemble one batch as (images, features, z, z_err).

    With include_mask=True the 0/1 pixel mask is concatenated onto the image
    stack, so the CNN sees 12 channels: the nine VIGNETs followed by the three
    mask planes for VIGNET_b, VIGNET_g and VIGNET_u. The mask is an input
    rather than a multiplicative weight, so the network can tell a genuinely
    zero pixel from a pixel that was never measured.

    Setting data["images"] or data["features"] to None drops that branch, for
    single-modality ablations; the corresponding batch entry is then None,
    which the model's encode() accepts.

    augment applies a random dihedral transform to the image stack; it needs an
    rng and is a no-op when there is no image branch.
    """
    images = data.get("images")
    if images is not None:
        images = images[bidx]
        if include_mask and data.get("image_mask") is not None:
            images = np.concatenate([images, data["image_mask"][bidx]], axis=-1)
        if augment:
            if rng is None:
                raise ValueError("augment=True requires an rng")
            images = random_dihedral(images, rng)
        images = jnp.asarray(images)

    features = data.get("features")
    if features is not None:
        features = jnp.asarray(features[bidx])

    return (
        images,
        features,
        jnp.asarray(data["z"][bidx]),
        jnp.asarray(data["z_err"][bidx]),
    )


def data_loader(
    data: Dict[str, Any],
    batch_size: int,
    rng: np.random.Generator,
    shuffle: bool = True,
    include_mask: bool = True,
    drop_last: bool = False,
    augment: bool = False,
) -> Iterator[Tuple[jnp.ndarray, ...]]:
    """Iterate once over a split, yielding (images, features, z, z_err).

    drop_last=True keeps every batch the same shape, so the jitted step
    functions compile once. Use it for training; leave it False for evaluation
    so no galaxy is skipped.

    Both the shuffle order and the augmentation draw from `rng`, so passing a
    freshly seeded generator makes an evaluation pass exactly reproducible.
    """
    n = n_rows(data)
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)

    for i in range(0, n, batch_size):
        bidx = idx[i:i + batch_size]
        if len(bidx) == 0:
            continue
        if drop_last and len(bidx) < batch_size:
            continue
        yield make_batch(data, bidx, include_mask=include_mask,
                         augment=augment, rng=rng)


def infinite_data_loader(
    data: Dict[str, Any],
    batch_size: int,
    rng: np.random.Generator,
    shuffle: bool = True,
    include_mask: bool = True,
    augment: bool = False,
):
    """Endless reshuffled stream of fixed-shape training batches."""
    while True:
        yield from data_loader(
            data,
            batch_size=batch_size,
            rng=rng,
            shuffle=shuffle,
            include_mask=include_mask,
            drop_last=True,
            augment=augment,
        )


### ---------------------------------------------------------- flow matching


def truncated_normal(
    key,
    mean: jnp.ndarray,
    std: jnp.ndarray,
    lo: float,
    hi: float,
    eps: float = 1e-6,
) -> jnp.ndarray:
    """Sample N(mean, std) restricted to [lo, hi], by inverse CDF.

    Used to draw the flow-matching target from the reported redshift
    uncertainty. ZERR_lovoccs has a heavy tail -- median 0.04 but a 99th
    percentile of 0.72, about twice the spread of the redshift distribution
    itself -- so an untruncated draw puts several percent of targets at
    negative redshift or above the label ceiling. Truncating keeps the
    reported uncertainty while never producing an impossible target.
    """
    std = jnp.maximum(std, eps)
    a = jax.scipy.special.ndtr((lo - mean) / std)
    b = jax.scipy.special.ndtr((hi - mean) / std)

    u = jax.random.uniform(key, mean.shape, minval=0.0, maxval=1.0)
    p = jnp.clip(a + u * (b - a), eps, 1.0 - eps)

    return jnp.clip(mean + std * jax.scipy.special.ndtri(p), lo, hi)


def censored_target(
    key,
    y_mean: jnp.ndarray,
    y_std: jnp.ndarray,
    y_lo: float,
    y_hi: float,
    grid_lo: float,
    grid_hi: float,
    sigma_floor: float = 0.0,
) -> jnp.ndarray:
    """Draw the flow-matching target, treating grid-edge labels as limits.

    Everything is in the model's standardized units. Three cases per object:

      label on the grid floor    -> one-sided draw in [y_lo, grid_lo]
      label on the grid ceiling  -> one-sided draw in [grid_hi, y_hi]
      anything else              -> the usual draw centred on the label, free
                                    to cross an edge because y_lo / y_hi are
                                    wider than the grid

    Objects whose error bar crosses an edge get their width floored at
    sigma_floor, which is what gives them room to extrapolate past it. That is
    applied by the crossing test, not by being on the edge: a label of 1.40
    with a large error is a real measurement that should stay centred on 1.40,
    it just needs to be able to reach beyond 1.49.
    """
    at_lo = y_mean <= grid_lo
    at_hi = y_mean >= grid_hi

    crosses = ((y_mean - y_std) < grid_lo) | ((y_mean + y_std) > grid_hi)
    std = jnp.where(crosses, jnp.maximum(y_std, sigma_floor), y_std)

    ### An edge label carries no information beyond "at least"/"at most" the
    ### edge, so the edge itself becomes the location
    mean = jnp.where(at_lo, grid_lo, jnp.where(at_hi, grid_hi, y_mean))
    lo = jnp.where(at_hi, grid_hi, y_lo)
    hi = jnp.where(at_lo, grid_lo, y_hi)

    return truncated_normal(key, mean, std, lo, hi)


def uncertainty_fm_loss(
    params,
    key,
    condition,       # [batch, condition_dim]
    y_mean,          # [batch, 1]
    y_std,           # [batch, 1]
    velocity_fn,     # velocity_fn(params, y_t, t, condition)
    y_lo: float = -jnp.inf,
    y_hi: float = jnp.inf,
    grid_lo: Optional[float] = None,
    grid_hi: Optional[float] = None,
    sigma_floor: float = 0.0,
):
    """Monte Carlo flow-matching loss with uncertain targets.

    All quantities are in the model's standardized space. y_lo / y_hi bound the
    target draw; leave them at +-inf to recover the plain Gaussian version.

    Passing grid_lo / grid_hi turns on censoring: labels sitting on those edges
    are treated as limits and drawn one-sided beyond them. Leave them None for
    labels that come off a continuous scale, such as the DESI spec-z.
    """

    if y_mean.ndim == 1:
        y_mean = y_mean[:, None]
        y_std = y_std[:, None]

    key_source, key_target, key_time = jax.random.split(key, 3)

    # Sample from the reported target distribution, restricted to the
    # physically allowed range
    if grid_lo is None or grid_hi is None:
        y1 = truncated_normal(key_target, y_mean, y_std, y_lo, y_hi)
    else:
        y1 = censored_target(key_target, y_mean, y_std, y_lo, y_hi,
                             grid_lo, grid_hi, sigma_floor)

    # Base distribution
    y0 = jax.random.normal(key_source, y_mean.shape)

    # Flow time
    t = jax.random.uniform(
        key_time,
        shape=(y_mean.shape[0], 1),
        minval=1e-4,
        maxval=1.0 - 1e-4,
    )

    # Linear probability path
    y_t = (1.0 - t) * y0 + t * y1

    # Exact conditional velocity for this path
    target_velocity = y1 - y0

    predicted_velocity = velocity_fn(
        params, y_t, t, condition
    )

    per_object_loss = jnp.mean(
        (predicted_velocity - target_velocity) ** 2,
        axis=-1,
    )

    return jnp.mean(per_object_loss)


### -------------------------------------------------------- state and steps


def create_train_state(
    model,
    rng_key,
    example_batch,
    learning_rate: float = 1e-3,
    warmup_steps: int = 500,
    num_train_steps: int = 50_000,
    grad_clipping: float = 1.0,
):
    """Initialize params from one example batch and build the optimizer."""
    images_ex, features_ex, z_ex, _ = example_batch
    B = z_ex.shape[0]

    params = model.init(
        rng_key,
        images_ex,
        features_ex,
        jnp.zeros((B, 1), dtype=jnp.float32),
        jnp.zeros((B, 1), dtype=jnp.float32),
        training=False,
    )["params"]

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=learning_rate / 100.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=max(num_train_steps, warmup_steps + 1),
        end_value=learning_rate / 50.0,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(grad_clipping),
        optax.adamw(schedule, weight_decay=1e-4),
    )
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def _standardization(model):
    """Read the redshift standardization constants off the model."""
    return float(getattr(model, "z_mean", 0.0)), float(getattr(model, "z_std", 1.0))


def make_loss_fn(model, y_lo: float, y_hi: float, training: bool,
                 grid_lo: Optional[float] = None,
                 grid_hi: Optional[float] = None,
                 sigma_floor: float = 0.0):
    """Build the flow-matching loss for one model, in standardized space."""
    z_mean, z_std = _standardization(model)

    def loss_fn(params, key, images, features, z, z_err):
        key_fm, key_drop_enc, key_drop_vel = jax.random.split(key, 3)
        rngs_enc = {"dropout": key_drop_enc} if training else None
        rngs_vel = {"dropout": key_drop_vel} if training else None

        # Reported redshift and its uncertainty in the flow's own units
        y_mean = (z - z_mean) / z_std
        y_std = z_err / z_std

        condition = model.apply(
            {"params": params}, images, features, training,
            method="encode", rngs=rngs_enc,
        )

        def velocity_fn(p, y_t, t, cond):
            return model.apply(
                {"params": p}, y_t, t, cond, training,
                method="velocity", rngs=rngs_vel,
            )

        return uncertainty_fm_loss(
            params, key_fm, condition, y_mean, y_std, velocity_fn,
            y_lo=y_lo, y_hi=y_hi,
            grid_lo=grid_lo, grid_hi=grid_hi, sigma_floor=sigma_floor,
        )

    return loss_fn


def make_train_step(model, y_lo: float, y_hi: float, **censor):
    """Return a jitted training step for this model."""
    loss_fn = make_loss_fn(model, y_lo, y_hi, training=True, **censor)

    @jax.jit
    def train_step(state, images, features, z, z_err, key):
        loss, grads = jax.value_and_grad(loss_fn)(
            state.params, key, images, features, z, z_err
        )
        return state.apply_gradients(grads=grads), loss

    return train_step


def make_eval_step(model, y_lo: float, y_hi: float, **censor):
    """Return a jitted evaluation step (no dropout, no gradient)."""
    loss_fn = make_loss_fn(model, y_lo, y_hi, training=False, **censor)

    @jax.jit
    def eval_step(params, images, features, z, z_err, key):
        return loss_fn(params, key, images, features, z, z_err)

    return eval_step


### ------------------------------------------------- sampling and metrics


def make_sampler(model, n_steps: int = 50, n_samples: int = 32):
    """Return a jitted sampler that draws redshifts by solving the flow ODE.

    The encoder runs once per batch and its output is reused across every ODE
    step and every posterior draw, which is the whole point of splitting
    ``encode`` from ``velocity`` in the model.

    Integration is Heun (second order), from t=0 to t=1 in n_steps.
    Returns (n_samples, batch, 1) redshifts in physical units.
    """
    z_mean, z_std = _standardization(model)
    dt = 1.0 / n_steps

    @jax.jit
    def sample(params, images, features, key):
        condition = model.apply(
            {"params": params}, images, features, False, method="encode"
        )
        B = condition.shape[0]

        def velocity(y, t_scalar):
            t = jnp.full((B, 1), t_scalar, dtype=jnp.float32)
            return model.apply(
                {"params": params}, y, t, condition, False, method="velocity"
            )

        def one_draw(draw_key):
            y0 = jax.random.normal(draw_key, (B, 1))

            def step(y, i):
                t = i * dt
                v1 = velocity(y, t)
                v2 = velocity(y + dt * v1, t + dt)
                return y + 0.5 * dt * (v1 + v2), None

            y1, _ = jax.lax.scan(step, y0, jnp.arange(n_steps, dtype=jnp.float32))
            return y1

        draws = jax.vmap(one_draw)(jax.random.split(key, n_samples))
        return draws * z_std + z_mean

    return sample


def sample_dataset(
    sampler,
    params,
    data: Dict[str, Any],
    batch_size: int,
    key,
    include_mask: bool = True,
    augment: bool = False,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the sampler over a whole split in order.

    Returns (z_pred, z_sigma), each (n,), the posterior mean and standard
    deviation of the sampled redshifts per galaxy.

    Augmentation defaults off here even when it is on for training: the point
    estimates this feeds are used for model selection, so they should not move
    just because a stamp happened to be drawn in a different orientation.
    """
    n = n_rows(data)
    means, sigmas = [], []
    rng = np.random.default_rng(seed)

    for i in range(0, n, batch_size):
        bidx = np.arange(i, min(i + batch_size, n))
        images, features, _, _ = make_batch(data, bidx, include_mask=include_mask,
                                            augment=augment, rng=rng)
        key, subkey = jax.random.split(key)
        draws = np.asarray(sampler(params, images, features, subkey))  # (S,B,1)
        means.append(draws.mean(axis=0)[:, 0])
        sigmas.append(draws.std(axis=0)[:, 0])

    return np.concatenate(means), np.concatenate(sigmas)


def photoz_metrics(z_pred: np.ndarray, z_true: np.ndarray) -> Dict[str, float]:
    """Standard photometric-redshift quality metrics.

    dz = (z_pred - z_true) / (1 + z_true)
      bias      median dz
      nmad      1.4826 * median |dz - median dz|, the outlier-robust scatter
      outlier   fraction with |dz| > 0.15, the usual catastrophic-failure cut
    """
    z_pred = np.asarray(z_pred).reshape(-1)
    z_true = np.asarray(z_true).reshape(-1)
    dz = (z_pred - z_true) / (1.0 + z_true)
    bias = float(np.median(dz))

    return dict(
        bias=bias,
        nmad=float(1.4826 * np.median(np.abs(dz - bias))),
        outlier_frac=float(np.mean(np.abs(dz) > 0.15)),
        rmse=float(np.sqrt(np.mean((z_pred - z_true) ** 2))),
        mean_offset=float(np.mean(dz)),
    )


### ------------------------------------------------------------- training


def train_model(
    train_data: Dict[str, Any],
    val_data: Dict[str, Any],
    model,
    test_data: Optional[Dict[str, Any]] = None,
    batch_size: int = 64,
    num_train_steps: int = 20_000,
    eval_every: int = 250,
    sample_every: int = 500,
    log_every: int = 100,
    num_eval_batches: Optional[int] = None,
    learning_rate: float = 1e-3,
    warmup_steps: int = 500,
    grad_clipping: float = 1.0,
    include_mask: bool = True,
    augment: bool = True,
    augment_val: bool = True,
    target_z_lo: float = TARGET_Z_LO,
    target_z_hi: float = TARGET_Z_HI,
    censor: bool = True,
    censor_z_lo: float = CENSOR_Z_LO,
    censor_z_hi: float = CENSOR_Z_HI,
    censor_sigma_floor: float = CENSOR_SIGMA_FLOOR,
    n_ode_steps: int = 50,
    n_posterior_samples: int = 32,
    early_stopping_patience: Optional[int] = 10,
    seed: int = 42,
    wandb_project: str = "superbit-redshift-flow",
    wandb_mode: str = "online",
    wandb_dir: str = WANDB_DIR,
    wandb_notes: str = "",
    cfg_dict: Optional[dict] = None,
    checkpoint_dir: str = "./runtime_checkpoints",
    checkpoint_prefix: str = "redshift_flow",
    max_runtime_hours: float = 7.8,
    runtime_buffer_minutes: float = 10.0,
):
    """Train the redshift flow-matching model.

    Two signals are tracked on validation. The flow-matching loss is computed
    every eval_every steps and is cheap. Every sample_every steps the flow ODE
    is solved over the whole validation split to produce actual redshifts, and
    the photo-z metrics are computed from those; early stopping and
    best-checkpoint selection use sigma_NMAD from that second signal, because
    the flow-matching MSE can improve while point-estimate quality does not.

    test_data is optional and is **monitoring only**. Its loss and metrics are
    logged and returned so the photo-z -> spec-z transfer can be watched, but
    nothing about it feeds the gradient, the early-stopping decision or the
    choice of best checkpoint. Its stamps are never augmented.
    """
    rng_key = jax.random.PRNGKey(seed)
    rng_key, init_key = jax.random.split(rng_key)

    train_rng = np.random.default_rng(seed)

    os.makedirs(checkpoint_dir, exist_ok=True)

    ### Bounds on the smeared training target, in the model's standardized
    ### units. These deliberately do NOT track the label range: pinning the
    ### upper bound to max(z) meant the 176 objects sitting on the BPZ grid
    ### ceiling at 1.49 could only ever be smeared downward, shifting their
    ### targets by -0.21 on average and compressing the high-z end. Headroom
    ### above the ceiling makes that smearing symmetric. The lower bound is
    ### the one that should bite -- negative redshift is unphysical.
    z_mean, z_std = _standardization(model)
    y_lo = (target_z_lo - z_mean) / z_std
    y_hi = (target_z_hi - z_mean) / z_std

    label_lo = float(train_data["z"].min())
    label_hi = float(train_data["z"].max())

    ### Censoring is a property of the BPZ grid, so it applies to the LoVoCCS
    ### train/validation labels only. The DESI test labels are continuous and
    ### reach z = 0.0001, so applying the same rule there would wrongly treat
    ### genuine low-z spec-z as limits.
    censor_cfg = {}
    if censor:
        censor_cfg = dict(
            grid_lo=(censor_z_lo - z_mean) / z_std,
            grid_hi=(censor_z_hi - z_mean) / z_std,
            sigma_floor=censor_sigma_floor / z_std,
        )

    n_channels = 0
    if train_data.get("images") is not None:
        n_channels = train_data["images"].shape[-1]
        if include_mask and train_data.get("image_mask") is not None:
            n_channels += train_data["image_mask"].shape[-1]

    print(f"\nTraining on {n_rows(train_data)} galaxies, "
          f"validating on {n_rows(val_data)}")
    if n_channels:
        print(f"  image channels {n_channels}"
              f"{' (9 vignettes + 3 masks)' if include_mask else ''}")
    else:
        print("  no image branch")
    if train_data.get("features") is not None:
        print(f"  catalog features {train_data['features'].shape[-1]} "
              f"{train_data['feature_names']}")
    else:
        print("  no catalog branch")
    print(f"  labels span z [{label_lo:.4f}, {label_hi:.4f}]; "
          f"target draws bounded to z [{target_z_lo:.2f}, {target_z_hi:.2f}] "
          f"= standardized [{y_lo:.3f}, {y_hi:.3f}]")
    print(f"  z_mean={z_mean:.4f} z_std={z_std:.4f}")
    if censor:
        n_lo = int((train_data["z"] <= censor_z_lo).sum())
        n_hi = int((train_data["z"] >= censor_z_hi).sum())
        n_cross = int((((train_data["z"] - train_data["z_err"]) < censor_z_lo) |
                       ((train_data["z"] + train_data["z_err"]) > censor_z_hi)).sum())
        print(f"  censoring on: {n_lo} labels on the z={censor_z_lo} floor and "
              f"{n_hi} on the z={censor_z_hi} ceiling drawn one-sided beyond it; "
              f"{n_cross} edge-crossers get sigma floored at {censor_sigma_floor}")
    else:
        print("  censoring off")
    print(f"  dihedral augmentation: train={augment} validation={augment_val} "
          f"test=False")
    if test_data is not None:
        print(f"  monitoring {n_rows(test_data)} test galaxies "
              f"({test_data['metadata']['redshift_column']}) -- logged only, "
              f"never used for selection")

    train_step = make_train_step(model, y_lo, y_hi, **censor_cfg)
    eval_step = make_eval_step(model, y_lo, y_hi, **censor_cfg)
    ### No censoring on the DESI labels
    eval_step_test = make_eval_step(model, y_lo, y_hi)
    sampler = make_sampler(model, n_steps=n_ode_steps, n_samples=n_posterior_samples)

    train_stream = infinite_data_loader(
        train_data, batch_size, rng=train_rng, shuffle=True,
        include_mask=include_mask, augment=augment,
    )
    example_batch = next(
        data_loader(
            train_data,
            batch_size=min(batch_size, 4),
            rng=train_rng,
            shuffle=False,
            include_mask=include_mask,
        )
    )

    state = create_train_state(
        model,
        init_key,
        example_batch,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        num_train_steps=num_train_steps,
        grad_clipping=grad_clipping,
    )
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"  {n_params:,} parameters")

    ### cfg_dict is typically vars(args) from the calling script, which repeats
    ### several of these keys; let it override rather than collide
    wandb_config = dict(
        learning_rate=float(learning_rate),
        warmup_steps=int(warmup_steps),
        grad_clipping=float(grad_clipping),
        batch_size=int(batch_size),
        num_train_steps=int(num_train_steps),
        include_mask=bool(include_mask),
        n_image_channels=int(n_channels),
        feature_names=list(train_data["feature_names"]),
        n_ode_steps=int(n_ode_steps),
        n_posterior_samples=int(n_posterior_samples),
        z_mean=z_mean,
        z_std=z_std,
        target_z_lo=float(target_z_lo),
        censor=bool(censor),
        censor_z_lo=float(censor_z_lo),
        censor_z_hi=float(censor_z_hi),
        censor_sigma_floor=float(censor_sigma_floor),
        target_z_hi=float(target_z_hi),
        label_z_lo=label_lo,
        label_z_hi=label_hi,
        n_params=int(n_params),
        notes=wandb_notes,
    )
    if cfg_dict:
        wandb_config.update(cfg_dict)

    os.makedirs(wandb_dir, exist_ok=True)
    run = wandb.init(
        entity="erichabjan-northeastern-university",
        project=wandb_project,
        mode=wandb_mode,
        dir=wandb_dir,
        config=wandb_config,
    )
    print(f"  wandb run data -> {os.path.join(wandb_dir, 'wandb')}")

    def save_checkpoint(state, step, reason, tag=""):
        """Saves the full TrainState as flax bytes plus a pickled metadata blob.

        The running best is written to a fixed filename so it overwrites itself
        rather than leaving one copy per improvement; its step is recorded in
        the metadata. Terminal checkpoints keep the step in the filename.
        """
        name = f"{checkpoint_prefix}{tag}"
        suffix = "" if tag else f"_step{step}"
        state_path = os.path.join(checkpoint_dir, f"{name}_state{suffix}.msgpack")
        meta_path = os.path.join(checkpoint_dir, f"{name}_meta{suffix}.pkl")

        with open(state_path, "wb") as f:
            f.write(serialization.to_bytes(state))

        meta = dict(
            step=int(step),
            train_losses=np.asarray(train_losses, dtype=np.float32),
            val_losses=np.asarray(val_losses, dtype=np.float32),
            test_losses=np.asarray(test_losses, dtype=np.float32),
            metrics_history=metrics_history,
            test_metrics_history=test_metrics_history,
            z_mean=z_mean,
            z_std=z_std,
            target_z_lo=float(target_z_lo),
            censor=bool(censor),
            censor_z_lo=float(censor_z_lo),
            censor_z_hi=float(censor_z_hi),
            censor_sigma_floor=float(censor_sigma_floor),
            target_z_hi=float(target_z_hi),
            label_z_lo=label_lo,
            label_z_hi=label_hi,
            include_mask=bool(include_mask),
            feature_names=list(train_data["feature_names"]),
            reason=str(reason),
            wandb_run_id=None if run is None else run.id,
        )
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)

        print(f"\nSaved checkpoint ({reason}):")
        print(f"  state: {state_path}")
        print(f"  meta : {meta_path}")

    def eval_loss(data, aug, rng_seed, step_fn=None):
        """Mean flow-matching loss over a split.

        The generator is reseeded on every call so the shuffle order and any
        augmentation are identical across evaluations, which keeps the curve
        comparable from step to step.
        """
        total, count = 0.0, 0
        it = data_loader(
            data, batch_size=batch_size, rng=np.random.default_rng(rng_seed),
            shuffle=False, include_mask=include_mask, augment=aug,
        )
        fn = eval_step if step_fn is None else step_fn
        for k, batch in enumerate(it):
            if num_eval_batches is not None and k >= num_eval_batches:
                break
            loss = fn(state.params, *batch, jax.random.PRNGKey(1000 + k))
            total += float(loss)
            count += 1
        return total / max(count, 1)

    def eval_metrics(data, step, offset):
        """Solve the flow ODE over a split and score the resulting redshifts."""
        z_pred, z_sigma = sample_dataset(
            sampler, state.params, data, batch_size,
            jax.random.PRNGKey(offset + step), include_mask=include_mask,
        )
        m = photoz_metrics(z_pred, data["z"])
        m["posterior_sigma"] = float(np.mean(z_sigma))
        return m

    train_losses, val_losses, test_losses = [], [], []
    metrics_history, test_metrics_history = [], []
    best_nmad, best_step, evals_without_improvement = np.inf, -1, 0

    start_time = time.time()
    soft_limit_seconds = max_runtime_hours * 3600.0
    buffer_seconds = runtime_buffer_minutes * 60.0

    for step in range(1, num_train_steps + 1):
        batch = next(train_stream)
        rng_key, step_key = jax.random.split(rng_key)

        state, loss_train = train_step(state, *batch, step_key)
        loss_train = float(loss_train)
        train_losses.append(loss_train)
        run.log({"train_loss": loss_train}, step=step)

        if step % log_every == 0:
            elapsed_hours = (time.time() - start_time) / 3600.0
            print(f"Step {step} | train_loss: {loss_train:.6f} | "
                  f"elapsed: {elapsed_hours:.2f} hr")

        if step % eval_every == 0:
            loss_val = eval_loss(val_data, augment_val, seed + 1)
            val_losses.append(loss_val)
            logged = {"val_loss": loss_val}
            msg = f"Step {step} | val_loss: {loss_val:.6f}"

            ### Monitoring only -- never compared against, never selected on
            if test_data is not None:
                loss_test = eval_loss(test_data, False, seed + 2, eval_step_test)
                test_losses.append(loss_test)
                logged["test_loss"] = loss_test
                msg += f" | test_loss: {loss_test:.6f}"

            print(msg)
            run.log(logged, step=step)

        if step % sample_every == 0:
            m = eval_metrics(val_data, step, 2000)
            m["step"] = step
            metrics_history.append(m)
            print(f"Step {step} | nmad: {m['nmad']:.4f} "
                  f"outliers: {100 * m['outlier_frac']:.1f}% "
                  f"bias: {m['bias']:+.4f} rmse: {m['rmse']:.4f} "
                  f"posterior sigma: {m['posterior_sigma']:.4f}")
            run.log({f"val_{k}": v for k, v in m.items() if k != "step"}, step=step)

            if test_data is not None:
                tm = eval_metrics(test_data, step, 3000)
                tm["step"] = step
                test_metrics_history.append(tm)
                print(f"Step {step} | TEST nmad: {tm['nmad']:.4f} "
                      f"outliers: {100 * tm['outlier_frac']:.1f}% "
                      f"bias: {tm['bias']:+.4f} rmse: {tm['rmse']:.4f}")
                run.log({f"test_{k}": v for k, v in tm.items() if k != "step"},
                        step=step)

            if m["nmad"] < best_nmad:
                best_nmad, best_step = m["nmad"], step
                evals_without_improvement = 0
                save_checkpoint(state, step, "best_nmad", tag="_best")
            else:
                evals_without_improvement += 1
                if (early_stopping_patience is not None
                        and evals_without_improvement >= early_stopping_patience):
                    print(f"\nEarly stopping at step {step}: no sigma_NMAD "
                          f"improvement in {evals_without_improvement} sampled evals "
                          f"(best {best_nmad:.4f} at step {best_step})")
                    save_checkpoint(state, step, "early_stopping")
                    run.finish()
                    return (state, np.asarray(train_losses),
                            np.asarray(val_losses), np.asarray(test_losses),
                            metrics_history, test_metrics_history)

        # Check walltime after the step has fully completed
        elapsed_seconds = time.time() - start_time
        if soft_limit_seconds - elapsed_seconds <= buffer_seconds:
            print(f"\nApproaching runtime limit: "
                  f"elapsed={elapsed_seconds / 3600.0:.2f} hr")
            if step % sample_every != 0:
                m = eval_metrics(val_data, step, 2000)
                m["step"] = step
                metrics_history.append(m)
                run.log({f"val_{k}": v for k, v in m.items() if k != "step"}, step=step)
                if test_data is not None:
                    tm = eval_metrics(test_data, step, 3000)
                    tm["step"] = step
                    test_metrics_history.append(tm)
                    run.log({f"test_{k}": v for k, v in tm.items() if k != "step"},
                            step=step)
            save_checkpoint(state, step, "approaching_walltime")
            run.finish()
            return (state, np.asarray(train_losses), np.asarray(val_losses),
                    np.asarray(test_losses), metrics_history,
                    test_metrics_history)

    save_checkpoint(state, num_train_steps, "finished_training")
    print(f"\nBest sigma_NMAD {best_nmad:.4f} at step {best_step}")
    run.finish()
    return (state, np.asarray(train_losses), np.asarray(val_losses),
            np.asarray(test_losses), metrics_history, test_metrics_history)
