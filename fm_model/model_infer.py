"""Sample a trained redshift flow model on the train, validation and test splits.

Rebuilds the model from the config written by train_redshift_flow.py, loads the
weights, and solves the flow ODE for every galaxy in each split to draw a full
posterior over its redshift.

For each split three files are written to OUT_DIR:
    {run}_{split}_samples.npy       (n, n_samples) float32, redshift draws
    {run}_{split}_z_true.npy        (n,)           float32, the reported redshift
    {run}_{split}_predictions.npz   everything above plus summary statistics,
                                    the reported uncertainty, and enough
                                    provenance to match rows back to a catalog

The truth column is Z_lovoccs for train and validation and Z_desi for test;
which one was used is recorded in the npz as z_source. Rows are in file order,
so samples[i] and z_true[i] are the same galaxy.

    python3 model_infer.py --run-name baseline
    python3 model_infer.py --run-name baseline --checkpoint final --splits test
"""

import argparse
import os
import pickle
import sys

import numpy as np
import jax
from flax import serialization

### Custom code
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from training_structure import (
    make_batch,
    make_sampler,
    n_rows,
    photoz_metrics,
    preload_hdf5_to_memory,
)
from redshfit_flow_model import RedshiftFlowModel


DATA_DIR = '/projects/mccleary_group/habjan.e/SuperBIT/data/model_datasets/'
MODEL_DIR = '/projects/mccleary_group/habjan.e/SuperBIT/data/model_files/'
OUT_DIR = '/projects/mccleary_group/habjan.e/SuperBIT/data/model_datasets/'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-name', default='baseline',
                   help='run to load from MODEL_DIR, as passed to train_redshift_flow.py')
    p.add_argument('--checkpoint', choices=['best', 'final'], default='best',
                   help='best = lowest validation sigma_NMAD, final = last step')
    p.add_argument('--splits', nargs='+', default=['train', 'validation', 'test'])
    p.add_argument('--posterior-samples', type=int, default=200,
                   help='redshift draws per galaxy')
    p.add_argument('--ode-steps', type=int, default=100,
                   help='Heun steps for the flow ODE solve')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--augment-test', action='store_true',
                   help='apply random dihedral augmentation when sampling the '
                        'test split. Off by default: the DESI spec-z are the '
                        'one clean external benchmark and are left untouched.')
    p.add_argument('--augment-train-val', action='store_true',
                   help='likewise for the train and validation splits')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--model-dir', default=MODEL_DIR)
    p.add_argument('--data-dir', default=DATA_DIR)
    p.add_argument('--out-dir', default=OUT_DIR)
    return p.parse_args()


def load_config(model_dir, run_name):
    path = os.path.join(model_dir, f'{run_name}_config.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'no config at {path}; run train_redshift_flow.py --run-name {run_name} first'
        )
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_params(model_dir, run_name, which):
    """Load weights, either from the best checkpoint or the final params dump."""
    if which == 'final':
        path = os.path.join(model_dir, f'{run_name}_params.pkl')
        if not os.path.exists(path):
            raise FileNotFoundError(f'no final weights at {path}')
        with open(path, 'rb') as f:
            return pickle.load(f), path

    ### The best checkpoint is a serialized TrainState; pull the params out of
    ### it without having to rebuild the optimizer state
    path = os.path.join(model_dir, f'{run_name}_best_state.msgpack')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'no best checkpoint at {path}; pass --checkpoint final to use the '
            f'last-step weights instead'
        )
    with open(path, 'rb') as f:
        restored = serialization.msgpack_restore(f.read())
    if 'params' not in restored:
        raise KeyError(f'{path} has no "params" entry; keys are {sorted(restored)}')
    return restored['params'], path


def best_step(model_dir, run_name):
    """Step the best checkpoint was taken at, if the metadata is present."""
    path = os.path.join(model_dir, f'{run_name}_best_meta.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f).get('step')


def draw_samples(sampler, params, data, batch_size, key, include_mask,
                 augment=False, seed=0):
    """Draw the full posterior for every galaxy in a split, in file order.

    Returns (n, n_samples) float32.
    """
    n = n_rows(data)
    out = []
    rng = np.random.default_rng(seed)
    for i in range(0, n, batch_size):
        bidx = np.arange(i, min(i + batch_size, n))
        images, features, _, _ = make_batch(data, bidx, include_mask=include_mask,
                                            augment=augment, rng=rng)
        key, subkey = jax.random.split(key)
        draws = np.asarray(sampler(params, images, features, subkey))  # (S, B, 1)
        out.append(draws[:, :, 0].T.astype(np.float32))                # (B, S)
    return np.concatenate(out, axis=0)


def coverage(samples, z_true, level):
    """Fraction of galaxies whose true redshift falls in the central interval."""
    lo = np.percentile(samples, 50 * (1 - level), axis=1)
    hi = np.percentile(samples, 50 * (1 + level), axis=1)
    return float(np.mean((z_true >= lo) & (z_true <= hi)))


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    config = load_config(args.model_dir, args.run_name)
    params, params_path = load_params(args.model_dir, args.run_name, args.checkpoint)

    model = RedshiftFlowModel(**config['model_kwargs'])
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))

    print(f'run          : {args.run_name}')
    print(f'weights      : {params_path}')
    if args.checkpoint == 'best':
        print(f'best step    : {best_step(args.model_dir, args.run_name)}')
    print(f'parameters   : {n_params:,}')
    print(f'features     : {config["feature_names"]}')
    print(f'include_mask : {config["include_mask"]}  '
          f'normalized: {config["normalized"]}')
    print(f'z_mean/z_std : {model.z_mean:.4f} / {model.z_std:.4f}')
    print(f'sampling     : {args.posterior_samples} draws, '
          f'{args.ode_steps} ODE steps')

    sampler = make_sampler(
        model, n_steps=args.ode_steps, n_samples=args.posterior_samples
    )

    summary = {}
    for split in args.splits:
        data = preload_hdf5_to_memory(
            os.path.join(args.data_dir, f'{split}.hdf5'),
            feature_names=config['feature_names'],
            normalized=config['normalized'],
            include_mask=config['include_mask'],
        )

        augment = args.augment_test if split == 'test' else args.augment_train_val
        samples = draw_samples(
            sampler, params, data, args.batch_size,
            jax.random.PRNGKey(args.seed), config['include_mask'],
            augment=augment, seed=args.seed,
        )

        z_true = np.asarray(data['z']).reshape(-1)
        z_err = np.asarray(data['z_err']).reshape(-1)
        z_source = str(data['metadata']['redshift_column'])

        z_pred_mean = samples.mean(axis=1)
        z_pred_median = np.median(samples, axis=1)
        z_pred_sigma = samples.std(axis=1)
        z_p16, z_p84 = np.percentile(samples, [16, 84], axis=1)

        metrics = photoz_metrics(z_pred_mean, z_true)
        metrics['coverage_68'] = coverage(samples, z_true, 0.68)
        metrics['coverage_95'] = coverage(samples, z_true, 0.95)
        metrics['mean_posterior_sigma'] = float(z_pred_sigma.mean())
        summary[split] = metrics

        base = os.path.join(args.out_dir, f'{args.run_name}_{split}')
        np.save(f'{base}_samples.npy', samples)
        np.save(f'{base}_z_true.npy', z_true.astype(np.float32))
        np.savez(
            f'{base}_predictions.npz',
            samples=samples,
            z_true=z_true.astype(np.float32),
            z_err=z_err.astype(np.float32),
            z_source=z_source,
            z_pred_mean=z_pred_mean.astype(np.float32),
            z_pred_median=z_pred_median.astype(np.float32),
            z_pred_sigma=z_pred_sigma.astype(np.float32),
            z_pred_p16=z_p16.astype(np.float32),
            z_pred_p84=z_p84.astype(np.float32),
            ra=data['ra'],
            dec=data['dec'],
            field=data['field'],
            group_id=data['group_id'],
            split=split,
            run_name=args.run_name,
            checkpoint=args.checkpoint,
            augmented=augment,
        )

        print(f'\n{split}: {samples.shape[0]} galaxies x {samples.shape[1]} draws '
              f'against {z_source}'
              + ('  [augmented]' if augment else ''))
        print(f'  nmad {metrics["nmad"]:.4f}  '
              f'outliers {100 * metrics["outlier_frac"]:.1f}%  '
              f'bias {metrics["bias"]:+.4f}  rmse {metrics["rmse"]:.4f}')
        print(f'  posterior sigma {metrics["mean_posterior_sigma"]:.4f}  '
              f'coverage 68% {100 * metrics["coverage_68"]:.1f}  '
              f'95% {100 * metrics["coverage_95"]:.1f}')
        print(f'  wrote {base}_samples.npy, _z_true.npy, _predictions.npz')

    with open(os.path.join(args.out_dir, f'{args.run_name}_infer_metrics.pkl'), 'wb') as f:
        pickle.dump(summary, f)

    print(f'\nSaved samples for {len(args.splits)} split(s) to {args.out_dir}')


if __name__ == '__main__':
    main()
