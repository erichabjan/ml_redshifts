"""Train the CNN+MLP flow-matching redshift model.

Loads the HDF5 splits written by data_process.py, builds the model from
redshfit_flow_model.py, and runs the training loop from training_structure.py.
Weights, loss curves and metrics all land in SAVE_DIR.

The DESI test split is loaded for monitoring only: its loss and metrics are
logged to wandb and saved, but it never touches the gradient, the early-stopping
decision or the choice of best checkpoint, and its stamps are never augmented.
Pass --no-test-monitor to keep it entirely out of the run.

    python3 train_redshift_flow.py --run-name baseline
    python3 train_redshift_flow.py --run-name no_mask --no-mask
    python3 train_redshift_flow.py --run-name catalog_only --no-images
"""

import argparse
import os
import pickle
import sys

import numpy as np

### Custom code
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from training_structure import (
    CENSOR_SIGMA_FLOOR,
    CENSOR_Z_HI,
    CENSOR_Z_LO,
    MLP_FEATURES,
    TARGET_Z_HI,
    TARGET_Z_LO,
    WANDB_DIR,
    preload_hdf5_to_memory,
    train_model,
)
from redshfit_flow_model import RedshiftFlowModel


DATA_DIR = '/projects/mccleary_group/habjan.e/SuperBIT/data/model_datasets/'
SAVE_DIR = '/projects/mccleary_group/habjan.e/SuperBIT/data/model_files/'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument('--run-name', default='redshift_flow',
                   help='prefix for every file written to SAVE_DIR')

    ### Inputs
    p.add_argument('--features', nargs='+', default=list(MLP_FEATURES),
                   help='catalog columns for the MLP branch')
    p.add_argument('--no-mask', action='store_true',
                   help='do not append the 3 pixel-mask channels to the images')
    p.add_argument('--no-images', action='store_true',
                   help='ablation: catalog features only, no CNN branch')
    p.add_argument('--no-features', action='store_true',
                   help='ablation: images only, no MLP branch')
    p.add_argument('--raw-inputs', action='store_true',
                   help='load unnormalized images/features instead of the '
                        'pre-normalized ones')
    p.add_argument('--no-augment', action='store_true',
                   help='disable random dihedral augmentation of the stamps')
    p.add_argument('--no-augment-val', action='store_true',
                   help='disable augmentation for the validation split only')
    p.add_argument('--no-test-monitor', action='store_true',
                   help='do not load the DESI test split at all; by default it '
                        'is loaded and its loss logged, but never used for '
                        'gradients, early stopping or checkpoint selection')

    ### Architecture
    p.add_argument('--cnn-features', nargs='+', type=int, default=[24, 48, 96])
    p.add_argument('--mlp-features', nargs='+', type=int, default=[64, 64])
    p.add_argument('--embed-dim', type=int, default=128)
    p.add_argument('--velocity-width', type=int, default=112)
    p.add_argument('--velocity-depth', type=int, default=3)
    p.add_argument('--dropout-rate', type=float, default=0.15,
                   help='dropout in the encoders')
    p.add_argument('--spatial-dropout-rate', type=float, default=0.1,
                   help='channel dropout inside the conv stack')
    p.add_argument('--head-dropout-rate', type=float, default=0.1,
                   help='dropout in the velocity head')
    p.add_argument('--time-embed-dim', type=int, default=64)

    ### Optimization
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-train-steps', type=int, default=20_000)
    p.add_argument('--learning-rate', type=float, default=1e-3)
    p.add_argument('--warmup-steps', type=int, default=500)
    p.add_argument('--grad-clipping', type=float, default=1.0)
    p.add_argument('--target-z-lo', type=float, default=TARGET_Z_LO,
                   help='lower bound on the smeared training target')
    p.add_argument('--target-z-hi', type=float, default=TARGET_Z_HI,
                   help='upper bound on the smeared training target; kept above '
                        'the 1.49 label ceiling so those objects can be smeared '
                        'symmetrically instead of only downward')
    p.add_argument('--no-censor', action='store_true',
                   help='disable censoring of labels sitting on the BPZ grid '
                        'edges; they are then treated as ordinary measurements')
    p.add_argument('--censor-z-lo', type=float, default=CENSOR_Z_LO,
                   help='BPZ grid floor; labels here mean "at most this"')
    p.add_argument('--censor-z-hi', type=float, default=CENSOR_Z_HI,
                   help='BPZ grid ceiling; labels here mean "at least this"')
    p.add_argument('--censor-sigma-floor', type=float, default=CENSOR_SIGMA_FLOOR,
                   help='minimum smearing width for objects whose error bar '
                        'crosses a grid edge')
    p.add_argument('--seed', type=int, default=42)

    ### Evaluation
    p.add_argument('--eval-every', type=int, default=250)
    p.add_argument('--sample-every', type=int, default=500)
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--ode-steps', type=int, default=50)
    p.add_argument('--posterior-samples', type=int, default=32)
    p.add_argument('--patience', type=int, default=10,
                   help='sampled evals without a sigma_NMAD improvement before '
                        'stopping; 0 disables early stopping')

    ### Bookkeeping
    p.add_argument('--wandb-project', default='superbit-redshift-flow')
    p.add_argument('--wandb-mode', default='online',
                   choices=['online', 'offline', 'disabled'])
    p.add_argument('--wandb-dir', default=WANDB_DIR,
                   help='parent of the wandb/ run folder')
    p.add_argument('--notes', default='')
    p.add_argument('--max-runtime-hours', type=float, default=7.8)

    args = p.parse_args()
    if args.no_images and args.no_features:
        p.error('--no-images and --no-features cannot both be set')
    return args


def main():
    args = parse_args()
    os.makedirs(SAVE_DIR, exist_ok=True)

    include_mask = not args.no_mask
    normalized = not args.raw_inputs

    train_data = preload_hdf5_to_memory(
        os.path.join(DATA_DIR, 'train.hdf5'),
        feature_names=args.features,
        normalized=normalized,
        include_mask=include_mask,
    )
    val_data = preload_hdf5_to_memory(
        os.path.join(DATA_DIR, 'validation.hdf5'),
        feature_names=args.features,
        normalized=normalized,
        include_mask=include_mask,
    )
    ### Monitoring only -- see the module docstring
    test_data = None
    if not args.no_test_monitor:
        test_data = preload_hdf5_to_memory(
            os.path.join(DATA_DIR, 'test.hdf5'),
            feature_names=args.features,
            normalized=normalized,
            include_mask=include_mask,
        )

    splits = [d for d in (train_data, val_data, test_data) if d is not None]

    ### Ablations: dropping a branch's array makes make_batch pass None for it
    if args.no_images:
        for d in splits:
            d['images'] = None
            d['image_mask'] = None
    if args.no_features:
        for d in splits:
            d['features'] = None

    ### The flow runs on (z - z_mean) / z_std, always using the training split
    model_kwargs = dict(
        cnn_features=tuple(args.cnn_features),
        mlp_features=tuple(args.mlp_features),
        embed_dim=args.embed_dim,
        velocity_width=args.velocity_width,
        velocity_depth=args.velocity_depth,
        time_embed_dim=args.time_embed_dim,
        dropout_rate=args.dropout_rate,
        spatial_dropout_rate=args.spatial_dropout_rate,
        head_dropout_rate=args.head_dropout_rate,
        z_mean=train_data['z_mean'],
        z_std=train_data['z_std'],
    )
    model = RedshiftFlowModel(**model_kwargs)

    (state, train_loss, val_loss, test_loss,
     metrics_history, test_metrics_history) = train_model(
        train_data,
        val_data,
        model,
        test_data=test_data,
        batch_size=args.batch_size,
        num_train_steps=args.num_train_steps,
        eval_every=args.eval_every,
        sample_every=args.sample_every,
        log_every=args.log_every,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        grad_clipping=args.grad_clipping,
        include_mask=include_mask,
        augment=not args.no_augment,
        augment_val=not (args.no_augment or args.no_augment_val),
        target_z_lo=args.target_z_lo,
        target_z_hi=args.target_z_hi,
        censor=not args.no_censor,
        censor_z_lo=args.censor_z_lo,
        censor_z_hi=args.censor_z_hi,
        censor_sigma_floor=args.censor_sigma_floor,
        n_ode_steps=args.ode_steps,
        n_posterior_samples=args.posterior_samples,
        early_stopping_patience=args.patience if args.patience > 0 else None,
        seed=args.seed,
        wandb_project=args.wandb_project,
        wandb_mode=args.wandb_mode,
        wandb_dir=args.wandb_dir,
        wandb_notes=args.notes,
        cfg_dict=vars(args),
        checkpoint_dir=SAVE_DIR,
        checkpoint_prefix=args.run_name,
        max_runtime_hours=args.max_runtime_hours,
    )

    ### Final weights, plus everything needed to rebuild the model for inference
    params_path = os.path.join(SAVE_DIR, f'{args.run_name}_params.pkl')
    with open(params_path, 'wb') as f:
        pickle.dump(state.params, f)

    config_path = os.path.join(SAVE_DIR, f'{args.run_name}_config.pkl')
    with open(config_path, 'wb') as f:
        pickle.dump(dict(
            model_kwargs=model_kwargs,
            feature_names=train_data['feature_names'],
            image_channels=train_data['image_channels'],
            mask_channels=train_data['mask_channels'],
            include_mask=include_mask,
            normalized=normalized,
            args=vars(args),
        ), f)

    np.save(os.path.join(SAVE_DIR, f'{args.run_name}_train_loss.npy'), train_loss)
    np.save(os.path.join(SAVE_DIR, f'{args.run_name}_val_loss.npy'), val_loss)
    np.save(os.path.join(SAVE_DIR, f'{args.run_name}_test_loss.npy'), test_loss)
    with open(os.path.join(SAVE_DIR, f'{args.run_name}_metrics.pkl'), 'wb') as f:
        pickle.dump(metrics_history, f)
    with open(os.path.join(SAVE_DIR, f'{args.run_name}_test_metrics.pkl'), 'wb') as f:
        pickle.dump(test_metrics_history, f)

    if metrics_history:
        best = min(metrics_history, key=lambda m: m['nmad'])
        print(f"\nBest validation: sigma_NMAD {best['nmad']:.4f}, "
              f"outliers {100 * best['outlier_frac']:.1f}%, "
              f"bias {best['bias']:+.4f}, rmse {best['rmse']:.4f} "
              f"at step {best['step']}")
    if test_metrics_history:
        t = min(test_metrics_history, key=lambda m: abs(m['step'] - best['step']))
        print(f"Test at that step:  sigma_NMAD {t['nmad']:.4f}, "
              f"outliers {100 * t['outlier_frac']:.1f}%, "
              f"bias {t['bias']:+.4f}, rmse {t['rmse']:.4f}")

    print(f'\nSaved weights and results to {SAVE_DIR} under "{args.run_name}"')


if __name__ == '__main__':
    main()
