"""Build HDF5 train/validation/test datasets for the CNN+MLP redshift model.

Each output file holds both the postage-stamp images (CNN input) and the
catalog features (MLP input), stored raw and normalized, plus the redshift
labels and enough provenance to trace any row back to its source catalog.

Training/validation labels come from LoVoCCS photo-z (``Z_lovoccs``) over four
SuperBIT cluster fields; test labels come from DESI spec-z (``Z_desi``) in
COSMOS113.

Two catalogs the earlier version of this script referenced are deliberately
absent: Abell3667 has no ``Z_lovoccs`` column and no ``VIGNET_*_im``/``_wt``
extensions, and Abell1689 has a ``Z_lovoccs`` column that is NaN in every row
(and likewise lacks the ``_im``/``_wt`` vignettes).
"""

import os

import h5py
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u


### Paths
DATA_DIR = '/projects/mccleary_group/habjan.e/SuperBIT/data/'
SAVE_DIR = '/projects/mccleary_group/habjan.e/SuperBIT/data/model_datasets/'

### Fields with LoVoCCS photo-z, used for training and validation
TRAIN_FIELDS = ['Abell2384a', 'Abell2384b', 'Abell3571', 'Abell3827']
TRAIN_Z, TRAIN_ZERR = 'Z_lovoccs', 'ZERR_lovoccs'

### Field with DESI spec-z, held out as the test set
TEST_FIELD = 'COSMOS113'
TEST_Z, TEST_ZERR = 'Z_desi', 'ZERR_desi'

### Both redshift estimates are carried in every file regardless of which one is
### the training target, so a row can always be cross-checked against the other.
### Stored as (dataset name, value column, error column); NaN where unavailable.
ALL_REDSHIFTS = [
    ('z_lovoccs', 'Z_lovoccs', 'ZERR_lovoccs'),
    ('z_desi', 'Z_desi', 'ZERR_desi'),
]

### 'z'/'z_err' hold whichever estimate is the target for that split
Z_DATASETS = ['z', 'z_err'] + [
    n + s for n, _, _ in ALL_REDSHIFTS for s in ('', '_err')
]

### CNN input: 51x51 stamps stacked into 9 channels
IMAGE_CHANNELS = [
    'VIGNET_b', 'VIGNET_g', 'VIGNET_u',
    'VIGNET_b_im', 'VIGNET_g_im', 'VIGNET_u_im',
    'VIGNET_b_wt', 'VIGNET_g_wt', 'VIGNET_u_wt',
]

### The three raw VIGNETs carry a -1e30 SExtractor mask; the others do not
MASKED_CHANNELS = ['VIGNET_b', 'VIGNET_g', 'VIGNET_u']
SENTINEL = 1e10

### MLP input
FEATURE_NAMES = [
    'm_b', 'm_g', 'm_u',
    'R_b', 'R_g', 'R_u',
    'R_b_prepsf', 'R_g_prepsf', 'R_u_prepsf',
    'FLUX_AUTO_b', 'FLUX_AUTO_g', 'FLUX_AUTO_u',
    'color_bg', 'color_ub',
]

### Fluxes span ~5 decades, so standardize them in log space
LOG_FEATURES = ['FLUX_AUTO_b', 'FLUX_AUTO_g', 'FLUX_AUTO_u']
FLUX_FLOOR = 1e-3

### Objects seen in more than one pointing must not straddle the train/val split
MATCH_RADIUS = 1.0 * u.arcsec

### Approximate number of validation rows; whole object groups are moved, so
### the realized count lands slightly above this
N_VALIDATION = 1000
SEED = 42

### Test-set cuts: DESI targets that are not galaxies, or stamps off the edge
DROP_SPECTYPE = {'STAR', 'QSO'}

### Rows per read/write chunk (256 x 9 x 51 x 51 x 4 B ~ 24 MB)
CHUNK = 256
STAMP = 51


def catalog_path(field):
    return os.path.join(DATA_DIR, field + '_colors_mags.fits')


def read_catalog(field, zcol, zerrcol, extra_cols=()):
    """Return the non-image columns for rows with a usable redshift.

    Reads only scalar columns so the 400 MB of vignettes stay on disk.
    """
    with fits.open(catalog_path(field), memmap=True) as hdul:
        rec = hdul[1].data
        present = set(rec.columns.names)

        missing = [c for c in [zcol, zerrcol] + FEATURE_NAMES + IMAGE_CHANNELS
                   if c not in present]
        if missing:
            raise KeyError(f'{field} is missing required columns: {missing}')

        z = np.asarray(rec[zcol], dtype=np.float64)
        good = np.isfinite(z) & (z > 0)
        rows = np.flatnonzero(good)

        cat = {
            'rows': rows,
            'z': z[rows].astype(np.float32),
            'z_err': np.asarray(rec[zerrcol], dtype=np.float64)[rows].astype(np.float32),
            'ra': np.asarray(rec['ra'], dtype=np.float64)[rows],
            'dec': np.asarray(rec['dec'], dtype=np.float64)[rows],
            'features': np.column_stack(
                [np.asarray(rec[c], dtype=np.float64)[rows] for c in FEATURE_NAMES]
            ).astype(np.float32),
        }
        ### Keep both redshift estimates alongside the training target
        for name, vcol, ecol in ALL_REDSHIFTS:
            for suffix, col in (('', vcol), ('_err', ecol)):
                if col in present:
                    vals = np.asarray(rec[col], dtype=np.float64)[rows]
                else:
                    vals = np.full(len(rows), np.nan)
                cat[name + suffix] = vals.astype(np.float32)

        for c in extra_cols:
            cat[c] = np.asarray(rec[c])[rows]
    return cat


def group_repeated_objects(ra, dec):
    """Label sources within MATCH_RADIUS of each other with a shared group id.

    Abell2384a and Abell2384b are overlapping pointings of one cluster, so
    ~1100 objects appear twice. Splitting those rows independently would leak
    near-identical images (with identical labels) into the validation set.
    """
    coords = SkyCoord(ra, dec, unit='deg')
    left, right, _, _ = coords.search_around_sky(coords, MATCH_RADIUS)

    parent = np.arange(len(ra))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in zip(left, right):
        if a >= b:
            continue
        ra_, rb_ = find(a), find(b)
        if ra_ != rb_:
            parent[ra_] = rb_

    roots = np.array([find(i) for i in range(len(ra))])
    _, group_id = np.unique(roots, return_inverse=True)
    return group_id.astype(np.int32)


def split_by_group(group_id, n_target, seed):
    """Assign whole object groups to validation until n_target rows are reached."""
    rng = np.random.default_rng(seed)
    groups = np.unique(group_id)
    rng.shuffle(groups)

    sizes = np.bincount(group_id)
    taken, val_groups = 0, []
    for g in groups:
        if taken >= n_target:
            break
        val_groups.append(g)
        taken += sizes[g]

    is_val = np.isin(group_id, val_groups)
    return ~is_val, is_val


def create_file(path, n_rows):
    """Open an HDF5 file with the full schema pre-allocated for n_rows."""
    f = h5py.File(path, 'w')
    opts = dict(compression='gzip', compression_opts=4, shuffle=True)

    img_shape = (n_rows, STAMP, STAMP, len(IMAGE_CHANNELS))
    rows_per_chunk = max(1, min(32, n_rows))
    img_chunk = (rows_per_chunk, STAMP, STAMP, len(IMAGE_CHANNELS))
    f.create_dataset('images', img_shape, dtype='f4', chunks=img_chunk, **opts)
    f.create_dataset('images_norm', img_shape, dtype='f4', chunks=img_chunk, **opts)
    f.create_dataset(
        'image_mask',
        (n_rows, STAMP, STAMP, len(MASKED_CHANNELS)),
        dtype='bool',
        chunks=(rows_per_chunk, STAMP, STAMP, len(MASKED_CHANNELS)),
        **opts,
    )
    f.create_dataset('features', (n_rows, len(FEATURE_NAMES)), dtype='f4')
    f.create_dataset('features_norm', (n_rows, len(FEATURE_NAMES)), dtype='f4')
    for name in Z_DATASETS:
        f.create_dataset(name, (n_rows,), dtype='f4')
    for name in ('ra', 'dec'):
        f.create_dataset(name, (n_rows,), dtype='f8')
    f.create_dataset('field', (n_rows,), dtype=h5py.string_dtype())
    f.create_dataset('group_id', (n_rows,), dtype='i4')
    return f


def write_images(dest, field, source_rows, dest_start):
    """Stream vignettes from one FITS file into an open HDF5 file."""
    images = dest['images']
    masks = dest['image_mask']
    mask_idx = [IMAGE_CHANNELS.index(c) for c in MASKED_CHANNELS]

    with fits.open(catalog_path(field), memmap=True) as hdul:
        rec = hdul[1].data
        for lo in range(0, len(source_rows), CHUNK):
            rows = source_rows[lo:lo + CHUNK]
            block = np.stack(
                [np.asarray(rec[c][rows], dtype=np.float32) for c in IMAGE_CHANNELS],
                axis=-1,
            )

            ### -1e30 marks pixels SExtractor could not measure
            bad = ~np.isfinite(block) | (np.abs(block) > SENTINEL)
            block[bad] = 0.0

            a, b = dest_start + lo, dest_start + lo + len(rows)
            images[a:b] = block
            masks[a:b] = ~bad[..., mask_idx]


def channel_stats(path, n_sample, seed):
    """Robust per-channel center and scale from a random subset of train rows.

    Median/MAD rather than mean/std: the stamps have a long bright tail (maxima
    ~1e4 against a median near 0.5) that would otherwise set the scale.
    """
    with h5py.File(path, 'r') as f:
        n = f['images'].shape[0]
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=min(n_sample, n), replace=False))

        pixels = []
        for lo in range(0, len(idx), CHUNK):
            pixels.append(f['images'][idx[lo:lo + CHUNK]])
        pixels = np.concatenate(pixels).reshape(-1, len(IMAGE_CHANNELS))

    median = np.median(pixels, axis=0)
    scale = 1.4826 * np.median(np.abs(pixels - median), axis=0)
    ### Weight maps are near-constant, so guard against a degenerate scale
    scale[scale <= 0] = np.std(pixels, axis=0)[scale <= 0]
    scale[scale <= 0] = 1.0
    return median.astype(np.float32), scale.astype(np.float32)


def transform_features(features):
    """Apply the log10 transform to the flux columns, leaving the rest alone."""
    out = features.astype(np.float64).copy()
    for name in LOG_FEATURES:
        j = FEATURE_NAMES.index(name)
        out[:, j] = np.log10(np.maximum(out[:, j], FLUX_FLOOR))
    return out


def apply_normalization(path, img_median, img_scale, feat_mean, feat_std):
    """Fill images_norm / features_norm and record the stats as file attributes."""
    with h5py.File(path, 'r+') as f:
        n = f['images'].shape[0]
        for lo in range(0, n, CHUNK):
            hi = min(lo + CHUNK, n)
            f['images_norm'][lo:hi] = (f['images'][lo:hi] - img_median) / img_scale

        feats = transform_features(f['features'][:])
        f['features_norm'][:] = ((feats - feat_mean) / feat_std).astype(np.float32)

        f.attrs['image_channels'] = IMAGE_CHANNELS
        f.attrs['image_mask_channels'] = MASKED_CHANNELS
        f.attrs['image_median'] = img_median
        f.attrs['image_scale'] = img_scale
        f.attrs['feature_names'] = FEATURE_NAMES
        f.attrs['feature_mean'] = feat_mean.astype(np.float32)
        f.attrs['feature_std'] = feat_std.astype(np.float32)
        f.attrs['feature_log10'] = np.array(
            [name in LOG_FEATURES for name in FEATURE_NAMES]
        )
        f.attrs['flux_floor'] = FLUX_FLOOR
        f.attrs['stamp_size'] = STAMP


def build_train_validation():
    """Write train.hdf5 and validation.hdf5 from the four LoVoCCS fields."""
    cats = {f: read_catalog(f, TRAIN_Z, TRAIN_ZERR) for f in TRAIN_FIELDS}
    for f in TRAIN_FIELDS:
        print(f'  {f}: {len(cats[f]["rows"])} rows with {TRAIN_Z}')

    ra = np.concatenate([cats[f]['ra'] for f in TRAIN_FIELDS])
    dec = np.concatenate([cats[f]['dec'] for f in TRAIN_FIELDS])
    field_of_row = np.concatenate(
        [np.full(len(cats[f]['rows']), f) for f in TRAIN_FIELDS]
    )

    group_id = group_repeated_objects(ra, dec)
    n_dup = len(group_id) - len(np.unique(group_id))
    print(f'  {len(group_id)} rows -> {len(np.unique(group_id))} unique objects '
          f'({n_dup} repeat observations)')

    is_train, is_val = split_by_group(group_id, N_VALIDATION, SEED)
    print(f'  group-aware split: {is_train.sum()} train / {is_val.sum()} validation')

    paths = {}
    for split, keep in (('train', is_train), ('validation', is_val)):
        path = os.path.join(SAVE_DIR, split + '.hdf5')
        paths[split] = path
        dest = create_file(path, int(keep.sum()))

        kept_group_id = group_id[keep]
        cursor = 0
        offset = 0
        for f in TRAIN_FIELDS:
            n_f = len(cats[f]['rows'])
            sel = keep[offset:offset + n_f]
            offset += n_f
            if not sel.any():
                continue

            local = np.flatnonzero(sel)
            n_sel = len(local)
            a, b = cursor, cursor + n_sel

            dest['features'][a:b] = cats[f]['features'][local]
            for name in Z_DATASETS:
                dest[name][a:b] = cats[f][name][local]
            dest['ra'][a:b] = cats[f]['ra'][local]
            dest['dec'][a:b] = cats[f]['dec'][local]
            dest['field'][a:b] = [f] * n_sel
            dest['group_id'][a:b] = kept_group_id[a:b]

            write_images(dest, f, cats[f]['rows'][local], a)
            cursor = b

        dest.attrs['split'] = split
        dest.attrs['redshift_column'] = TRAIN_Z
        dest.attrs['source_fields'] = TRAIN_FIELDS
        dest.close()
        print(f'  wrote {path} ({keep.sum()} rows)')

    ### Sanity check that field labels line up with the row ordering
    with h5py.File(paths['train'], 'r') as f:
        assert set(np.unique(f['field'][:].astype(str))) <= set(TRAIN_FIELDS)

    return paths['train'], paths['validation']


def build_test():
    """Write test.hdf5 from COSMOS113, keeping only clean galaxy targets."""
    extra = ['SPECTYPE', 'at_edge', 'ZWARN']
    cat = read_catalog(TEST_FIELD, TEST_Z, TEST_ZERR, extra_cols=extra)
    n_all = len(cat['rows'])

    spectype = np.char.strip(cat['SPECTYPE'].astype(str))
    at_edge = cat['at_edge'].astype(bool)
    keep = ~np.isin(spectype, list(DROP_SPECTYPE)) & ~at_edge

    print(f'  {TEST_FIELD}: {n_all} rows with {TEST_Z}')
    for label in sorted(DROP_SPECTYPE):
        print(f'    dropping {int((spectype == label).sum())} {label}')
    print(f'    dropping {int(at_edge.sum())} at_edge')
    print(f'    keeping {int(keep.sum())}')

    local = np.flatnonzero(keep)
    n = len(local)

    path = os.path.join(SAVE_DIR, 'test.hdf5')
    dest = create_file(path, n)
    dest['features'][:] = cat['features'][local]
    for name in Z_DATASETS:
        dest[name][:] = cat[name][local]
    dest['ra'][:] = cat['ra'][local]
    dest['dec'][:] = cat['dec'][local]
    dest['field'][:] = [TEST_FIELD] * n
    ### No overlapping pointings here, so every row is its own group
    dest['group_id'][:] = np.arange(n, dtype=np.int32)

    dest.create_dataset(
        'spectype', data=spectype[local].tolist(), dtype=h5py.string_dtype()
    )
    dest.create_dataset('at_edge', data=at_edge[local])
    dest.create_dataset('zwarn', data=cat['ZWARN'][local].astype(np.int64))

    write_images(dest, TEST_FIELD, cat['rows'][local], 0)

    dest.attrs['split'] = 'test'
    dest.attrs['redshift_column'] = TEST_Z
    dest.attrs['source_fields'] = [TEST_FIELD]
    dest.attrs['dropped_spectype'] = sorted(DROP_SPECTYPE)
    dest.close()
    print(f'  wrote {path} ({n} rows)')
    return path


def summarize(path):
    with h5py.File(path, 'r') as f:
        z = f['z'][:]
        img = f['images_norm']
        sample = img[:min(256, img.shape[0])]
        print(f'\n{os.path.basename(path)}: {len(z)} rows, {f.attrs["redshift_column"]}')
        print(f'  z      min={z.min():.4f} med={np.median(z):.4f} max={z.max():.4f}')
        for name, _, _ in ALL_REDSHIFTS:
            v = f[name][:]
            ok = np.isfinite(v) & (v > 0)
            if ok.any():
                print(f'  {name:<10} {ok.sum():>5}/{len(v)} rows  '
                      f'min={v[ok].min():.4f} med={np.median(v[ok]):.4f} max={v[ok].max():.4f}')
            else:
                print(f'  {name:<10} {ok.sum():>5}/{len(v)} rows')
        print(f'  images {img.shape}  norm med={np.median(sample):+.4f} '
              f'p99={np.percentile(sample, 99):.3f}')
        fn = f['features_norm'][:]
        print(f'  feats  {fn.shape}  |mean|max={np.abs(fn.mean(axis=0)).max():.4f} '
              f'std range=[{fn.std(axis=0).min():.3f}, {fn.std(axis=0).max():.3f}]')
        print(f'  masked pixels: {100 * (1 - f["image_mask"][:256].mean()):.2f}%')


if __name__ == '__main__':
    os.makedirs(SAVE_DIR, exist_ok=True)

    print('Building train/validation from LoVoCCS fields...')
    train_path, val_path = build_train_validation()

    print('\nBuilding test from DESI spec-z...')
    test_path = build_test()

    ### Normalization is fit on the training split alone, then applied everywhere
    print('\nFitting normalization on the training split...')
    img_median, img_scale = channel_stats(train_path, n_sample=1000, seed=SEED)
    with h5py.File(train_path, 'r') as f:
        train_feats = transform_features(f['features'][:])
    feat_mean = train_feats.mean(axis=0)
    feat_std = train_feats.std(axis=0)
    feat_std[feat_std <= 0] = 1.0

    for name, med, sc in zip(IMAGE_CHANNELS, img_median, img_scale):
        print(f'  {name:<16} median={med:>10.4g}  scale={sc:>10.4g}')

    for path in (train_path, val_path, test_path):
        apply_normalization(path, img_median, img_scale, feat_mean, feat_std)

    for path in (train_path, val_path, test_path):
        summarize(path)

    print(f'\nSuccessfully saved data to {SAVE_DIR}')
