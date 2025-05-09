import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from sklearn.model_selection import train_test_split
from functools import partial
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf

import numpy as np 
import matplotlib.pyplot as plt 
from astropy.io import fits
from astropy.table import Table
from astropy.table import vstack
import os

### Import SuperBIT table

fits_file_2384a = '/work/mccleary_group/saha/data/Abell2384a/sextractor_dualmode/out/Abell2384a_colors_mags.fits'
hdul_2384a = fits.open(fits_file_2384a)
data_2384a = Table(hdul_2384a[1].data)

fits_file_2384b = '/work/mccleary_group/saha/data/Abell2384b/sextractor_dualmode/out/Abell2384b_colors_mags.fits'
hdul_2384b = fits.open(fits_file_2384b)
data_2384b = Table(hdul_2384b[1].data)

fits_file_3667 = '/work/mccleary_group/saha/data/Abell3667/sextractor_dualmode/out/Abell3667_colors_mags.fits'
hdul_3667 = fits.open(fits_file_3667)
data_3667 = Table(hdul_3667[1].data)

fits_file_3571 = '/work/mccleary_group/saha/data/Abell3571/sextractor_dualmode/out/Abell3571_colors_mags.fits'
hdul_3571 = fits.open(fits_file_3571)
data_3571 = Table(hdul_3571[1].data)

fits_file_3827 = '/work/mccleary_group/saha/data/Abell3827/sextractor_dualmode/out/Abell3827_colors_mags.fits'
hdul_3827 = fits.open(fits_file_3827)
data_3827 = Table(hdul_3827[1].data)

data = vstack([data_2384a, data_2384b, data_3667, data_3571, data_3827])


### Object with LoVoCCS BPZ redshifts
redshift_col = 'Z_best'
data_z = data[~np.isnan(np.array(data[redshift_col])) & (np.array(data[redshift_col]) > 0)]

### Random subset for test data
arr = np.arange(len(data_z))
subset = np.random.choice(arr, size=200, replace=False)

suffix = ''

### Vignette data in different bands
vig_u = np.array(data_z['VIGNET_u' + suffix])
vig_b = np.array(data_z['VIGNET_b' + suffix])
vig_g = np.array(data_z['VIGNET_g' + suffix])

### Make training and test datasets for the CNN
vig_cnn = np.array([vig_u, vig_b, vig_g])
vig_cnn[abs(vig_cnn) > 10**10] = 0
vig_cnn = np.transpose(vig_cnn, (1, 2, 3, 0))

### Format data for Jax

trainx = vig_cnn[~np.isin(arr, subset), :, :, :]
trainy = np.array([data_z[redshift_col][~np.isin(arr, subset)]]).T

testx = vig_cnn[np.isin(arr, subset), :, :, :]
testy = np.array([data_z[redshift_col][subset]]).T

### Save training and test numpy arrays
save_data = os.getcwd().replace('ml_redshifts/Jax_models', 'data') + '/cnn_data/'
np.save(save_data + 'trainx_cnn_jax' + suffix + '.npy', trainx)
np.save(save_data + 'trainy_cnn_jax' + suffix + '.npy', trainy)
np.save(save_data + 'testx_cnn_jax' + suffix + '.npy', testx)
np.save(save_data + 'testy_cnn_jax' + suffix + '.npy', testy)

def create_dataloader(x, y, batch_size, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(x))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = create_dataloader(trainx, trainy, batch_size=16, shuffle=True)
test_ds = create_dataloader(testx, testy, batch_size=16, shuffle=False)

### Define model

import sys
sys.path.append(os.getcwd() + '/Jax_models')
from jax_models import CNNModel
model = CNNModel()

### Training function

def create_train_state(rng, model, learning_rate, input_shape):
    params = model.init(rng, jnp.ones(input_shape))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

### Training loop

@jax.jit
def train_step(state, batch, rng):
    def loss_fn(params):
        x, y = batch
        pred = state.apply_fn({'params': params}, x, rngs={'dropout': rng}, training=True)
        return jnp.mean((pred.squeeze() - y) ** 2)

    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)

def train_model(train_ds, test_ds, model, epochs=50, batch_size=16):
    rng = jax.random.PRNGKey(0)
    input_shape = (batch_size,) + trainx.shape[1:]
    state = create_train_state(rng, model, 1e-3, input_shape)

    for epoch in range(epochs):
        for x_batch, y_batch in tfds.as_numpy(train_ds):
            rng, subrng = jax.random.split(rng)
            state = train_step(state, (x_batch, y_batch), subrng)
        print(f"Epoch {epoch+1} complete")
    return state

### Train model

if __name__ == "__main__":
    model = CNNModel()
    trained_state = train_model(train_ds, test_ds, model)

# Save model parameters
import pickle
save_path = os.getcwd() + '/cnn_model_params' + suffix + '.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(trained_state.params, f)