import numpy as np 
import matplotlib.pyplot as plt 
from astropy.io import fits
from astropy.table import Table
from astropy.table import vstack
import os

### ML pacakges
from scipy import stats
import tensorflow as tf
import tensorflow_datasets as tfds
import keras_tuner as kt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from tensorflow.keras import layers, models


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

suffix = '_im'

### Vignette data in different bands
vig_u = np.array(data_z['VIGNET_u' + suffix])
vig_b = np.array(data_z['VIGNET_b' + suffix])
vig_g = np.array(data_z['VIGNET_g' + suffix])

### Make training and test datasets for the CNN
vig_cnn = np.array([vig_u, vig_b, vig_g])
vig_cnn[abs(vig_cnn) > 10**10] = 0
vig_cnn = np.transpose(vig_cnn, (1, 2, 3, 0))

trainx = vig_cnn[~np.isin(arr, subset), :, :, :]
trainy = np.array([data_z[redshift_col][~np.isin(arr, subset)]]).T

testx = vig_cnn[np.isin(arr, subset), :, :, :]
testy = np.array([data_z[redshift_col][subset]]).T

### Save training and test numpy arrays
save_data = os.getcwd().replace('ml_redshifts', 'data') + '/cnn_data/'
np.save(save_data + 'trainx_cnn' + suffix + '.npy', trainx)
np.save(save_data + 'trainy_cnn' + suffix + '.npy', trainy)
np.save(save_data + 'testx_cnn' + suffix + '.npy', testx)
np.save(save_data + 'testy_cnn' + suffix + '.npy', testy)

### Make tensorflow objects
tf_train = tf.data.Dataset.from_tensor_slices((trainx, trainy)).cache()
tf_test = tf.data.Dataset.from_tensor_slices((testx, testy)).cache()

tf_train = tf_train.shuffle(len(tf_train))

tf_train = tf_train.shuffle(500).batch(16)
tf_test = tf_test.batch(16)

tf_train = tf_train.prefetch(tf.data.AUTOTUNE)
tf_test = tf_test.prefetch(tf.data.AUTOTUNE)

### CNN code
cnn_model = models.Sequential([
    tf.keras.Input(shape=trainx.shape[1:]),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.25),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.25),
    layers.Dense(1)
])

cnn_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

### Train model
cnn_model.fit(tf_train, validation_data = tf_test, epochs=50, batch_size=64, validation_split=0.1, verbose=True)

### Save model
path = os.getcwd() + '/cnn_redshift' + suffix + '.keras'
cnn_model.save(path)