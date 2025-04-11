import numpy as np 
import matplotlib.pyplot as plt 
import os
from astropy.io import fits
from astropy.table import Table
from astropy.table import vstack

### ML pacakges
from scipy import stats
import tensorflow as tf
import tensorflow_datasets as tfds
import keras_tuner as kt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns


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


### Make training and test arrays

data_z = data[~np.isnan(np.array(data['redshift']))]

arr = np.arange(len(data_z))
subset = np.random.choice(arr, size=200, replace=False)
#x_cols = [col for col in data_z.colnames if (col == 'redshift') & (col != 'ra') & (col != 'dec') & (col != 'VIGNET_b') & (col != 'VIGNET_g') & (col != 'VIGNET_u')]
x_cols = ['m_b', 'm_g', 'm_u', 'R_b', 'R_g',  'R_u']

trainx = np.array([data_z[x_cols[i]][~np.isin(arr, subset)] for i in range(len(x_cols))]).T
trainy = np.array([data_z['redshift'][~np.isin(arr, subset)]]).T

testx = np.array([data_z[x_cols[i]][subset] for i in range(len(x_cols))]).T
testy = np.array([data_z['redshift'][subset]]).T

### Convert data to tensorflow objects

tf_train = tf.data.Dataset.from_tensor_slices((trainx, trainy)).cache()
tf_test = tf.data.Dataset.from_tensor_slices((testx, testy)).cache()

tf_train = tf_train.shuffle(len(tf_train))

tf_train = tf_train.shuffle(500).batch(16)
tf_test = tf_test.batch(16)

tf_train = tf_train.prefetch(tf.data.AUTOTUNE)
tf_test = tf_test.prefetch(tf.data.AUTOTUNE)


### Model code


def model_func(hp):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.Input(shape=(trainx.shape[1],)))
    
    #model.add(tf.keras.layers.Dense(units=hp.Int(f"neurons", min_value=100, max_value=600), activation='relu')) #, kernel_regularizer='l1_l2'))

    for i in range(hp.Int(f"layers", min_value=1, max_value=5)):
        model.add(tf.keras.layers.Dense(units=hp.Int(f"neurons_{i}", min_value=10, max_value=600), activation='relu', kernel_regularizer='l2'))

        drop = hp.Float(f"dropout_{i}", min_value=0.01, max_value=0.3)
        model.add(tf.keras.layers.Dropout(rate=drop))
    
    model.add(tf.keras.layers.Dense(1, activation = None))

    #lr = hp.Float(f'learning rate', min_value=10**-4, max_value=10**-2)

    model.compile(optimizer = tf.keras.optimizers.Adam(), #(learning_rate=lr), 
              loss = tf.keras.losses.MeanSquaredError(), 
              metrics = [tf.keras.metrics.MeanAbsoluteError()],)
    
    return model


### Optimization code with Hyperband

hyperband_dirc = os.getcwd() + '/mlp_redshift_dirc'

tuner = kt.Hyperband(
    model_func,
    objective=kt.Objective('val_mean_absolute_error', direction='min'),  # for regression
    factor=10,
    directory=hyperband_dirc,
    project_name='intro_to_kt'
)

callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',     # or 'val_mean_absolute_error'
    min_delta=0.001,
    patience=5,
    restore_best_weights=True  # optional but recommended
)

### Run optimization

tuner.search(tf_train, epochs=50, validation_data=tf_test, callbacks=[callback], verbose = 1)
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

### Define a check point callback

path = os.getcwd() + '/mlp_redshift.keras'

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    path, 
    monitor='val_loss',
    save_best_only=True
)

### Train the best model

model = tuner.hypermodel.build(best_hps)
model.fit(tf_train, epochs=50, callbacks=[checkpoint], validation_data=tf_test, verbose = 1)