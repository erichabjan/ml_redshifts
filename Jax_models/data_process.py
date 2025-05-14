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
redshift_col = 'Z_lovoccs'
redshift_col_err = 'ZERR_lovoccs'
data_z = data[~np.isnan(np.array(data[redshift_col])) & (np.array(data[redshift_col]) > 0)]

### Size of test dataset
test_size = 1000

### Random subset for test data
arr = np.arange(len(data_z))
subset = np.random.choice(arr, size=test_size, replace=False)

### Vignette data in different bands
vig_u = np.array(data_z['VIGNET_u_im'])
vig_b = np.array(data_z['VIGNET_b_im'])
vig_g = np.array(data_z['VIGNET_g_im'])

### Make training and test datasets for the CNN
vig_cnn = np.array([vig_u, vig_b, vig_g])
vig_cnn[abs(vig_cnn) > 10**10] = 0
vig_cnn = np.transpose(vig_cnn, (1, 2, 3, 0))

### Format data for Jax
trainx = vig_cnn[~np.isin(arr, subset), :, :, :]
trainy = np.array([data_z[redshift_col][~np.isin(arr, subset)]]).T
trainy_err = np.array([data_z[redshift_col_err][~np.isin(arr, subset)]]).T

testx = vig_cnn[np.isin(arr, subset), :, :, :]
testy = np.array([data_z[redshift_col][subset]]).T
testy_err = np.array([data_z[redshift_col_err][subset]]).T

### Save training and test numpy arrays
save_data = os.getcwd().replace('ml_redshifts/Jax_models', 'data') + '/cnn_data/'
np.save(save_data + 'trainx_cnn_jax.npy', trainx)
np.save(save_data + 'trainy_cnn_jax.npy', trainy)
np.save(save_data + 'trainy_err_cnn_jax.npy', trainy_err)
np.save(save_data + 'testx_cnn_jax.npy', testx)
np.save(save_data + 'testy_cnn_jax.npy', testy)
np.save(save_data + 'testy_err_cnn_jax.npy', testy_err)

print('Successfully saved data to' + save_data)