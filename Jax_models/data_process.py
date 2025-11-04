import numpy as np 
import matplotlib.pyplot as plt 
from astropy.io import fits
from astropy.table import Table
from astropy.table import vstack
import os


### Import SuperBIT table

fits_file_2384a = '/projects/mccleary_group/habjan.e/SuperBIT/data/Abell2384a_colors_mags.fits'
hdul_2384a = fits.open(fits_file_2384a)
data_2384a = Table(hdul_2384a[1].data)

fits_file_2384b = '/projects/mccleary_group/habjan.e/SuperBIT/data/Abell2384b_colors_mags.fits'
hdul_2384b = fits.open(fits_file_2384b)
data_2384b = Table(hdul_2384b[1].data)

fits_file_3667 = '/projects/mccleary_group/habjan.e/SuperBIT/data/Abell3667_colors_mags.fits'
hdul_3667 = fits.open(fits_file_3667)
data_3667 = Table(hdul_3667[1].data)

fits_file_3571 = '/projects/mccleary_group/habjan.e/SuperBIT/data/Abell3571_colors_mags.fits'
hdul_3571 = fits.open(fits_file_3571)
data_3571 = Table(hdul_3571[1].data)

fits_file_3827 = '/projects/mccleary_group/habjan.e/SuperBIT/data/Abell3827_colors_mags.fits'
hdul_3827 = fits.open(fits_file_3827)
data_3827 = Table(hdul_3827[1].data)

fits_file_cosmos113 = '/projects/mccleary_group/habjan.e/SuperBIT/data/COSMOS113_colors_mags.fits'
hdul_cosmos113 = fits.open(fits_file_cosmos113)
data_cosmos113 = Table(hdul_cosmos113[1].data)

fits_file_1689 = '/projects/mccleary_group/habjan.e/SuperBIT/data/Abell1689_colors_mags.fits'
hdul_1689 = fits.open(fits_file_1689)
data_1689 = Table(hdul_1689[1].data)

data = vstack([data_2384a, data_2384b, data_3667, data_3571, data_3827])
#data_desi = vstack([data_cosmos113, data_1689]) removed Abell 1689 because something seems to be wrong with vignettes
data_desi = vstack([data_cosmos113])

### Objects with LoVoCCS BPZ redshifts
redshift_col = 'Z_lovoccs'
redshift_col_err = 'ZERR_lovoccs'
data_z = data[~np.isnan(np.array(data[redshift_col])) & (np.array(data[redshift_col]) > 0)]

### Objects with DESI redshifts
cosmos_col = 'Z_desi'
cosmos_col_err = 'ZERR_desi'
data_desi_test = data_desi[~np.isnan(np.array(data_desi[cosmos_col])) & (np.array(data_desi[cosmos_col]) > 0)]

### Vignette data in different bands
vig_u = np.array(data_z['VIGNET_u_im'])
vig_b = np.array(data_z['VIGNET_b_im'])
vig_g = np.array(data_z['VIGNET_g_im'])

vig_u_test = np.array(data_desi_test['VIGNET_u_im'])
vig_b_test = np.array(data_desi_test['VIGNET_b_im'])
vig_g_test = np.array(data_desi_test['VIGNET_g_im'])

### Make training and test datasets for the CNN
vig_cnn = np.array([vig_u, vig_b, vig_g])
vig_cnn[abs(vig_cnn) > 10**10] = 0
vig_cnn = np.transpose(vig_cnn, (1, 2, 3, 0))

vig_cnn_test = np.array([vig_u_test, vig_b_test, vig_g_test])
vig_cnn_test[abs(vig_cnn_test) > 10**10] = 0
vig_cnn_test = np.transpose(vig_cnn_test, (1, 2, 3, 0))

### Size of validation dataset
val_lovoccs = 1000

### Random subset for validation data
arr_lovoccs = np.arange(len(data_z))
subset_lovoccs = np.random.choice(arr_lovoccs, size= val_lovoccs, replace=False)

### Format data for Jax
trainx = vig_cnn[~np.isin(arr_lovoccs, subset_lovoccs), :, :, :]
trainy = np.array([data_z[redshift_col]]).T[~np.isin(arr_lovoccs, subset_lovoccs)]
trainy_err = np.array([data_z[redshift_col_err]]).T[~np.isin(arr_lovoccs, subset_lovoccs)]

validationx = vig_cnn[np.isin(arr_lovoccs, subset_lovoccs), :, :, :]
validationy = np.array([data_z[redshift_col]]).T[np.isin(arr_lovoccs, subset_lovoccs)]
validationy_err = np.array([data_z[redshift_col_err]]).T[np.isin(arr_lovoccs, subset_lovoccs)]

testx = vig_cnn_test
testy = np.array([data_desi_test[cosmos_col]]).T
testy_err = np.array([data_desi_test[cosmos_col_err]]).T

### Save training and test numpy arrays
save_data = '/projects/mccleary_group/habjan.e/SuperBIT/data/cnn_data/'
np.save(save_data + 'trainx_cnn_jax.npy', trainx)
np.save(save_data + 'trainy_cnn_jax.npy', trainy)
np.save(save_data + 'trainy_err_cnn_jax.npy', trainy_err)
np.save(save_data + 'validationx_cnn_jax.npy', validationx)
np.save(save_data + 'validationy_cnn_jax.npy', validationy)
np.save(save_data + 'validationy_err_cnn_jax.npy', validationy_err)
np.save(save_data + 'testx_cnn_jax.npy', testx)
np.save(save_data + 'testy_cnn_jax.npy', testy)
np.save(save_data + 'testy_err_cnn_jax.npy', testy_err)

print('Successfully saved data to' + save_data)