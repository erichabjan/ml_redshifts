import numpy as np 
import matplotlib.pyplot as plt 
from astropy.io import fits
from astropy.table import Table
from astropy.table import vstack
import os


### Import SuperBIT table

fits_file_2384a = '/home/habjan.e/SuperBIT_code/Redshift_ml/data/Abell2384a_colors_mags.fits'
hdul_2384a = fits.open(fits_file_2384a)
data_2384a = Table(hdul_2384a[1].data)

fits_file_2384b = '/home/habjan.e/SuperBIT_code/Redshift_ml/data/Abell2384b_colors_mags.fits'
hdul_2384b = fits.open(fits_file_2384b)
data_2384b = Table(hdul_2384b[1].data)

fits_file_3667 = '/home/habjan.e/SuperBIT_code/Redshift_ml/data/Abell3667_colors_mags.fits'
hdul_3667 = fits.open(fits_file_3667)
data_3667 = Table(hdul_3667[1].data)

fits_file_3571 = '/home/habjan.e/SuperBIT_code/Redshift_ml/data/Abell3571_colors_mags.fits'
hdul_3571 = fits.open(fits_file_3571)
data_3571 = Table(hdul_3571[1].data)

fits_file_3827 = '/home/habjan.e/SuperBIT_code/Redshift_ml/data/Abell3827_colors_mags.fits'
hdul_3827 = fits.open(fits_file_3827)
data_3827 = Table(hdul_3827[1].data)

fits_file_cosmos113 = '/home/habjan.e/SuperBIT_code/Redshift_ml/data/COSMOS113_colors_mags.fits'
hdul_cosmos113 = fits.open(fits_file_cosmos113)
data_cosmos113 = Table(hdul_cosmos113[1].data)

data = vstack([data_2384a, data_2384b, data_3667, data_3571, data_3827])

### Object with LoVoCCS BPZ redshifts
redshift_col = 'Z_lovoccs'
redshift_col_err = 'ZERR_lovoccs'
data_z = data[~np.isnan(np.array(data[redshift_col])) & (np.array(data[redshift_col]) > 0)]

### Object with DESI redshifts
cosmos_col = 'Z_desi'
cosmos_col_err = 'ZERR_desi'
data_cosmos = data_cosmos113[~np.isnan(np.array(data_cosmos113[cosmos_col])) & (np.array(data_cosmos113[cosmos_col]) > 0)]

### Size of test dataset
test_lovoccs = 1000
test_desi = 50

### Random subset for test data
arr_lovoccs = np.arange(len(data_z))
subset_lovoccs = np.random.choice(arr_lovoccs, size=test_lovoccs, replace=False)

arr_desi = np.arange(len(data_cosmos))
subset_desi = np.random.choice(arr_desi, size=test_desi, replace=False)

### Vignette data in different bands
vig_u_lovoccs = np.array(data_z['VIGNET_u_im'])
vig_b_lovoccs = np.array(data_z['VIGNET_b_im'])
vig_g_lovoccs = np.array(data_z['VIGNET_g_im'])

vig_u_desi = np.array(data_cosmos['VIGNET_u_im'])
vig_b_desi = np.array(data_cosmos['VIGNET_b_im'])
vig_g_desi = np.array(data_cosmos['VIGNET_g_im'])

### Make training and test datasets for the CNN
vig_lovoccs = np.array([vig_u_lovoccs, vig_b_lovoccs, vig_g_lovoccs])
vig_lovoccs[abs(vig_lovoccs) > 10**10] = 0
vig_lovoccs = np.transpose(vig_lovoccs, (1, 2, 3, 0))

vig_desi = np.array([vig_u_desi, vig_b_desi, vig_g_desi])
vig_desi[abs(vig_desi) > 10**10] = 0
vig_desi = np.transpose(vig_desi, (1, 2, 3, 0))

### Format data for Jax
trainx = np.concatenate((vig_desi[~np.isin(arr_desi, subset_desi), :, :, :], vig_lovoccs[~np.isin(arr_lovoccs, subset_lovoccs), :, :, :]))
trainy = np.concatenate((np.array([data_cosmos[cosmos_col][~np.isin(arr_desi, subset_desi)]]).T, np.array([data_z[redshift_col][~np.isin(arr_lovoccs, subset_lovoccs)]]).T))
trainy_err = np.concatenate((np.array([data_cosmos[cosmos_col_err][~np.isin(arr_desi, subset_desi)]]).T, np.array([data_z[redshift_col_err][~np.isin(arr_lovoccs, subset_lovoccs)]]).T))

testx = np.concatenate((vig_desi[np.isin(arr_desi, subset_desi), :, :, :], vig_lovoccs[np.isin(arr_lovoccs, subset_lovoccs), :, :, :]))
testy = np.concatenate((np.array([data_cosmos[cosmos_col][np.isin(arr_desi, subset_desi)]]).T, np.array([data_z[redshift_col][np.isin(arr_lovoccs, subset_lovoccs)]]).T))
testy_err = np.concatenate((np.array([data_cosmos[cosmos_col_err][np.isin(arr_desi, subset_desi)]]).T, np.array([data_z[redshift_col_err][np.isin(arr_lovoccs, subset_lovoccs)]]).T))

### Save training and test numpy arrays
save_data = os.getcwd().replace('ml_redshifts/Jax_models', 'data') + '/cnn_data/'
np.save(save_data + 'trainx_cnn_jax.npy', trainx)
np.save(save_data + 'trainy_cnn_jax.npy', trainy)
np.save(save_data + 'trainy_err_cnn_jax.npy', trainy_err)
np.save(save_data + 'testx_cnn_jax.npy', testx)
np.save(save_data + 'testy_cnn_jax.npy', testy)
np.save(save_data + 'testy_err_cnn_jax.npy', testy_err)

print('Successfully saved data to' + save_data)