import numpy as np 
import matplotlib.pyplot as plt 
from astropy.io import fits
from astropy.table import Table
from astropy.table import vstack
import os


### Import SuperBIT table
fits_file_cosmos113 = '/home/habjan.e/SuperBIT_code/Redshift_ml/data/COSMOS113_colors_mags.fits'
hdul_cosmos113 = fits.open(fits_file_cosmos113)
data_cosmos113 = Table(hdul_cosmos113[1].data)

### Object with DESI redshifts
cosmos_col = 'Z_desi'
cosmos_col_err = 'ZERR_desi'
data_cosmos = data_cosmos113[~np.isnan(np.array(data_cosmos113[cosmos_col])) & (np.array(data_cosmos113[cosmos_col]) > 0)]

### Size of test dataset
test_size = 50

### Random subset for test data
arr = np.arange(len(data_cosmos))
subset = np.random.choice(arr, size=test_size, replace=False)

### Vignette data in different bands
vig_u = np.array(data_cosmos['VIGNET_u_im'])
vig_b = np.array(data_cosmos['VIGNET_b_im'])
vig_g = np.array(data_cosmos['VIGNET_g_im'])

### Make training and test datasets for the CNN
vig_cnn = np.array([vig_u, vig_b, vig_g])
vig_cnn[abs(vig_cnn) > 10**10] = 0
vig_cnn = np.transpose(vig_cnn, (1, 2, 3, 0))

### Format data for Jax
trainx = vig_cnn[~np.isin(arr, subset), :, :, :]
trainy = np.array([data_cosmos[cosmos_col][~np.isin(arr, subset)]]).T
trainy_err = np.array([data_cosmos[cosmos_col_err][~np.isin(arr, subset)]]).T

testx = vig_cnn[np.isin(arr, subset), :, :, :]
testy = np.array([data_cosmos[cosmos_col][subset]]).T
testy_err = np.array([data_cosmos[cosmos_col_err][subset]]).T

### Save training and test numpy arrays
save_data = os.getcwd().replace('ml_redshifts/Jax_models', 'data') + '/cnn_data/'
np.save(save_data + 'trainx_cnn_jax.npy', trainx)
np.save(save_data + 'trainy_cnn_jax.npy', trainy)
np.save(save_data + 'trainy_err_cnn_jax.npy', trainy_err)
np.save(save_data + 'testx_cnn_jax.npy', testx)
np.save(save_data + 'testy_cnn_jax.npy', testy)
np.save(save_data + 'testy_err_cnn_jax.npy', testy_err)

print('Successfully saved data to' + save_data)