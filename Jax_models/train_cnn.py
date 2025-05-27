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
import os
import sys
import pickle

### Custom code 
sys.path.append(os.getcwd() + '/Jax_models')
from training_structure import JaxTraining
from jax_models import CNNModel

### Add a suffix for a new model
suffix = ''

### Import data
data_path = os.getcwd().replace('ml_redshifts/Jax_models', 'data') + '/cnn_data/'

trainx = np.load(data_path + 'trainx_cnn_jax.npy')
trainy = np.load(data_path + 'trainy_cnn_jax.npy')
trainy_err = np.load(data_path + 'trainy_err_cnn_jax.npy')
testx = np.load(data_path + 'testx_cnn_jax.npy')
testy = np.load(data_path + 'testy_cnn_jax.npy')
testy_err = np.load(data_path + 'testy_err_cnn_jax.npy')

# Create data loaders
batch_size = 128
train_ds = JaxTraining.create_dataloader(trainx, trainy, trainy_err, batch_size=batch_size, shuffle=True)
test_ds = JaxTraining.create_dataloader(testx, testy, testy_err, batch_size=batch_size, shuffle=False)

### Train model
if __name__ == "__main__":
    # Define hyperparameters
    epochs = 500
    learning_rate = 1e-3
    
    # Create and train the model
    model = CNNModel()
    trained_state, train_loss, test_loss = JaxTraining.train_model(
        train_ds=train_ds, 
        test_ds=test_ds, 
        model=model, 
        epochs=epochs, 
        batch_size=batch_size,
        learning_rate=learning_rate
    )

    # Save model parameters
    save_path = os.getcwd() + '/cnn_model_params' + suffix + '.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(trained_state.params, f)
    
    # Save loss curves
    save_data = os.getcwd().replace('ml_redshifts/Jax_models', 'data') + '/cnn_data/'
    np.save(save_data + 'train_loss.npy', train_loss)
    np.save(save_data + 'test_loss.npy', test_loss)   