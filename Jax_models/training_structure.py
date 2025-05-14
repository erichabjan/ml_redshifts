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

class JaxTraining:
    """
    This class defines all the functions needed to train a ML model in Jax
    """

    @staticmethod
    def create_dataloader(x, y, sigma, batch_size, shuffle=True):
        """
        Create a TensorFlow dataset from NumPy arrays.
        
        Args:
            x: Input features
            y: Target values
            sigma: Uncertainty/error values for each target
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the dataset
            
        Returns:
            A TensorFlow dataset
        """
        ds = tf.data.Dataset.from_tensor_slices((x, y, sigma))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(x))
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds
    
    @staticmethod
    def create_train_state(rng, model, learning_rate, input_shape):
        """
        Initialize model parameters and create a training state.
        
        Args:
            rng: JAX random number generator key
            model: Flax model to train
            learning_rate: Learning rate for the optimizer
            input_shape: Shape of the input data
            
        Returns:
            A TrainState containing initialized parameters and optimizer
        """
        params = model.init(rng, jnp.ones(input_shape))['params']
        tx = optax.adam(learning_rate)
        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    @staticmethod
    def weighted_mse_loss(params, x, y, sigma, apply_fn, rng, training=True):
        """
        Calculate weighted mean squared error loss.
        
        Args:
            params: Model parameters
            x: Input features
            y: Target values
            sigma: Uncertainty/error values for each target
            apply_fn: Function to apply the model
            rng: JAX random number generator key
            training: Whether in training mode (for dropout, batch norm, etc.)
            
        Returns:
            Weighted MSE loss value
        """
        # If using dropout, you'd use this line instead:
        # preds = apply_fn({'params': params}, x, rngs={'dropout': rng}, training=training)
        preds = apply_fn({'params': params}, x, training=training)
        preds = preds.squeeze()
        residual = (preds - y) / sigma
        return jnp.mean(residual**2)
    
    @staticmethod
    @jax.jit
    def train_step(state, batch, rng):
        """
        Perform a single training step (forward pass, loss computation, backward pass).
        
        Args:
            state: Current training state
            batch: Batch of data (x, y, sigma)
            rng: JAX random number generator key
            
        Returns:
            Updated training state
        """
        x, y, sigma = batch
        
        def loss_fn(params):
            return JaxTraining.weighted_mse_loss(params, x, y, sigma, state.apply_fn, rng, training=True)
        
        grads = jax.grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads)
    
    @staticmethod
    @jax.jit
    def eval_step(state, batch, rng):
        """
        Perform a single evaluation step.
        
        Args:
            state: Current training state
            batch: Batch of data (x, y, sigma)
            rng: JAX random number generator key
            
        Returns:
            Loss value for this batch
        """
        x, y, sigma = batch
        loss = JaxTraining.weighted_mse_loss(
            state.params, x, y, sigma, state.apply_fn, rng, training=False
        )
        return loss
    
    @staticmethod
    def train_model(train_ds, test_ds, model, epochs=50, batch_size=16, learning_rate=1e-3):
        """
        Train a model using the specified datasets.
        
        Args:
            train_ds: Training dataset
            test_ds: Test/validation dataset
            model: Flax model to train
            epochs: Number of training epochs
            batch_size: Batch size (for shape inference if not in dataset)
            learning_rate: Learning rate for the optimizer
            
        Returns:
            Trained model state
        """
        # Initialize master RNG key
        master_rng = jax.random.PRNGKey(0)
        
        # Determine the input shape from the first batch
        for x_sample, _, _ in tfds.as_numpy(train_ds.take(1)):
            input_shape = x_sample.shape
            break
        
        # Create and initialize the training state with a separate RNG key
        master_rng, init_rng = jax.random.split(master_rng)
        state = JaxTraining.create_train_state(init_rng, model, learning_rate, input_shape)

        # Training loop
        for epoch in range(epochs):
            # Get a new RNG key for this epoch's training
            master_rng, train_epoch_rng = jax.random.split(master_rng)
            
            # Train on batches
            for x_batch, y_batch, sigma_batch in tfds.as_numpy(train_ds):
                # Get a new RNG key for this specific batch
                train_epoch_rng, step_rng = jax.random.split(train_epoch_rng)
                state = JaxTraining.train_step(state, (x_batch, y_batch, sigma_batch), step_rng)
            
            # Get a separate RNG key for evaluation
            master_rng, eval_rng = jax.random.split(master_rng)
            
            # Evaluate model on test data
            total_loss = 0.0
            count = 0
            for x_test, y_test, sigma_test in tfds.as_numpy(test_ds):
                # Get a batch-specific evaluation RNG
                eval_rng, batch_eval_rng = jax.random.split(eval_rng)
                loss_val = JaxTraining.eval_step(state, (x_test, y_test, sigma_test), batch_eval_rng)
                total_loss += loss_val
                count += 1
            
            # Calculate and report average test loss
            avg_loss = total_loss / max(count, 1)
            print(f"Epoch {epoch+1}/{epochs}, Test Loss: {avg_loss:.4f}")
            
        return state

    @staticmethod
    def predict(state, x, batch_size=32):
        """
        Make predictions with a trained model.
        
        Args:
            state: Trained model state
            x: Input features
            batch_size: Batch size for predictions
            
        Returns:
            Array of predictions
        """
        # Create a dataset for prediction
        pred_ds = tf.data.Dataset.from_tensor_slices(x).batch(batch_size)
        
        # Initialize an empty list to store predictions
        predictions = []
        
        # Get predictions batch by batch
        for x_batch in tfds.as_numpy(pred_ds):
            # Forward pass without dropout
            preds = state.apply_fn({'params': state.params}, x_batch, training=False)
            predictions.append(preds)
        
        # Concatenate all batch predictions
        return jnp.concatenate(predictions, axis=0).squeeze()