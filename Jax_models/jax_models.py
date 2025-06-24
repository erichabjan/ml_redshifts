from flax import linen as nn

class CNNModel(nn.Module):
    @nn.compact
    def __call__(self, x, training=True):
        x = nn.Conv(features=64, kernel_size=(3, 3), kernel_init=nn.initializers.xavier_uniform())(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.05, deterministic=not training)(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), kernel_init=nn.initializers.xavier_uniform())(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.05, deterministic=not training)(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=128, kernel_size=(3, 3), kernel_init=nn.initializers.xavier_uniform())(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.05, deterministic=not training)(x)
        x = nn.Conv(features=128, kernel_size=(3, 3), kernel_init=nn.initializers.xavier_uniform())(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.05, deterministic=not training)(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(256, kernel_init=nn.initializers.xavier_uniform())(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.05, deterministic=not training)(x)

        x = nn.Dense(128, kernel_init=nn.initializers.xavier_uniform())(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.05, deterministic=not training)(x)

        x = nn.Dense(1)(x)
        return x

class CNN_deep(nn.Module):
    @nn.compact
    def __call__(self, x, training=True):
        x = nn.Conv(features=128, kernel_size=(3, 3), kernel_init=nn.initializers.xavier_uniform(), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.05, deterministic=not training)(x)
        x = nn.Conv(features=128, kernel_size=(3, 3), kernel_init=nn.initializers.xavier_uniform(), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.05, deterministic=not training)(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=256, kernel_size=(3, 3), kernel_init=nn.initializers.xavier_uniform(), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.05, deterministic=not training)(x)
        x = nn.Conv(features=256, kernel_size=(3, 3), kernel_init=nn.initializers.xavier_uniform(), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.05, deterministic=not training)(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(256, kernel_init=nn.initializers.xavier_uniform())(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.05, deterministic=not training)(x)

        x = nn.Dense(128, kernel_init=nn.initializers.xavier_uniform())(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.05, deterministic=not training)(x)

        x = nn.Dense(64, kernel_init=nn.initializers.xavier_uniform())(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.05, deterministic=not training)(x)

        x = nn.Dense(1)(x)
        return x


class CNN_shallow(nn.Module):
    @nn.compact
    def __call__(self, x, training=True):
        x = nn.Conv(features=128, kernel_size=(3, 3), kernel_init=nn.initializers.xavier_uniform(), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=128, kernel_size=(3, 3), kernel_init=nn.initializers.xavier_uniform(), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(128, kernel_init=nn.initializers.xavier_uniform())(x)
        x = nn.relu(x)

        x = nn.Dense(64, kernel_init=nn.initializers.xavier_uniform())(x)
        x = nn.relu(x)

        x = nn.Dense(1)(x)
        return x