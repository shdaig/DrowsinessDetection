import tensorflow as tf
from keras import layers
from keras.models import Model


class DeepConvNetAutoencoder(Model):
    def __init__(self, latent_dim: int = 7, train: bool = True):
        self.latent_dim = latent_dim
        self.train = train
        super(DeepConvNetAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(1000,)),
            layers.Reshape((1000, 1)),
            layers.Conv1D(filters=64,
                          kernel_size=11,
                          activation='relu',
                          padding='valid',
                          strides=1),
            layers.MaxPooling1D(pool_size=3, strides=3),

            layers.Conv1D(filters=32,
                          kernel_size=10,
                          activation='relu',
                          padding='valid',
                          strides=1),
            layers.MaxPooling1D(pool_size=3, strides=3),

            layers.Conv1D(filters=16,
                          kernel_size=12,
                          activation='relu',
                          padding='valid',
                          strides=1),
            layers.MaxPooling1D(pool_size=3, strides=3),

            layers.Conv1D(filters=8,
                          kernel_size=12,
                          activation='relu',
                          padding='valid',
                          strides=1),
            layers.MaxPooling1D(pool_size=3, strides=3),

            layers.Conv1D(filters=1,
                          kernel_size=1,
                          activation='sigmoid',
                          padding='valid',
                          strides=1),
            layers.Flatten(),
        ])

        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(self.latent_dim,)),
            layers.Reshape((7, 1)),
            layers.Conv1DTranspose(filters=8,
                                   kernel_size=1,
                                   strides=1,
                                   activation='relu',
                                   padding='valid'),

            layers.UpSampling1D(size=3),
            layers.Conv1DTranspose(filters=16,
                                   kernel_size=12,
                                   strides=1,
                                   activation='relu',
                                   padding='valid'),

            layers.UpSampling1D(size=3),
            layers.Conv1DTranspose(filters=32,
                                   kernel_size=12,
                                   strides=1,
                                   activation='relu',
                                   padding='valid'),

            layers.UpSampling1D(size=3),
            layers.Conv1DTranspose(filters=64,
                                   kernel_size=10,
                                   strides=1,
                                   activation='relu',
                                   padding='valid'),

            layers.UpSampling1D(size=3),
            layers.Conv1DTranspose(filters=64,
                                   kernel_size=11,
                                   strides=1,
                                   activation='relu',
                                   padding='valid'),

            layers.Conv1D(filters=1,
                          kernel_size=1,
                          activation='sigmoid',
                          padding='same'),
            layers.Reshape((1000,))
        ])

    def call(self, x):
        if self.train:
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded
        else:
            encoded = self.encoder(x)
            return encoded


if __name__ == "__main__":
    model = DeepConvNetAutoencoder()
    model.encoder.summary()
    model.decoder.summary()
