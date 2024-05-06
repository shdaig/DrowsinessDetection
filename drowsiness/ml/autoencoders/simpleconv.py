import tensorflow as tf
from keras import layers
from keras.models import Model


class ConvAutoencoder(Model):
    def __init__(self, latent_dim: int = 150):
        self.latent_dim = latent_dim
        super(ConvAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(12, 1000, 1)),
            layers.Conv2D(filters=64,
                          kernel_size=(3, 10),
                          activation='relu',
                          padding='valid',
                          strides=(1, 2)),
            layers.Conv2D(filters=32,
                          kernel_size=(3, 10),
                          activation='relu',
                          padding='valid',
                          strides=(1, 2)),
            layers.BatchNormalization(),
            layers.Conv2D(filters=16,
                          kernel_size=(3, 10),
                          activation='relu',
                          padding='valid',
                          strides=(1, 2)),
            layers.Conv2D(filters=8,
                          kernel_size=(3, 10),
                          activation='relu',
                          padding='valid',
                          strides=(1, 2)),
            layers.BatchNormalization(),
            layers.Conv2D(filters=4,
                          kernel_size=(3, 15),
                          activation='relu',
                          padding='valid',
                          strides=(1, 2)),
            layers.Flatten(),
            layers.Dense(self.latent_dim, activation="relu")
        ])

        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(self.latent_dim,)),
            layers.Dense(168, activation="relu"),
            layers.Reshape((2, 21, 4)),
            layers.Conv2DTranspose(filters=4,
                                   kernel_size=(3, 15),
                                   strides=(1, 2),
                                   activation='relu',
                                   padding='valid'),
            layers.BatchNormalization(),
            layers.Conv2DTranspose(filters=8,
                                   kernel_size=(3, 10),
                                   strides=(1, 2),
                                   activation='relu',
                                   padding='valid'),
            layers.Conv2DTranspose(filters=16,
                                   kernel_size=(3, 10),
                                   strides=(1, 2),
                                   activation='relu',
                                   padding='valid'),
            layers.BatchNormalization(),
            layers.Conv2DTranspose(filters=32,
                                   kernel_size=(3, 10),
                                   strides=(1, 2),
                                   activation='relu',
                                   padding='valid'),
            layers.Conv2DTranspose(filters=64,
                                   kernel_size=(3, 10),
                                   strides=(1, 2),
                                   activation='relu',
                                   padding='valid'),
            layers.Conv2D(filters=1,
                          kernel_size=(1, 1),
                          activation='sigmoid',
                          padding='valid')])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


if __name__ == "__main__":
    model = ConvAutoencoder()
    model.encoder.summary()
    model.decoder.summary()
