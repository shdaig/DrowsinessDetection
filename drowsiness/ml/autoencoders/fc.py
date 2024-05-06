import tensorflow as tf
from keras import layers
from keras.models import Model


class FCAutoencoder(Model):
    def __init__(self, latent_dim: int = 300):
        self.latent_dim = latent_dim
        super(FCAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(12, 1000, 1)),
            layers.Flatten(),
            layers.Dense(500, activation="relu"),
            layers.Dense(self.latent_dim, activation="relu")
        ])

        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(self.latent_dim,)),
            layers.Dense(500, activation="relu"),
            layers.Dense(12000, activation='sigmoid'),
            layers.Reshape((12, 1000, 1))])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


if __name__ == "__main__":
    model = FCAutoencoder()
    model.encoder.summary()
    model.decoder.summary()
