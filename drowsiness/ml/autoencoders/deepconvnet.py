import tensorflow as tf
from keras import layers
from keras.models import Model


class DeepConvNetAutoencoder(Model):
    def __init__(self, latent_dim: int = 300):
        self.latent_dim = latent_dim
        super(DeepConvNetAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(12, 1000, 1)),
            layers.Conv2D(filters=25,
                          kernel_size=(1, 11),
                          activation='relu',
                          padding='valid',
                          strides=(1, 1)),
            layers.Conv2D(filters=25,
                          kernel_size=(12, 1),
                          activation='relu',
                          padding='valid',
                          strides=(1, 1)),
            layers.Conv2D(filters=25,
                          kernel_size=(1, 3),
                          activation='relu',
                          padding='valid',
                          strides=(1, 3)),
            # layers.MaxPooling2D(pool_size=(1, 3), strides=(1, 3)),

            layers.Conv2D(filters=50,
                          kernel_size=(1, 10),
                          activation='relu',
                          padding='valid',
                          strides=(1, 1)),
            layers.Conv2D(filters=50,
                          kernel_size=(1, 3),
                          activation='relu',
                          padding='valid',
                          strides=(1, 3)),
            # layers.MaxPooling2D(pool_size=(1, 3), strides=(1, 3)),

            layers.Conv2D(filters=100,
                          kernel_size=(1, 12),
                          activation='relu',
                          padding='valid',
                          strides=(1, 1)),
            layers.Conv2D(filters=100,
                          kernel_size=(1, 3),
                          activation='relu',
                          padding='valid',
                          strides=(1, 3)),
            # layers.MaxPooling2D(pool_size=(1, 3), strides=(1, 3)),

            layers.Conv2D(filters=200,
                          kernel_size=(1, 12),
                          activation='relu',
                          padding='valid',
                          strides=(1, 1)),
            layers.Conv2D(filters=200,
                          kernel_size=(1, 3),
                          activation='relu',
                          padding='valid',
                          strides=(1, 3)),
            # layers.MaxPooling2D(pool_size=(1, 3), strides=(1, 3)),

            layers.Conv2D(filters=400,
                          kernel_size=(1, 7),
                          activation='relu',
                          padding='valid',
                          strides=(1, 1)),

            layers.Flatten(),
            layers.Dense(self.latent_dim, activation="relu")
        ])

        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(self.latent_dim,)),
            layers.Dense(400, activation="relu"),
            layers.Reshape((1, 1, 400)),

            layers.Conv2DTranspose(filters=400,
                                   kernel_size=(1, 7),
                                   strides=(1, 1),
                                   activation='relu',
                                   padding='valid'),

            # layers.UpSampling2D(size=(1, 3)),
            layers.Conv2DTranspose(filters=200,
                                   kernel_size=(1, 3),
                                   strides=(1, 3),
                                   activation='relu',
                                   padding='valid'),
            layers.Conv2DTranspose(filters=200,
                                   kernel_size=(1, 12),
                                   strides=(1, 1),
                                   activation='relu',
                                   padding='valid'),

            # layers.UpSampling2D(size=(1, 3)),
            layers.Conv2DTranspose(filters=100,
                                   kernel_size=(1, 3),
                                   strides=(1, 3),
                                   activation='relu',
                                   padding='valid'),
            layers.Conv2DTranspose(filters=100,
                                   kernel_size=(1, 12),
                                   strides=(1, 1),
                                   activation='relu',
                                   padding='valid'),

            # layers.UpSampling2D(size=(1, 3)),
            layers.Conv2DTranspose(filters=50,
                                   kernel_size=(1, 3),
                                   strides=(1, 3),
                                   activation='relu',
                                   padding='valid'),
            layers.Conv2DTranspose(filters=50,
                                   kernel_size=(1, 10),
                                   strides=(1, 1),
                                   activation='relu',
                                   padding='valid'),

            # layers.UpSampling2D(size=(1, 3)),
            layers.Conv2DTranspose(filters=25,
                                   kernel_size=(1, 3),
                                   strides=(1, 3),
                                   activation='relu',
                                   padding='valid'),
            layers.Conv2DTranspose(filters=25,
                                   kernel_size=(12, 1),
                                   strides=(1, 1),
                                   activation='relu',
                                   padding='valid'),
            layers.Conv2DTranspose(filters=25,
                                   kernel_size=(1, 11),
                                   strides=(1, 1),
                                   activation='relu',
                                   padding='valid'),
            layers.Conv2D(filters=1,
                          kernel_size=(1, 1),
                          activation='sigmoid',
                          padding='same')])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


if __name__ == "__main__":
    model = DeepConvNetAutoencoder()
    model.encoder.summary()
    model.decoder.summary()
