import tensorflow as tf
from keras import layers
from keras.models import Model


class DeepConvNetAutoencoder(Model):
    def __init__(self, latent_dim: int = 56, train: bool = False):
        self.train = train
        self.latent_dim = latent_dim
        super(DeepConvNetAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(4, 1000)),
            layers.Reshape((4, 1000, 1)),
            layers.Conv2D(filters=64,
                          kernel_size=(1, 11),
                          activation='relu',
                          padding='valid',
                          strides=(1, 1)),
            layers.MaxPooling2D(pool_size=(1, 3), strides=(1, 3)),

            layers.Conv2D(filters=32,
                          kernel_size=(1, 10),
                          activation='relu',
                          padding='valid',
                          strides=(1, 1)),
            layers.MaxPooling2D(pool_size=(1, 3), strides=(1, 3)),

            layers.Conv2D(filters=16,
                          kernel_size=(1, 12),
                          activation='relu',
                          padding='valid',
                          strides=(1, 1)),
            layers.MaxPooling2D(pool_size=(1, 3), strides=(1, 3)),

            layers.Conv2D(filters=8,
                          kernel_size=(1, 12),
                          activation='relu',
                          padding='valid',
                          strides=(1, 1)),
            layers.MaxPooling2D(pool_size=(1, 3), strides=(1, 3)),

            layers.Conv2D(filters=8,
                          kernel_size=(4, 1),
                          activation='relu',
                          padding='valid',
                          strides=(1, 1)),
            layers.Flatten(),
        ])

        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(self.latent_dim,)),
            layers.Reshape((1, 7, 8)),

            layers.Conv2DTranspose(filters=8,
                                   kernel_size=(4, 1),
                                   strides=(1, 1),
                                   activation='relu',
                                   padding='valid'),

            layers.UpSampling2D(size=(1, 3)),
            layers.Conv2DTranspose(filters=8,
                                   kernel_size=(1, 12),
                                   strides=(1, 1),
                                   activation='relu',
                                   padding='valid'),
            layers.UpSampling2D(size=(1, 3)),
            layers.Conv2DTranspose(filters=16,
                                   kernel_size=(1, 12),
                                   strides=(1, 1),
                                   activation='relu',
                                   padding='valid'),
            layers.UpSampling2D(size=(1, 3)),
            layers.Conv2DTranspose(filters=32,
                                   kernel_size=(1, 10),
                                   strides=(1, 1),
                                   activation='relu',
                                   padding='valid'),
            layers.UpSampling2D(size=(1, 3)),
            layers.Conv2DTranspose(filters=64,
                                   kernel_size=(1, 11),
                                   strides=(1, 1),
                                   activation='relu',
                                   padding='valid'),

            layers.Conv2D(filters=1,
                          kernel_size=(1, 1),
                          activation='sigmoid',
                          padding='same'),
            layers.Reshape((4, 1000))
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
