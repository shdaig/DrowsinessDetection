import tensorflow as tf
from keras import layers
from keras.models import Model


class EEGNet(Model):
    def __init__(self,
                 num_samples: int,
                 sfreq: int,
                 f1: int = 4,
                 f2: int = 8,
                 d: int = 2):
        super(EEGNet, self).__init__()
        kernel_length = sfreq // 2
        self.classifier = tf.keras.Sequential([
            layers.Input(shape=(2, num_samples)),
            layers.Reshape((2, num_samples, 1)),
            layers.Conv2D(filters=f1, kernel_size=(1, kernel_length), strides=(1, 1), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.DepthwiseConv2D(kernel_size=(2, 1), depth_multiplier=d, padding="valid", activation='relu'),
            layers.BatchNormalization(),
            layers.ELU(),
            layers.AvgPool2D(pool_size=(1, 4)),
            layers.Dropout(rate=0.25),
            layers.SeparableConv2D(filters=f2, kernel_size=(1, 16), strides=(1, 1), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.ELU(),
            layers.AvgPool2D(pool_size=(1, 8)),
            layers.Dropout(rate=0.25),
            layers.Flatten(),
            layers.Dense(units=1, activation="sigmoid")
        ])

    def call(self, x):
        classified = self.classifier(x)
        return classified


if __name__ == "__main__":
    model = EEGNet(250, 500)
    model.classifier.summary()
