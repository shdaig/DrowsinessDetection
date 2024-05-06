import tensorflow as tf
from keras import layers
from keras.models import Model


class SimpleRNN(Model):
    def __init__(self, input_shape: tuple[int, int]):
        super(SimpleRNN, self).__init__()
        self.model = tf.keras.Sequential([
            layers.Input(shape=input_shape),
            tf.keras.layers.SimpleRNN(4),
            # tf.keras.layers.GRU(4, return_sequences=True),  # , dropout=0.2),
            # tf.keras.layers.GRU(2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    def call(self, x):
        label = self.model(x)
        return label


if __name__ == "__main__":
    model = SimpleRNN(input_shape=(116, 56))
    model.model.summary()
