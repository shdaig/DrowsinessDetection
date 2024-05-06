import tensorflow as tf
from keras import layers
from keras.models import Model


class SimpleLSTM(Model):
    def __init__(self, input_shape: tuple[int, int]):
        super(SimpleLSTM, self).__init__()
        self.model = tf.keras.Sequential([
            layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(4),  # , return_sequences=True),
            # tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2),
            # tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    def call(self, x):
        label = self.model(x)
        return label


if __name__ == "__main__":
    model = SimpleLSTM(input_shape=(116, 56))
    model.model.summary()
