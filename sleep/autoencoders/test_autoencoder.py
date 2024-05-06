import gc
import glob
import os
import time
from datetime import datetime

import tensorflow as tf
from keras import losses
import numpy as np
import matplotlib.pyplot as plt

from drowsiness.ml.autoencoders import fc
from drowsiness.ml.autoencoders import simpleconv
from drowsiness.ml.autoencoders import deepconvnet
from drowsiness.ml.autoencoders import channelconvnet_168


def samples_generator(files):
    for file in files:
        with open(file, 'rb') as f:
            samples = np.load(f)
        yield samples


if __name__ == "__main__":
    features_dir = "../temp_features_dir"
    train_results_dir = "temp_train_results"
    subject_files = ["msurkov_20231026", "msurkov_20231030", "msurkov_20231116"]
    now = datetime.now()
    date_time = now.strftime("%d_%m_%Y_%H%M")
    ae_type = "deepconvnet_1"

    save_result_dir = os.path.join(train_results_dir, f"{date_time}_{ae_type}")
    os.mkdir(save_result_dir)

    result_loss = []

    for test_file in subject_files:
        train_files = [train_file for train_file in subject_files if train_file != test_file]
        print()
        print(f"Test: {test_file}")
        print(f"Train: {train_files}")

        train_features_files = []
        for train_file in train_files:
            train_features_files += glob.glob(os.path.join(features_dir, train_file, "*.npy"))
        train_features_files.sort()
        print(len(train_features_files))

        test_features_files = glob.glob(os.path.join(features_dir, test_file, "*.npy"))
        test_features_files.sort()

        num_files = len(train_features_files)
        with open(train_features_files[0], 'rb') as f:
            sample = np.load(f)
        num_slices, num_channels, num_samples, _ = sample.shape
        del sample
        gc.collect()

        train_dataset_x = tf.data.Dataset.from_generator(generator=lambda: samples_generator(train_features_files),
                                                         output_shapes=(num_slices, num_channels, num_samples, 1),
                                                         output_types=tf.float64)
        autoencoder_train_dataset = tf.data.Dataset.zip((train_dataset_x, train_dataset_x))
        autoencoder_train_dataset = autoencoder_train_dataset.unbatch()
        autoencoder_train_dataset = autoencoder_train_dataset.batch(batch_size=64)

        test_dataset_x = tf.data.Dataset.from_generator(generator=lambda: samples_generator(test_features_files),
                                                        output_shapes=(num_slices, num_channels, num_samples, 1),
                                                        output_types=tf.float64)
        autoencoder_test_dataset = tf.data.Dataset.zip((test_dataset_x, test_dataset_x))
        autoencoder_test_dataset = autoencoder_test_dataset.unbatch()
        autoencoder_test_dataset = autoencoder_test_dataset.batch(batch_size=64)

        # autoencoder = simpleconv.ConvAutoencoder()
        # autoencoder = deepconvnet.DeepConvNetAutoencoder()
        autoencoder = channelconvnet_168.DeepConvNetAutoencoder()
        autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
        start = time.time()
        print("autoencoder learning start...")
        history = autoencoder.fit(autoencoder_train_dataset,
                                  validation_data=autoencoder_test_dataset,
                                  epochs=30)
        print("autoencoder learning end...")
        end = time.time()
        print(end - start)

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'model mse - test: {test_file}')
        plt.ylabel('mse')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(os.path.join(save_result_dir, f"{test_file}.png"))
        plt.close()

        os.mkdir(os.path.join(save_result_dir, test_file))
        autoencoder.save_weights(os.path.join(save_result_dir, test_file, f"{test_file}_ae.weights"))

        val_loss_final = autoencoder.evaluate(autoencoder_test_dataset)
        print(val_loss_final)
        result_loss.append(val_loss_final)

    with open(os.path.join(save_result_dir, 'loss.txt'), 'w') as f:
        for i, loss in enumerate(result_loss):
            f.write(f"- val_loss {subject_files[i]}: {loss}\n")
        f.write(f"mean val_loss: {np.array(result_loss).mean()}\n")
