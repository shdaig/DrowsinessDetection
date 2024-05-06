import utils.eeg as eeg
import utils.path as path
import utils.global_configs as gcfg
from statesutils.model_selection.sample_former import get_samples_labels
from utils.color_print import *

import tensorflow as tf
import keras.backend as k
from keras import layers, losses
from keras.models import Model
from keras.callbacks import Callback


import numpy as np
import scipy.signal as signal
import mne

import os
import shutil
import time
import glob
import gc


class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        k.clear_session()


class BlinkAutoencoder(Model):
    def __init__(self, latent_dim, shape):
        super(BlinkAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.shape = shape
        self.encoder = tf.keras.Sequential([
          layers.Dense(128, activation="relu", name="dense_1"),
          layers.Dense(64, activation="relu", name="dense_2"),
          layers.Dense(32, activation="relu", name="dense_3"),
          layers.Dense(latent_dim, activation="relu", name="dense_4")])

        self.decoder = tf.keras.Sequential([
          layers.Dense(32, activation="relu", name="dense_5"),
          layers.Dense(64, activation="relu", name="dense_6"),
          layers.Dense(128, activation="relu", name="dense_7"),
          layers.Dense(shape, activation="sigmoid", name="dense_8")])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def bandpass_filter(data, lowcut, highcut, signal_freq, filter_order):
    nyquist_freq = 0.5 * signal_freq
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b, a = signal.butter(filter_order, [low, high], btype="band")
    y = signal.lfilter(b, a, data)
    return y


def __time_to_samples(time_string: str, sample_rate: int) -> int:
    h_units, m_units, s_units = time_string.split(":")
    return (int(h_units) * 60 * 60 + int(m_units) * 60 + int(s_units)) * sample_rate


def get_sleep_state_from_file(raw: mne.io.Raw, labels_file: str) -> np.ndarray:
    begins = []
    ends = []
    sample_rate = 500
    with open(os.path.join("../autoencoders/video_labels", labels_file), 'r') as f:
        lines = f.read().split('\n')
        for line in lines:
            sleep_time = line.split(';')[0]
            begin_time, end_time = sleep_time.split('-')
            begins.append(__time_to_samples(begin_time, sample_rate))
            ends.append(__time_to_samples(end_time, sample_rate))

    last_sample = len(raw.times)

    states = []
    start_indx = 0
    for i in range(len(begins)):
        for j in range(start_indx, begins[i]):
            states.append(1)
        for j in range(begins[i], ends[i]):
            states.append(0)
        start_indx = ends[i]
    for i in range(start_indx, last_sample):
        states.append(1)

    return np.array(states)


def remkdir(dirname: str):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)


def samples_generator(files):
    for file in files:
        with open(file, 'rb') as f:
            sample = np.load(f)

        yield sample


if __name__ == "__main__":
    file_names, stripped_file_names = path.find_by_format(gcfg.PROJ_SORTED_PATH, '**/*.raw.fif.gz')
    printlg("\nAvailable files:\n")
    for i, name in enumerate(stripped_file_names):
        print(f"[{i}] {name}")
    print()

    subject_files = {"msurkov_20231026": {"idx": 2, "labels_file": "msurkov_20231026.txt"},
                     "msurkov_20231030": {"idx": 3, "labels_file": "msurkov_20231030.txt"},
                     "msurkov_20231116": {"idx": 5, "labels_file": "msurkov_20231116.txt"}}
    models_dir = "temp_saved_models"
    features_dir = "../temp_features_dir"
    labels_dir = "temp_labels_dir"
    #
    # remkdir(features_dir)
    # remkdir(labels_dir)
    #
    # start = time.time()
    #
    # for file in subject_files:
    #     idx = subject_files[file]["idx"]
    #     print(f"[{idx}] {stripped_file_names[idx]} loading...")
    #
    #     raw = eeg.read_fif(file_names[idx])
    #     times, channel_names, channel_data = eeg.fetch_channels(raw)
    #     events, events_id = mne.events_from_annotations(raw, verbose=0)
    #
    #     # get labels for eeg signal
    #     sleep_state = get_sleep_state_from_file(raw, subject_files[file]["labels_file"])
    #     print(f"[{idx}] {file} labeled")
    #
    #     fp1, fp2 = channel_data[channel_names == "Fp1"][0], channel_data[channel_names == "Fp2"][0]
    #     fp_avg = np.clip((fp1 + fp2) / 2, -0.0002, 0.0002)
    #     fp_avg = bandpass_filter(fp_avg, lowcut=0.1,
    #                              highcut=30.0,
    #                              signal_freq=500,
    #                              filter_order=4)
    #
    #     events_samples = np.concatenate((events[events[:, 2] == events_id['GZ']][:, 0],
    #                                      events[events[:, 2] == events_id['GO']][:, 0]))
    #     events_samples = np.sort(events_samples)
    #     try:
    #         work_start = events_samples[events_samples < len(times) // 2][-1] + 30 * 500
    #     except IndexError:
    #         work_start = 0
    #     try:
    #         work_end = events_samples[events_samples > len(times) // 2][0]
    #     except IndexError:
    #         work_end = len(times)
    #
    #     fp_avg = fp_avg[work_start: work_end]
    #     sleep_state = sleep_state[work_start: work_end]
    #
    #     x_raw, y = get_samples_labels(fp_avg,
    #                                   sleep_state,
    #                                   data_depth=1,
    #                                   prediction_horizon=0)
    #
    #     print(f"[{idx}] {file} samples formed. Features array shape: {x_raw.shape}")
    #
    #     os.mkdir(os.path.join(features_dir, file))
    #     for i in range(x_raw.shape[0]):
    #         save_path = os.path.join(features_dir, file, f"{i}.npy")
    #         with open(save_path, 'wb') as f:
    #             np.save(f, x_raw[i])
    #
    #     os.mkdir(os.path.join(labels_dir, file))
    #     for i in range(y.shape[0]):
    #         save_path = os.path.join(labels_dir, file, f"{i}.npy")
    #         with open(save_path, 'wb') as f:
    #             np.save(f, y[i])
    #
    # print("Dataset formed")
    #
    # end = time.time()
    # print(end - start)

    for test_file in subject_files:
        train_files = [train_file for train_file in subject_files if train_file != test_file]
        print()
        print(f"Test: {test_file}")
        print(f"Train: {train_files}")

        test_features_files = glob.glob(os.path.join(features_dir, test_file, "*.npy"))
        train_features_files = []
        for train_file in train_files:
            train_features_files += glob.glob(os.path.join(features_dir, train_file, "*.npy"))
        test_features_files.sort()
        train_features_files.sort()

        test_labels_files = glob.glob(os.path.join(labels_dir, test_file, "*.npy"))
        train_labels_files = []
        for train_file in train_files:
            train_labels_files += glob.glob(os.path.join(labels_dir, train_file, "*.npy"))
        test_labels_files.sort()
        train_labels_files.sort()

        train_dataset_x = tf.data.Dataset.from_generator(generator=lambda: samples_generator(train_features_files),
                                                         output_shapes=(None, 500),
                                                         output_types=tf.float64)
        train_dataset_y = tf.data.Dataset.from_generator(generator=lambda: samples_generator(train_labels_files),
                                                         output_shapes=(),
                                                         output_types=tf.int64)
        test_dataset_x = tf.data.Dataset.from_generator(generator=lambda: samples_generator(test_features_files),
                                                        output_shapes=(None, 500),
                                                        output_types=tf.float64)
        test_dataset_y = tf.data.Dataset.from_generator(generator=lambda: samples_generator(test_labels_files),
                                                        output_shapes=(),
                                                        output_types=tf.int64)

        latent_dim = 30
        autoencoder = BlinkAutoencoder(latent_dim, 500)
        autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

        print("autoencoder weights loading...")
        autoencoder.load_weights(os.path.join(models_dir, test_file, f"{test_file}_ae.weights")).expect_partial()
        print("autoencoder weights loaded")

        # # encoded_test_data = autoencoder.encoder(autoencoder_x_test).numpy().reshape((test_samples, test_depth, -1))
        #
        # # print(f"Encoded test features dataset shape: {encoded_test_data.shape}")
        # print(f"Encoded train features dataset shape: {encoded_train_data.shape}")
        #
        # model = tf.keras.Sequential([
        #     tf.keras.layers.LSTM(128),
        #     tf.keras.layers.Dropout(0.2),
        #     tf.keras.layers.Dense(1, activation='sigmoid')
        # ])
        #
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        #
        # history = model.fit(encoded_train_data, y_train, epochs=20, batch_size=512)


