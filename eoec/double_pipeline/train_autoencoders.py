import shutil

import keras.losses

import drowsiness.utils.path as path
import drowsiness.utils.global_configs as gcfg
import drowsiness.eeg.eeg as eeg
import drowsiness.preprocessing.eoec as eoec
from drowsiness.ml.autoencoders import convae_4channels as ae

from datetime import datetime

import matplotlib.pyplot as plt

import os

import numpy as np


def _extract_features_labels_fp(file_name: str, sfreq: int,
                             channels: list[str],
                             step: int, depth: int) -> tuple[np.ndarray, np.ndarray]:
    slice_normalization = True
    raw = eeg.read_fif(file_name)
    eo_samples, ec_samples = eoec.eoec_get_samples(raw, sfreq, step=step)
    eo_features = eoec.get_slices(raw, channels, eo_samples, depth=depth, slice_normalization=slice_normalization)
    ec_features = eoec.get_slices(raw, channels, ec_samples, depth=depth, slice_normalization=slice_normalization)
    eo_labels = np.full((eo_features.shape[0],), 1)
    ec_labels = np.full((ec_features.shape[0],), 0)
    features = np.concatenate((eo_features, ec_features))
    # mean = np.mean(features)
    # std_dev = np.std(features)
    # features = (features - mean) / std_dev
    labels = np.concatenate((eo_labels, ec_labels))
    return features, labels


def _extract_features_labels_o(file_name: str, sfreq: int,
                             channels: list[str],
                             step: int, depth: int) -> tuple[np.ndarray, np.ndarray]:
    slice_normalization = False
    raw = eeg.read_fif(file_name)
    eo_samples, ec_samples = eoec.eoec_get_samples(raw, sfreq, step=step)
    eo_features = eoec.get_slices(raw, channels, eo_samples, depth=depth, slice_normalization=slice_normalization)
    ec_features = eoec.get_slices(raw, channels, ec_samples, depth=depth, slice_normalization=slice_normalization)
    eo_labels = np.full((eo_features.shape[0],), 1)
    ec_labels = np.full((ec_features.shape[0],), 0)
    features = np.concatenate((eo_features, ec_features))
    mean = np.mean(features)
    std_dev = np.std(features)
    features = (features - mean) / std_dev
    labels = np.concatenate((eo_labels, ec_labels))
    return features, labels


def _extract_features_labels(file_name: str, sfreq: int,
                             channels_fp: list[str], channels_o: list[str],
                             step: int, depth: int) -> tuple[np.ndarray, np.ndarray]:
    features_fp, _ = _extract_features_labels_fp(file_name, sfreq, channels_fp, step, depth)
    features_o, labels = _extract_features_labels_o(file_name, sfreq, channels_o, step, depth)

    features = np.concatenate((features_fp, features_o), axis=1)
    return features, labels


if __name__ == "__main__":
    subjs = ["dshepelev", "egipko", "borovensky", "golenishev", "azhogin"]
    year = "2023"
    sfreq = 500

    trained_ae_dir = "temp_autoencoders"
    if not os.path.exists(trained_ae_dir):
        os.mkdir(trained_ae_dir)

    for subj in subjs:
        file_names, stripped_file_names = path.find_by_format(gcfg.PROJ_SORTED_PATH, f'**/{subj}/{year}*/*.raw.fif.gz')
        print("\nAvailable files:\n")
        for i, name in enumerate(stripped_file_names):
            print(f"[{i}] {name}")
        print()
        # file_idx = 0  # int(input("Enter idx: "))
        # if file_idx not in range(0, len(stripped_file_names)):
        #     exit(0)

        depth = 1000
        step = 50

        eoec_dataset = {}
        for file_idx in range(len(file_names)):
            eoec_dataset[file_idx] = _extract_features_labels(file_names[file_idx], sfreq,
                                                              channels_fp=["Fp1", "Fp2"],
                                                              channels_o=["O1", "O2"], step=step,
                                                              depth=depth)

        for k in range(3):
            now = datetime.now()
            date_time = now.strftime("%d_%m_%Y_%H%M%S")
            classifier_type = "4channels"
            save_model_results_dir = os.path.join(trained_ae_dir, f"{date_time}_{classifier_type}_{subj}_{year}")
            os.mkdir(save_model_results_dir)

            for test_idx in eoec_dataset:
                test_file = f"{stripped_file_names[test_idx].split('/')[0]}_{stripped_file_names[test_idx].split('/')[1]}"
                model_save_dir = os.path.join(save_model_results_dir, test_file)
                if os.path.exists(model_save_dir):
                    shutil.rmtree(model_save_dir)
                os.mkdir(model_save_dir)

                train_idxs = [train_idx for train_idx in eoec_dataset if train_idx != test_idx]
                x_train = np.concatenate([eoec_dataset[train_idx][0] for train_idx in train_idxs])
                x_test = eoec_dataset[test_idx][0]

                idx = np.random.permutation(len(x_train))
                x_train = x_train[idx]
                idx = np.random.permutation(len(x_test))
                x_test = x_test[idx]

                model = ae.DeepConvNetAutoencoder(train=True)
                model.compile(loss=keras.losses.MeanSquaredError(),
                              optimizer=keras.optimizers.Adam(learning_rate=1e-5),
                              metrics=keras.metrics.MeanSquaredError())
                history = model.fit(x_train,
                                    x_train,
                                    validation_data=(x_test, x_test),
                                    epochs=100,
                                    batch_size=64)

                test_file = f"{stripped_file_names[test_idx].split('/')[0]}_{stripped_file_names[test_idx].split('/')[1]}"

                os.mkdir(os.path.join(model_save_dir, "model"))
                model.save_weights(os.path.join(model_save_dir, "model", f"model.weights"))

                plt.plot(history.history['mean_squared_error'])
                plt.plot(history.history['val_mean_squared_error'])
                plt.title(f'model loss(mse) - test: {test_file}')
                plt.ylabel('mse')
                plt.xlabel('epoch')
                plt.legend(['train', 'val'], loc='upper left')
                plt.savefig(os.path.join(model_save_dir, f"{test_file}.png"))
                plt.close()





