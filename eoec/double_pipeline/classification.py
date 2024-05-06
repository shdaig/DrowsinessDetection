import drowsiness.utils.path as path
import drowsiness.utils.global_configs as gcfg
import drowsiness.eeg.eeg as eeg
import drowsiness.preprocessing.eoec as eoec
from drowsiness.ml.autoencoders import convae_4channels as ae
from drowsiness.ml.classifiers import simplelstm
from drowsiness.ml.classifiers import simplegru
from drowsiness.ml.classifiers import simplernn

import os
from datetime import datetime

import numpy as np
import keras
import matplotlib.pyplot as plt

import tensorflow as tf


def _extract_features_labels_fp(file_name: str, sfreq: int, channels: list[str],
                             sampling_step: int, slicing_step: int,
                             slice_depth: int, signal_depth_seconds: int) -> tuple[np.ndarray, np.ndarray]:
    slice_normalization = True
    raw = eeg.read_fif(file_name)
    eo_samples, ec_samples = eoec.eoec_get_samples(raw, sfreq, step=sampling_step)
    eo_signal_chunks = eoec.get_long_signal(raw, channels=channels, samples=eo_samples,
                                            depth=signal_depth_seconds * sfreq)
    ec_signal_chunks = eoec.get_long_signal(raw, channels=channels, samples=ec_samples,
                                            depth=signal_depth_seconds * sfreq)
    eo_labels = np.full((eo_signal_chunks.shape[0],), 1)
    ec_labels = np.full((ec_signal_chunks.shape[0],), 0)
    eo_features = eoec.slice_signal_chunks(eo_signal_chunks, slice_depth, slicing_step,
                                           slice_normalization=slice_normalization, channels=True)
    ec_features = eoec.slice_signal_chunks(ec_signal_chunks, slice_depth, slicing_step,
                                           slice_normalization=slice_normalization, channels=True)
    features = np.concatenate((eo_features, ec_features))
    labels = np.concatenate((eo_labels, ec_labels))
    return features, labels


def _extract_features_labels_o(file_name: str, sfreq: int, channels: list[str],
                             sampling_step: int, slicing_step: int,
                             slice_depth: int, signal_depth_seconds: int) -> tuple[np.ndarray, np.ndarray]:
    slice_normalization = False
    raw = eeg.read_fif(file_name)
    eo_samples, ec_samples = eoec.eoec_get_samples(raw, sfreq, step=sampling_step)
    eo_signal_chunks = eoec.get_long_signal(raw, channels=channels, samples=eo_samples,
                                            depth=signal_depth_seconds * sfreq)
    ec_signal_chunks = eoec.get_long_signal(raw, channels=channels, samples=ec_samples,
                                            depth=signal_depth_seconds * sfreq)
    eo_labels = np.full((eo_signal_chunks.shape[0],), 1)
    ec_labels = np.full((ec_signal_chunks.shape[0],), 0)
    eo_features = eoec.slice_signal_chunks(eo_signal_chunks, slice_depth, slicing_step,
                                           slice_normalization=slice_normalization, channels=True)
    ec_features = eoec.slice_signal_chunks(ec_signal_chunks, slice_depth, slicing_step,
                                           slice_normalization=slice_normalization, channels=True)
    features = np.concatenate((eo_features, ec_features))
    mean = np.mean(features)
    std_dev = np.std(features)
    features = (features - mean) / std_dev
    labels = np.concatenate((eo_labels, ec_labels))
    return features, labels


def _extract_features_labels(file_name: str, sfreq: int,
                             channels_fp: list[str], channels_o: list[str],
                             sampling_step: int, slicing_step: int,
                             slice_depth: int, signal_depth_seconds: int) -> tuple[np.ndarray, np.ndarray]:
    features_fp, _ = _extract_features_labels_fp(file_name, sfreq, channels_fp, sampling_step, slicing_step,
                                                 slice_depth, signal_depth_seconds)
    features_o, labels = _extract_features_labels_o(file_name, sfreq, channels_o, sampling_step, slicing_step,
                                                    slice_depth, signal_depth_seconds)
    features = np.concatenate((features_fp, features_o), axis=2)
    return features, labels


if __name__ == "__main__":
    problem_files = ["borovensky/20231110/record.20231110T122518727325.raw.fif.gz",
                     "borovensky/20231115/record.20231115T130856758835.raw.fif.gz"]
    subjs = ["dshepelev", "egipko", "borovensky", "golenishev", "azhogin"]
    year = "2023"
    sfreq = 500

    classifier_type = "gru"

    saved_model_dir = os.path.join("temp_autoencoders", "best_models")

    train_results_dir = "temp_train_results"
    if not os.path.exists(train_results_dir):
        os.mkdir(train_results_dir)

    for subj in subjs:
        subj_train_results_dir = os.path.join(train_results_dir, subj)
        if not os.path.exists(subj_train_results_dir):
            os.mkdir(subj_train_results_dir)

        file_names, stripped_file_names = path.find_by_format(gcfg.PROJ_SORTED_PATH, f'**/{subj}/{year}*/*.raw.fif.gz')
        for problem_file in problem_files:
            if problem_file in stripped_file_names:
                del_idx = np.argwhere(stripped_file_names == problem_file)[0][0]
                stripped_file_names = np.delete(stripped_file_names, del_idx)
                file_names = np.delete(file_names, del_idx)
        print("\nAvailable files:\n")
        for i, name in enumerate(stripped_file_names):
            print(f"[{i}] {name}")
        print()

        slice_depth = 1000
        signal_depth_seconds = 60
        sampling_step = 250
        slicing_step = 250

        eoec_dataset = {}
        for file_idx in range(len(file_names)):
            eoec_dataset[file_idx] = _extract_features_labels(file_names[file_idx],
                                                              sfreq,
                                                              channels_fp=["Fp1", "Fp2"], channels_o=["O1", "O2"],
                                                              sampling_step=sampling_step, slicing_step=slicing_step,
                                                              slice_depth=slice_depth,
                                                              signal_depth_seconds=signal_depth_seconds)
        for k in range(1):
            now = datetime.now()
            date_time = now.strftime("%d_%m_%Y_%H%M%S")
            save_result_dir = os.path.join(subj_train_results_dir, f"{date_time}_{classifier_type}_{subj}_{year}")
            os.mkdir(save_result_dir)

            result_accuracy = {}

            for test_idx in eoec_dataset:
                with tf.device('/GPU:1'):
                    train_idxs = [train_idx for train_idx in eoec_dataset if train_idx != test_idx]
                    test_file = f"{stripped_file_names[test_idx].split('/')[0]}_{stripped_file_names[test_idx].split('/')[1]}"
                    x_train = np.concatenate([eoec_dataset[train_idx][0] for train_idx in train_idxs])
                    y_train = np.concatenate([eoec_dataset[train_idx][1] for train_idx in train_idxs])
                    x_test = eoec_dataset[test_idx][0]
                    y_test = eoec_dataset[test_idx][1]

                    print(x_train.shape)
                    print(y_train.shape)
                    print(x_test.shape)
                    print(y_test.shape)

                    model = ae.DeepConvNetAutoencoder(train=False)
                    model.load_weights(os.path.join(saved_model_dir, test_file, "model", f"model.weights")).expect_partial()

                    x_train_shape_0 = x_train.shape[0]
                    x_train_shape_1 = x_train.shape[1]
                    x_test_shape_0 = x_test.shape[0]
                    x_test_shape_1 = x_test.shape[1]
                    x_train = x_train.reshape(x_train_shape_0 * x_train_shape_1, x_train.shape[2], x_train.shape[3])
                    x_test = x_test.reshape(x_test_shape_0 * x_test_shape_1, x_test.shape[2], x_test.shape[3])
                    x_train = model.predict(x_train, batch_size=4)
                    x_test = model.predict(x_test, batch_size=4)
                    x_train = x_train.reshape(x_train_shape_0, x_train_shape_1, x_train.shape[1])
                    x_test = x_test.reshape(x_test_shape_0, x_test_shape_1, x_test.shape[1])

                    print(x_train.shape)
                    print(x_test.shape)

                    idx = np.random.permutation(len(x_train))
                    x_train = x_train[idx]
                    y_train = y_train[idx]
                    idx = np.random.permutation(len(x_test))
                    x_test = x_test[idx]
                    y_test = y_test[idx]

                    classifier = None
                    if classifier_type == "lstm":
                        classifier = simplelstm.SimpleLSTM(input_shape=(x_train.shape[1], x_train.shape[2]))
                    elif classifier_type == "gru":
                        classifier = simplegru.SimpleGRU(input_shape=(x_train.shape[1], x_train.shape[2]))
                    elif classifier_type == "rnn":
                        classifier = simplernn.SimpleRNN(input_shape=(x_train.shape[1], x_train.shape[2]))
                    classifier.compile(loss='binary_crossentropy',
                                       optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                                       metrics=['accuracy'])
                    classifier_history = classifier.fit(x_train,
                                                        y_train,
                                                        validation_data=(x_test, y_test),
                                                        epochs=200,
                                                        batch_size=8)

                    plt.plot(classifier_history.history['accuracy'])
                    plt.plot(classifier_history.history['val_accuracy'])
                    plt.title(f'model accuracy - test: {test_file}')
                    plt.ylabel('accuracy')
                    plt.xlabel('epoch')
                    plt.legend(['train', 'val'], loc='upper left')
                    plt.savefig(os.path.join(save_result_dir, f"{test_file}_accuracy.png"))
                    plt.close()

                    plt.plot(classifier_history.history['loss'])
                    plt.plot(classifier_history.history['val_loss'])
                    plt.title(f'model loss - test: {test_file}')
                    plt.ylabel('loss')
                    plt.xlabel('epoch')
                    plt.legend(['train', 'val'], loc='upper left')
                    plt.savefig(os.path.join(save_result_dir, f"{test_file}_loss.png"))
                    plt.close()

                    # val_accuracy_final = classifier.evaluate(x=x_test, y=y_test, batch_size=4)
                    # print(val_accuracy_final)
                    result_accuracy[test_file] = np.max(classifier_history.history['val_accuracy'])

                    y_pred = classifier.predict(x_test)
                    with open(os.path.join(save_result_dir, f"{test_file}_pred.npy"), 'wb') as f:
                        np.save(f, y_pred)

            result_acc = []
            with open(os.path.join(save_result_dir, 'loss.txt'), 'w') as f:
                for test_file in result_accuracy:
                    f.write(f"- val_loss {test_file}: {result_accuracy[test_file]}\n")
                    result_acc.append(result_accuracy[test_file])
                f.write(f"mean val_loss: {np.array(result_acc).mean()}\n")

