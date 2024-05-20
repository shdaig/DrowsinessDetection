import drowsiness.utils.path as path
import drowsiness.utils.global_configs as gcfg
import drowsiness.eeg.eeg as eeg
import drowsiness.preprocessing.eoec as eoec

from drowsiness.ml.eegnet import eegnet

import os
from datetime import datetime

import keras.optimizers
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.callbacks import ModelCheckpoint


def _extract_features_labels(file_name: str, sfreq: int, step: int,
                             depth: int) -> tuple[np.ndarray, np.ndarray]:
    raw = eeg.read_fif(file_name)
    eo_samples, ec_samples = eoec.eoec_get_samples(raw, sfreq, step=step)
    eo_features = eoec.get_slices(raw, ["O1", "O2"], eo_samples, depth=depth, slice_normalization=False)
    ec_features = eoec.get_slices(raw, ["O1", "O2"], ec_samples, depth=depth, slice_normalization=False)
    eo_labels = np.full((eo_features.shape[0],), 1)
    ec_labels = np.full((ec_features.shape[0],), 0)
    features = np.concatenate((eo_features, ec_features))
    print(features.shape)
    mean = np.mean(features)
    std_dev = np.std(features)
    features = (features - mean) / std_dev
    print(features.shape)
    labels = np.concatenate((eo_labels, ec_labels))
    return features, labels


if __name__ == "__main__":

    problem_files = []
    subjs = ["dshepelev", "egipko", "borovensky", "golenishev", "azhogin"]  # "*" - all subjects
    # subjs = ["aaa"]
    year = "2023"
    sfreq = 500

    for subj in subjs:
        for k in range(3):
            now = datetime.now()
            date_time = now.strftime("%d_%m_%Y_%H%M%S")
            classifier_type = "eegnet"
            train_results_dir = "temp_train_results"
            subj_result_dir = os.path.join(train_results_dir, f"{subj}_{year}")
            if not os.path.exists(subj_result_dir):
                os.mkdir(subj_result_dir)
            save_result_dir = os.path.join(subj_result_dir, f"{date_time}_{classifier_type}_{subj}_{year}")
            os.mkdir(save_result_dir)

            result_accuracy = {}

            file_names, stripped_file_names = path.find_by_format(gcfg.PROJ_SORTED_PATH, f'**/{subj}/{year}*/*.raw.fif.gz')

            for problem_file in problem_files:
                if problem_file in stripped_file_names:
                    del_idx = np.argwhere(stripped_file_names == problem_file)[0][0]
                    stripped_file_names = np.delete(stripped_file_names, del_idx)
            print("\nAvailable files:\n")
            for i, name in enumerate(stripped_file_names):
                print(f"[{i}] {name}")
            print()
            # file_idx = 0  # int(input("Enter idx: "))
            # if file_idx not in range(0, len(stripped_file_names)):
            #     exit(0)

            depth = 1000
            step = 100

            eoec_dataset = {}
            for file_idx in range(len(file_names)):
                eoec_dataset[file_idx] = _extract_features_labels(file_names[file_idx], sfreq, step=step, depth=depth)

            for test_idx in eoec_dataset:
                with tf.device('/GPU:1'):
                    train_idxs = [train_idx for train_idx in eoec_dataset if train_idx != test_idx]
                    x_train = np.concatenate([eoec_dataset[train_idx][0] for train_idx in train_idxs])
                    y_train = np.concatenate([eoec_dataset[train_idx][1] for train_idx in train_idxs])
                    x_test = eoec_dataset[test_idx][0]
                    y_test = eoec_dataset[test_idx][1]

                    test_file = f"{stripped_file_names[test_idx].split('/')[0]}_{stripped_file_names[test_idx].split('/')[1]}"

                    os.mkdir(os.path.join(save_result_dir, f"{test_file}_weights"))
                    checkpoint_path = os.path.join(save_result_dir, f"{test_file}_weights", "cp.chkpt")
                    save_best_checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                                           save_best_only=True,
                                                           monitor='val_accuracy',
                                                           mode='max',
                                                           save_weights_only=True)
                    model = eegnet.EEGNet(depth, sfreq=500)
                    model.compile(loss='binary_crossentropy',
                                  optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                                  metrics=['accuracy'])
                    classifier_history = model.fit(x_train,
                                                   y_train,
                                                   validation_data=(x_test, y_test),
                                                   epochs=100,
                                                   batch_size=64,
                                                   callbacks=[save_best_checkpoint])

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

                    # result_accuracy[test_file] = np.max(classifier_history.history['val_accuracy'])
                    model.load_weights(checkpoint_path)
                    result_accuracy[test_file] = model.evaluate(x_test, y_test)[1]
                    y_pred = model.predict(x_test)
                    with open(os.path.join(save_result_dir, f"{test_file}_pred.npy"), 'wb') as f:
                        np.save(f, y_pred)

            result_acc = []
            with open(os.path.join(save_result_dir, 'loss.txt'), 'w') as f:
                for test_file in result_accuracy:
                    f.write(f"- val_accuracy {test_file}: {result_accuracy[test_file]}\n")
                    result_acc.append(result_accuracy[test_file])
                f.write(f"mean val_accuracy: {np.array(result_acc).mean()}\n")
