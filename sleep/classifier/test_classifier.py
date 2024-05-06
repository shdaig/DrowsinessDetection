from datetime import datetime

import keras.optimizers
import numpy as np
from matplotlib import pyplot as plt

from drowsiness.ml.classifiers import simplelstm
from drowsiness.ml.classifiers import simplegru

import os
import glob
import json

if __name__ == "__main__":
    # model_name = "13_02_2024_1409_channelconvnet_1536"
    model_name = "19_02_2024_0943_channelconvnet_latent56_filters64"
    encoded_features_dir = os.path.join("../temp_encoded_features_len1000_step250", model_name)
    now = datetime.now()
    date_time = now.strftime("%d_%m_%Y_%H%M")
    classifier_type = "gru"
    train_results_dir = "temp_train_results"
    save_result_dir = os.path.join(train_results_dir, f"{date_time}_{classifier_type}")
    os.mkdir(save_result_dir)

    subject_files = ["msurkov_20231026", "msurkov_20231030", "msurkov_20231116"]
    result_accuracy = {}

    with open(os.path.join(encoded_features_dir, 'labels.json')) as json_file:
        labels_dict = json.load(json_file)

    for test_file in subject_files:
        train_files = [train_file for train_file in subject_files if train_file != test_file]
        print()
        print(f"Test: {test_file}")
        print(f"Train: {train_files}")

        test_features_files = glob.glob(os.path.join(encoded_features_dir, test_file, test_file, "*.npy"))
        test_features_files.sort()
        test_samples = []
        test_labels = []
        for test_features_file in test_features_files:
            with open(test_features_file, 'rb') as f:
                test_samples.append(np.load(f))
            label_key = os.path.join(test_features_file.split('/')[-2], test_features_file.split('/')[-1])
            test_labels.append(labels_dict[label_key])
        test_samples = np.array(test_samples)
        test_labels = np.array(test_labels)

        train_features_files = []
        for train_file in train_files:
            train_features_files += glob.glob(os.path.join(encoded_features_dir, test_file, train_file, "*.npy"))
        train_features_files.sort()
        train_samples = []
        train_labels = []
        for train_features_file in train_features_files:
            with open(train_features_file, 'rb') as f:
                train_samples.append(np.load(f))
            label_key = os.path.join(train_features_file.split('/')[-2], train_features_file.split('/')[-1])
            train_labels.append(labels_dict[label_key])
        train_samples = np.array(train_samples)
        train_labels = np.array(train_labels)

        classifier = None
        if classifier_type == "lstm":
            classifier = simplelstm.SimpleLSTM()
        elif classifier_type == "gru":
            classifier = simplegru.SimpleGRU()
        classifier.compile(loss='binary_crossentropy',
                           optimizer=keras.optimizers.Adam(learning_rate=1e-5),
                           metrics=['accuracy'])
        classifier_history = classifier.fit(train_samples,
                                            train_labels,
                                            validation_data=(test_samples, test_labels),
                                            epochs=30,
                                            batch_size=4)

        plt.plot(classifier_history.history['accuracy'])
        plt.plot(classifier_history.history['val_accuracy'])
        plt.title(f'model accuracy - test: {test_file}')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(os.path.join(save_result_dir, f"{test_file}.png"))
        plt.close()

        val_accuracy_final = classifier.evaluate(x=test_samples, y=test_labels, batch_size=4)
        print(val_accuracy_final)
        result_accuracy[test_file] = val_accuracy_final[1]

    result_acc = []
    with open(os.path.join(save_result_dir, 'loss.txt'), 'w') as f:
        for test_file in result_accuracy:
            f.write(f"- val_loss {test_file}: {result_accuracy[test_file]}\n")
            result_acc.append(result_accuracy[test_file])
        f.write(f"mean val_loss: {np.array(result_acc).mean()}\n")
