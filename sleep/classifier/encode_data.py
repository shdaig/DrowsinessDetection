import gc
import glob

import drowsiness.utils.path as path
from drowsiness.ml.autoencoders import channelconvnet_latent56_filters64

import tensorflow as tf
from keras import losses
import numpy as np

import os
import time


def samples_generator(files):
    for file in files:
        with open(file, 'rb') as f:
            samples = np.load(f)
        yield samples


if __name__ == "__main__":
    features_dir = "../temp_features_len1000_step250"
    model_name = "19_02_2024_0943_channelconvnet_latent56_filters64"
    encoded_features_dir = os.path.join("../temp_encoded_features_len1000_step250", model_name)
    saved_model_dir = os.path.join("model", model_name)

    subject_files = ["msurkov_20231026", "msurkov_20231030", "msurkov_20231116"]

    path.remkdir(encoded_features_dir)

    for test_file in subject_files:
        print(f"Model will be loaded for - {test_file}")

        encoded_save_dir = os.path.join(encoded_features_dir, test_file)
        os.mkdir(encoded_save_dir)

        autoencoder = channelconvnet_latent56_filters64.DeepConvNetAutoencoder()

        autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
        autoencoder.load_weights(os.path.join(saved_model_dir, test_file, f"{test_file}_ae.weights")).expect_partial()

        for subject_file in subject_files:
            print(f"Encoding {subject_file}")
            encoded_subject_save_dir = os.path.join(encoded_save_dir, subject_file)
            os.mkdir(encoded_subject_save_dir)

            raw_features_files = glob.glob(os.path.join(features_dir, subject_file, "*.npy"))
            raw_features_files.sort()
            print(f"{len(raw_features_files)} elements will be encoded...")

            start_time = time.time()
            for raw_features_file in raw_features_files:
                f_name = raw_features_file.split('/')[-1]
                with open(raw_features_file, 'rb') as f:
                    sample = np.load(f)
                encoded_sample = autoencoder.encoder(sample)
                with open(os.path.join(encoded_subject_save_dir, f_name), 'wb') as f:
                    np.save(f, encoded_sample)
                del sample
                del encoded_sample
                gc.collect()
            end_time = time.time()
            print(f"Encoding time: {end_time - start_time} sec")







