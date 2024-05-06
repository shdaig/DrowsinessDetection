import os
import glob

import numpy as np

features_dir = os.path.join("..", "temp_features_len1000_step250")
train_file = "msurkov_20231026"
features_files = []
features_files += glob.glob(os.path.join(features_dir, train_file, "*.npy"))
features_files.sort()
print(len(features_files))

with open(features_files[0], 'rb') as f:
    sample = np.load(f)
print(sample.shape)

features_dir = os.path.join("..", "temp_encoded_features_len1000_step250",
                            "19_02_2024_0943_channelconvnet_latent56_filters64",
                            "msurkov_20231026")
train_file = "msurkov_20231026"
features_files = []
features_files += glob.glob(os.path.join(features_dir, train_file, "*.npy"))
features_files.sort()
print(len(features_files))

with open(features_files[0], 'rb') as f:
    sample = np.load(f)
print(sample.shape)
