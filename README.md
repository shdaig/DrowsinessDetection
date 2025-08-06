# 👁 Drowsiness Detection

A deep learning-based system for detecting blinks and classifying open/closed eye states using EEG signals from frontal and ococcipital electrodes.

## Description

The project focuses on developing algorithms for automatic analysis of electroencephalographic (EEG) signals aimed at identifying two key phenomena:

- Blink Artifacts Detection: Blinks artifacts are detected in the signal recorded from frontal electrodes through preprocessing using an autoencoder to encode the signal efficiently. The encoded representation captures important features, which are then classified using a recurrent neural network (RNN) capable of handling temporal dependencies present in sequential signals.

- Alpha Rhythm Emergence: Alpha rhythm emergence is investigated in occipital electrode recordings. For alpha wave detection, we employ [EEGNet](https://arxiv.org/abs/1611.08024) - a specialized convolutional deep-learning framework designed specifically for spatio-temporal feature extraction from EEG data.

## ⚙️ Configuration Setup

Before running the application, it's essential to configure settings by creating a config.ini file. Follow these steps:

1. Copy the provided `config_template.ini` file to `config.ini`
    ```bash
    cp config_template.ini config.ini
    ```
2. Edit the newly created `config.ini` file to reflect your specific dataset path.
