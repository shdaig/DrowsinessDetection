import drowsiness.utils.path as path
import drowsiness.utils.global_configs as gcfg
import drowsiness.eeg.eeg as eeg

import numpy as np
import mne

import os
import gc
import json


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def _fetch_full_path(subject: str, subjects_list: np.ndarray) -> str:
    """
    :param subject: Subject's file identifier in format subjectname_yyyymmdd
    :param subjects_list: General list of available files
    :return: Filepath from general list of available files
    """
    subject_name, experiment_date = subject.split("_")
    path_to_find = os.path.join(subject_name, experiment_date)
    for name in subjects_list:
        if name.find(path_to_find) != -1:
            return name


def _fetch_filtered_eeg_data(raw: mne.io.Raw) -> np.ndarray:
    """
    :param raw: Raw data from .fif file
    :return: Numpy array of EEG data from .fif filtered with Butterworth bandpass filter. Shape: (samples, channels)
    """
    eeg_data = []
    _, channel_names, channel_data = eeg.fetch_channels(raw)
    for channel_name in channel_names:
        if channel_name in mne.channels.make_standard_montage('standard_1020').ch_names:
            eeg_channel_data = channel_data[channel_names == channel_name][0]
            filtered_eeg_channel_data = eeg.bandpass_filter(eeg_channel_data, lowcut=0.1, highcut=40.0, signal_freq=500, filter_order=4)
            eeg_data.append(filtered_eeg_channel_data)
    return np.array(eeg_data).transpose()


def _time_to_samples(time_string: str, sample_rate: int) -> int:
    """
    :param time_string: Time string with format hh:mm:ss
    :param sample_rate: Sample rate
    :return: Sample index according to time
    """
    h_units, m_units, s_units = time_string.split(":")
    return (int(h_units) * 60 * 60 + int(m_units) * 60 + int(s_units)) * sample_rate


def _get_states_from_file(raw: mne.io.Raw, labels_path: str) -> np.ndarray:
    begins = []
    ends = []
    sample_rate = 500
    with open(labels_path, 'r') as f:
        lines = f.read().split('\n')
        for line in lines:
            sleep_time = line.split(';')[0]
            begin_time, end_time = sleep_time.split('-')
            begins.append(_time_to_samples(begin_time, sample_rate))
            ends.append(_time_to_samples(end_time, sample_rate))
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


def _fetch_work_time(raw: mne.io.Raw) -> tuple[int, int]:
    times, _, _ = eeg.fetch_channels(raw)
    events, events_id = mne.events_from_annotations(raw, verbose=0)
    events_samples = np.concatenate((events[events[:, 2] == events_id['GZ']][:, 0],
                                     events[events[:, 2] == events_id['GO']][:, 0]))
    events_samples = np.sort(events_samples)
    try:
        work_start = events_samples[events_samples < len(times) // 2][-1] + 30 * 500
    except IndexError:
        work_start = 0
    try:
        work_end = events_samples[events_samples > len(times) // 2][0]
    except IndexError:
        work_end = len(times)
    return work_start, work_end


def _get_samples_labels(eeg_chanel_data: np.ndarray,
                       state_labels: np.ndarray,
                       data_depth: int,
                       prediction_horizon: int = 0) -> tuple[np.ndarray, np.ndarray]:
    step = 250
    slice_size = 1000
    signal_slices = []
    labels = []
    data_depth_conv = data_depth * 60 * 500
    for i in range(data_depth_conv, eeg_chanel_data.shape[0] - (prediction_horizon * 60 * 500), step * 4):
        signal_slices.append(eeg_chanel_data[i - data_depth_conv: i])
        labels.append(state_labels[int(i - 1 + (prediction_horizon * 60 * 500))])
    features = []
    for signal_slice in signal_slices:
        slice_features = []
        for i in range(slice_size, len(signal_slice), step):
            x_window = signal_slice[i - slice_size: i]
            min_val = np.min(x_window, axis=0)
            max_val = np.max(x_window, axis=0)
            x_window = (x_window - min_val) / (max_val - min_val)
            slice_features.append(x_window.transpose())
        features.append(slice_features)
    features_np = np.array(features)
    labels_np = np.array(labels)
    return features_np, labels_np


def _get_samples_labels_optimized(features_dir: str,
                                  subject_file: str,
                                  eeg_chanel_data: np.ndarray,
                                  state_labels: np.ndarray,
                                  data_depth: int,
                                  prediction_horizon: int = 0) -> np.ndarray:
    step = 250
    slice_size = 1000
    signal_slices = []
    labels = []
    data_depth_conv = data_depth * 60 * 500
    for i in range(data_depth_conv, eeg_chanel_data.shape[0] - (prediction_horizon * 60 * 500), step * 4):
        signal_slices.append(eeg_chanel_data[i - data_depth_conv: i])
        labels.append(state_labels[int(i - 1 + (prediction_horizon * 60 * 500))])
    print(f"{subject_file} saving...")
    os.mkdir(os.path.join(features_dir, subject_file))
    for j, signal_slice in enumerate(signal_slices):
        slice_features = []
        for i in range(slice_size, len(signal_slice), step):
            x_window = signal_slice[i - slice_size: i]
            min_val = np.min(x_window, axis=0)
            max_val = np.max(x_window, axis=0)
            x_window = (x_window - min_val) / (max_val - min_val)
            slice_features.append(x_window.transpose())
        slice_features = np.array(slice_features)
        slice_features = np.expand_dims(slice_features, axis=-1)
        save_path = os.path.join(features_dir, subject_file, f"{j:04d}.npy")
        with open(save_path, 'wb') as f:
            np.save(f, slice_features)
        del slice_features
        gc.collect()
    return np.array(labels)


def generate_subject_dataset_optimized(features_dir: str, subject_file: str) -> np.ndarray:
    print(f"{subject_file} loading...")
    file_names, _ = path.find_by_format(gcfg.PROJ_SORTED_PATH, '**/*.raw.fif.gz')
    fif_subject_file = _fetch_full_path(subject_file, file_names)
    raw = eeg.read_fif(fif_subject_file)
    eeg_data = _fetch_filtered_eeg_data(raw)
    states = _get_states_from_file(raw, os.path.join("video_labels", f"{subject_file}.txt"))
    work_start, work_end = _fetch_work_time(raw)
    del raw
    gc.collect()
    eeg_data = eeg_data[work_start: work_end, :]
    states = states[work_start: work_end]
    y = _get_samples_labels_optimized(features_dir,
                                      subject_file,
                                      eeg_data,
                                      states,
                                      data_depth=1)
    del eeg_data
    del states
    gc.collect()
    print(f"{subject_file} loaded")
    return y


def generate_subject_dataset(subject_file: str) -> tuple[np.ndarray, np.ndarray]:
    print(f"{subject_file} loading...")
    file_names, _ = path.find_by_format(gcfg.PROJ_SORTED_PATH, '**/*.raw.fif.gz')
    fif_subject_file = _fetch_full_path(subject_file, file_names)
    raw = eeg.read_fif(fif_subject_file)
    eeg_data = _fetch_filtered_eeg_data(raw)
    states = _get_states_from_file(raw, os.path.join("video_labels", f"{subject_file}.txt"))
    work_start, work_end = _fetch_work_time(raw)
    del raw
    gc.collect()
    eeg_data = eeg_data[work_start: work_end, :]
    states = states[work_start: work_end]
    x_raw, y = _get_samples_labels(eeg_data,
                                   states,
                                   data_depth=1)
    del eeg_data
    del states
    gc.collect()
    x_raw = np.expand_dims(x_raw, axis=-1)
    print(f"{subject_file} loaded")
    return x_raw, y


if __name__ == "__main__":
    optimized = False
    save_only_labels = True
    if optimized:
        features_dir = "../temp_features_len1000_step250"
        path.remkdir(features_dir)

        subject_files = ["msurkov_20231026", "msurkov_20231030", "msurkov_20231116"]

        for subject_file in subject_files:
            y = generate_subject_dataset_optimized(features_dir, subject_file)
            print(y.shape)
            del y
            gc.collect()
            print(f"{subject_file} saved")
    else:
        features_dir = "../temp_features_len1000_step250"
        if save_only_labels:
            subject_files = ["msurkov_20231026", "msurkov_20231030", "msurkov_20231116"]
            labels_dict = {}
            for subject_file in subject_files:
                x_raw, y = generate_subject_dataset(subject_file)
                print(x_raw.shape)
                del x_raw
                gc.collect()
                print(y.shape)
                print(f"{subject_file} saving labels...")
                for i in range(y.shape[0]):
                    save_path = os.path.join(subject_file, f"{i:04d}.npy")
                    labels_dict[save_path] = y[i]
            with open(os.path.join(features_dir, 'labels.json'), 'w') as fp:
                json.dump(labels_dict, fp, cls=NpEncoder)
        else:
            path.remkdir(features_dir)
            subject_files = ["msurkov_20231026", "msurkov_20231030", "msurkov_20231116"]
            for subject_file in subject_files:
                x_raw, y = generate_subject_dataset(subject_file)
                print(x_raw.shape)
                print(y.shape)
                print(f"{subject_file} saving...")
                os.mkdir(os.path.join(features_dir, subject_file))
                for i in range(x_raw.shape[0]):
                    save_path = os.path.join(features_dir, subject_file, f"{i:04d}.npy")
                    with open(save_path, 'wb') as f:
                        np.save(f, x_raw[i])
                del x_raw
                del y
                gc.collect()
                print(f"{subject_file} saved")
