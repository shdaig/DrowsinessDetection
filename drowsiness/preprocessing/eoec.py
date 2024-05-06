import mne
import numpy as np
import drowsiness.eeg.eeg as eeg


def eoec_get_samples(raw: mne.io.Raw, sfreq: int, step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    :param raw: MNE Raw object extracted from fif
    :param sfreq: Sample frequency
    :param step: Size of sampling step for event indexes
    :return: Sample indexes that entered in EO/EC event range (EO samples, EC samples)
    """
    events, events_id = mne.events_from_annotations(raw, verbose=0)
    gz_idxs = np.sort(events[events[:, 2] == events_id['GZ']][:, 0])
    go_idxs = np.sort(events[events[:, 2] == events_id['GO']][:, 0])
    gz_samples = []
    go_samples = []
    for idx in gz_idxs:
        gz_samples.append(list(range(idx, idx + sfreq * 30, step)))
    for idx in go_idxs:
        go_samples.append(list(range(idx, idx + sfreq * 30, step)))
    gz_samples = np.array(gz_samples).flatten()
    go_samples = np.array(go_samples).flatten()

    return go_samples, gz_samples


def get_slices(raw: mne.io.Raw, channels: list[str],
               samples: np.ndarray, depth: int,
               slice_normalization: bool) -> np.ndarray:
    """
    :param raw: MNE Raw object extracted from fif
    :param channels: Names of required channels
    :param samples: Indexes of signal samples, as a final points of slices
    :param depth: Depth of slices
    :param slice_normalization: Set normalization inside slices
    :return: Array of slices with shape: (num_samples, num_channels, depth)
    """
    _, channel_names, data = eeg.fetch_channels(raw)
    print(channel_names)
    channels_idxs = []
    for channel in channels:
        channels_idxs.append(np.argwhere(channel_names == channel)[0][0])
    channel_data = data[channels_idxs]
    slices = []
    for sample in samples:
        signal_slice = channel_data[:, sample - depth: sample].copy()
        if slice_normalization:
            min_val = np.min(signal_slice)
            max_val = np.max(signal_slice)
            signal_slice = (signal_slice - min_val) / (max_val - min_val)
        slices.append(signal_slice)
    slices = np.array(slices)
    return slices


def get_long_signal(raw: mne.io.Raw, channels: list[str],
                    samples: np.ndarray, depth: int) -> np.ndarray:
    _, channel_names, data = eeg.fetch_channels(raw)
    channels_idxs = []
    for channel in channels:
        channels_idxs.append(np.argwhere(channel_names == channel)[0][0])
    channel_data = data[channels_idxs]
    slices = []
    for sample in samples:
        signal_slice = channel_data[:, sample - depth: sample]
        if signal_slice.shape[1] != 0:
            slices.append(signal_slice)
    slices = np.array(slices)
    return slices


def slice_signal_chunks(signal_chunks: np.array, depth: int, step: int,
                        slice_normalization: bool = True, channels: bool = False) -> np.ndarray:
    sliced_chunks = []
    for chunk in signal_chunks:
        slices = []
        chunk_end = chunk.shape[1] if channels else chunk.shape[0]
        for i in range(depth, chunk_end, step):
            signal_slice = None
            if channels:
                signal_slice = chunk[:, i - depth: i].copy()
            else:
                signal_slice = chunk[i - depth: i].copy()
            if slice_normalization:
                min_val = np.min(signal_slice)
                max_val = np.max(signal_slice)
                signal_slice = (signal_slice - min_val) / (max_val - min_val)
            slices.append(signal_slice)
        slices = np.array(slices)
        sliced_chunks.append(slices)
    sliced_chunks = np.array(sliced_chunks)
    return sliced_chunks
