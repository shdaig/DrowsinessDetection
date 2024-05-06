import mne
import numpy as np
import scipy.signal as signal


def read_fif(filename: str) -> mne.io.Raw:
    raw = mne.io.read_raw_fif(filename, preload=True, verbose=False)
    raw = raw.pick('eeg', verbose=False)
    raw = raw.set_eeg_reference(ref_channels='average', verbose=False)

    return raw


def fetch_channels(raw: mne.io.Raw) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :param raw: Raw data from fif file
    :return: Time samples, channel names, channel data
    """
    return raw.times, np.array(raw.ch_names), raw.get_data()


def bandpass_filter(data, lowcut, highcut, signal_freq, filter_order) -> np.ndarray:
    nyquist_freq = 0.5 * signal_freq
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b, a = signal.butter(filter_order, [low, high], btype="band")
    y = signal.lfilter(b, a, data)
    return y


def get_frequency_features(channel_names: np.ndarray, channel_data: np.ndarray, times: np.ndarray):
    filter_channels = ['C3', 'C4', 'P3', 'P4', 'Pz', 'Cz', 'T3', 'T4', 'O1', 'O2']
    eeg_bands = {'Delta': (0, 4), 'Theta': (4, 8), 'Alpha': (8, 12), 'Beta': (12, 30), 'Gamma': (30, 100)}
    fft_times = np.insert(times, 0, 0)

    eeg_band_fft_list = []

    for channel in filter_channels:
        if channel in channel_names:
            signal = channel_data[channel_names == channel][0]
            eeg_band_fft = {band: [] for band in eeg_bands}
            for t in range(len(fft_times) - 1):
                fft_vals = np.absolute(np.fft.rfft(signal[fft_times[t]:fft_times[t + 1]]))
                for band in eeg_bands:
                    fft_freq = np.fft.rfftfreq(len(fft_vals), 1.0 / 500)
                    freq_ix = np.where((fft_freq >= eeg_bands[band][0]) & (fft_freq <= eeg_bands[band][1]))[0]
                    eeg_band_fft[band].append(np.mean(fft_vals[freq_ix]))

            eeg_band_fft_list.append(eeg_band_fft)

    # mean or max
    eeg_band_fft_mean = {band: np.mean([fft_list[band] for fft_list in eeg_band_fft_list], axis=0)
                         for band in eeg_bands}

    eeg_band_fft_mean['Delta/Theta'] = eeg_band_fft_mean['Delta'] / eeg_band_fft_mean['Theta'] # 6
    eeg_band_fft_mean['Delta/Alpha'] = eeg_band_fft_mean['Delta'] / eeg_band_fft_mean['Alpha'] # 7
    eeg_band_fft_mean['Delta/Beta'] = eeg_band_fft_mean['Delta'] / eeg_band_fft_mean['Beta'] # 8
    eeg_band_fft_mean['Delta/Gamma'] = eeg_band_fft_mean['Delta'] / eeg_band_fft_mean['Gamma'] # 9

    eeg_band_fft_mean['Theta/Delta'] = eeg_band_fft_mean['Theta'] / eeg_band_fft_mean['Delta'] # 10
    eeg_band_fft_mean['Theta/Alpha'] = eeg_band_fft_mean['Theta'] / eeg_band_fft_mean['Alpha'] # 11
    eeg_band_fft_mean['Theta/Beta'] = eeg_band_fft_mean['Theta'] / eeg_band_fft_mean['Beta'] # 12
    eeg_band_fft_mean['Theta/Gamma'] = eeg_band_fft_mean['Theta'] / eeg_band_fft_mean['Gamma'] # 13

    eeg_band_fft_mean['Alpha/Delta'] = eeg_band_fft_mean['Alpha'] / eeg_band_fft_mean['Delta'] # 14
    eeg_band_fft_mean['Alpha/Theta'] = eeg_band_fft_mean['Alpha'] / eeg_band_fft_mean['Theta'] # 15
    eeg_band_fft_mean['Alpha/Beta'] = eeg_band_fft_mean['Alpha'] / eeg_band_fft_mean['Beta'] # 16
    eeg_band_fft_mean['Alpha/Gamma'] = eeg_band_fft_mean['Alpha'] / eeg_band_fft_mean['Gamma'] # 17

    eeg_band_fft_mean['Beta/Delta'] = eeg_band_fft_mean['Beta'] / eeg_band_fft_mean['Delta'] # 18
    eeg_band_fft_mean['Beta/Theta'] = eeg_band_fft_mean['Beta'] / eeg_band_fft_mean['Theta'] # 19
    eeg_band_fft_mean['Beta/Alpha'] = eeg_band_fft_mean['Beta'] / eeg_band_fft_mean['Alpha'] # 20
    eeg_band_fft_mean['Beta/Gamma'] = eeg_band_fft_mean['Beta'] / eeg_band_fft_mean['Gamma'] # 21

    eeg_band_fft_mean['Gamma/Delta'] = eeg_band_fft_mean['Gamma'] / eeg_band_fft_mean['Delta'] # 22
    eeg_band_fft_mean['Gamma/Theta'] = eeg_band_fft_mean['Gamma'] / eeg_band_fft_mean['Theta'] # 23
    eeg_band_fft_mean['Gamma/Alpha'] = eeg_band_fft_mean['Gamma'] / eeg_band_fft_mean['Alpha'] # 24
    eeg_band_fft_mean['Gamma/Beta'] = eeg_band_fft_mean['Gamma'] / eeg_band_fft_mean['Beta'] # 25

    return eeg_band_fft_mean

