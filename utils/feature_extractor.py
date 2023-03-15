import sys
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
    print(sys.path[0])
add_path('/home/bebin.huang/Code/emotion_recognition/code')
import numpy as np
import math
from scipy.signal import butter, lfilter
from datasets import config
slide_length = 0.5 # sliding length for differential entropy extraction

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data.T) # channel in dim 0
    return y # [chan, L]


def compute_DE(signal: np.array):
    variance = np.var(signal, ddof=1, axis=2)
    return np.log(2 * math.pi * math.e * variance) / 2

def feature_extractor(data: np.array):
    # breakpoint()
    window_length, channels = data.shape
    fs = config['fs']['eeg']
    delta = butter_bandpass_filter(data, 1, 4, fs, order=3)
    theta = butter_bandpass_filter(data, 4, 8, fs, order=3)
    alpha = butter_bandpass_filter(data, 8, 14, fs, order=3)
    beta = butter_bandpass_filter(data, 14, 31, fs, order=3)
    gamma = butter_bandpass_filter(data, 31, 51, fs, order=3)
    all_bands = np.stack([delta, theta, alpha, beta, gamma], axis=1) # [32, 5, T]

    slide_points = int(slide_length * fs)
    assert window_length > slide_points
    steps = window_length // slide_points # 2*T
    features = []
    for step in range(steps):
        feat = compute_DE(all_bands[:, :, step*slide_points:(step+1)*slide_points])
        features.append(feat)
    # breakpoint()
    features = np.stack(features, axis=-1)
    return features # [num_chans, num_bands, 2*T]

def feature_to_4D(feat: np.array):
    assert len(feat.shape) == 3
    num_chans, num_bands, L = feat.shape
    sign = [30, 16, 'mean', 12, 23, 
            18, 15, 13,     11, 9,
            29, 28, 'mean', 22, 21,
            17, 14, 2,      10, 8,
            27, 26, 'mean', 25, 24,
            0,  4, 3,       1, 5,
            'pad', 31, 'mean', 20, 'pad',
            'pad', 6, 19, 7, 'pad']
    # breakpoint()
    features = []
    for i, s in enumerate(sign):
        if s == 'mean':
            features.append(feat[[sign[i-1], sign[i+1]]].mean(axis=0))
        elif s == 'pad':
            features.append(np.zeros((num_bands, L)))
        else:
            features.append(feat[s, :, :])
    # breakpoint()
    features = np.stack(features, axis=0)
    features = features.reshape([8, 5, num_bands, L])
    return features # ([h, w, d, L])

if __name__ == "__main__":
    data = np.random.rand(2000, 32)
    features = feature_extractor(data)
    features = feature_to_4D(features)
    print(features.shape)
