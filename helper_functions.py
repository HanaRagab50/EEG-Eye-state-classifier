from cProfile import label
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from scipy.fft import rfft, rfftfreq
from collections import Counter

def energy_feature(signal):
    signal = np.asarray(signal)
    return np.sum(signal**2)

def rms_feature(signal):
    signal = np.asarray(signal)
    return np.sqrt(np.mean(signal**2))

def peak_feature(signal):
    signal = np.asarray(signal)
    return np.max(np.abs(signal))

def amplitude_feature(signal):
    signal = np.asarray(signal)
    return np.ptp(signal)

def skewness_feature(signal):
    signal = np.asarray(signal)
    return skew(signal)

def kurtosis_feature(signal):
    signal = np.asarray(signal)
    return kurtosis(signal)

def crest_factor_feature(signal):
    rms_val = rms_feature(signal)
    p = peak_feature(signal)
    return p / rms_val

def shape_factor_feature(signal):
    signal = np.asarray(signal)
    mean_abs_val = np.mean(np.abs(signal))
    r = rms_feature(signal)
    return r / mean_abs_val

def impulse_factor_feature(signal):
    signal = np.asarray(signal)
    mean_abs_val = np.mean(np.abs(signal))
    p = peak_feature(signal)
    return p / mean_abs_val

def clearance_factor_feature(signal):
    signal = np.asarray(signal)
    sqrt_mean_sqrt_abs = np.mean(np.sqrt(np.abs(signal)))**2
    p = peak_feature(signal)
    return p / sqrt_mean_sqrt_abs

def shannon_entropy_feature(signal, bins=10):
    signal = np.asarray(signal)
    hist, bin_edges = np.histogram(signal, bins=bins, density=False)
    prob = hist.astype(float) / np.sum(hist)
    prob = prob[prob > 0]
    return entropy(prob)

def zcr_feature(signal):
    signal = np.asarray(signal)
    n = len(signal)
    s = np.sign(signal)
    s[s == 0] = 1
    zero_crossings = np.sum(s[:-1] != s[1:])
    return zero_crossings / (n - 1)

def lempel_ziv_complexity(signal):

    x = np.asarray(signal)
    n = len(x)
    med = np.median(x)
    b = (x >= med).astype(int)
    s = ''.join(b.astype(str))
    i = 0
    c = 1
    l = 1
    while True:
        if i + l > n:
            break
        sub = s[i:i+l]
        found = False
        # search sub in previous part
        for j in range(0, i):
            if s[j:j+l] == sub:
                found = True
                break
        if not found:
            c += 1
            i += l
            l = 1
        else:
            l += 1
        if i + l > n:
            break
    return c


def katz_fractal_dimension(signal):

    # Katz fractal dimension: FD = log10(n) / (log10(n) + log10(d/L))

    x = np.asarray(signal).astype(float)
    n = len(x)
    diffs = np.abs(np.diff(x))
    L = np.sum(diffs)
    # distances from first point
    dists = np.abs(x - x[0])
    d = np.max(dists)
    if L == 0 or d == 0:
        return np.nan
    return np.log10(n) / (np.log10(n) + np.log10(d / L))

# def extract_features(signal):
#     features = {
#         'Mean':signal.mean(),
#         'Std':signal.std(),
#         'var':signal.std()**2,
#         # 'Energy': energy_feature(signal),
#         'RMS': rms_feature(signal),
#         'Peak': peak_feature(signal),
#         # 'Amplitude': amplitude_feature(signal),
#         'Skewness': skewness_feature(signal),
#         'Kurtosis': kurtosis_feature(signal),
#         'Crest Factor': crest_factor_feature(signal),
#         # 'Shape Factor': shape_factor_feature(signal),
#         # 'Impulse Factor': impulse_factor_feature(signal),
#         # 'Clearance Factor': clearance_factor_feature(signal),
#         'Shannon Entropy': shannon_entropy_feature(signal),
#         'Zero-Crossing Rate': zcr_feature(signal),
#         # 'Lempel-Ziv Complexity': lempel_ziv_complexity(signal),
#         # 'Fractal Dimension (Katz)': katz_fractal_dimension(signal)
#     }
    
#     return features


# def spectral_features(signal, sfreq):
#     fft_vals = np.abs(rfft(signal)) ** 2
#     freqs = rfftfreq(len(signal), 1/sfreq)

#     bands = {
#         "delta": (0.5, 4),
#         "theta": (4, 8),
#         "alpha": (8, 13),
#         "beta": (13, 30),
#         "gamma": (30, 50)
#     }

#     total_power = np.sum(fft_vals)
#     features = {}

#     for band, (low, high) in bands.items():
#         idx = np.logical_and(freqs >= low, freqs <= high)
#         band_power = np.sum(fft_vals[idx])

#         features[f"{band}_abs"] = band_power
#         features[f"{band}_rel"] = band_power / total_power

#     features["spectral_centroid"] = np.sum(freqs * fft_vals) / total_power
    
#     psd_norm = fft_vals / total_power
#     features["spectral_entropy"] = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))

#     return features

def features(signal, sfreq=128):
    signal = np.asarray(signal)

    # ---------- FFT ----------
    fft_vals = np.abs(rfft(signal)) ** 2
    freqs = rfftfreq(len(signal), 1 / sfreq)

    total_power = np.sum(fft_vals) + 1e-12  # avoid division by zero

    # ---------- Frequency Bands ----------
    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 50)
    }

    features = {}

    for band, (low, high) in bands.items():
        idx = (freqs >= low) & (freqs <= high)
        band_power = np.sum(fft_vals[idx])

        features[f"{band}_abs"] = band_power
        features[f"{band}_rel"] = band_power / total_power

    # ---------- Spectral Features ----------
    features["spectral_centroid"] = np.sum(freqs * fft_vals) / total_power

    psd_norm = fft_vals / total_power
    features["spectral_entropy"] = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))

    # ---------- Time-Domain Features ----------
    features["mean"] = signal.mean()
    features["std"] = signal.std()
    features["var"] = signal.var()
    features["rms"] = rms_feature(signal)
    features["peak"] = peak_feature(signal)
    features["skewness"] = skewness_feature(signal)
    features["kurtosis"] = kurtosis_feature(signal)
    features["crest_factor"] = crest_factor_feature(signal)
    features["shannon_entropy"] = shannon_entropy_feature(signal)
    features["zcr"] = zcr_feature(signal)

    return features

def create_windows(X,y,window_size,overlap,sfreq=128):
    window_samples=int(window_size*sfreq)
    step=int((window_samples)*(1-overlap))
    n_samples=len(X)
    windows=[]
    labels=[]
    start=0

    while start +(window_samples)<=n_samples:
        end= start+window_samples
        window=X.iloc[start:end]
        label_window=y.iloc[start:end]

        majority_label = Counter(label_window).most_common(1)[0][0]

        windows.append(window)
        labels.append(majority_label)

        start += step
    return np.array(windows),np.array(labels)

# - delta_rel
# - theta_rel
# - alpha_rel
# - beta_rel
# - gamma_rel
# - rms
# - peak
# - kurtosis
# - crest_factor
def channel_features(signal ,sfreq=128):
    signal = np.asarray(signal)

    # ---------- FFT ----------
    fft_vals = np.abs(rfft(signal)) ** 2
    freqs = rfftfreq(len(signal), 1 / sfreq)

    total_power = np.sum(fft_vals) + 1e-12  # avoid division by zero

    # ---------- Frequency Bands ----------
    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 50)
    }

    features = {}

    for band, (low, high) in bands.items():
        idx = (freqs >= low) & (freqs <= high)
        band_power = np.sum(fft_vals[idx])

        features[f"{band}_abs"] = band_power
        # features[f"{band}_rel"] = band_power / total_power

    # ---------- Spectral Features ----------
    features["spectral_centroid"] = np.sum(freqs * fft_vals) / total_power

    psd_norm = fft_vals / total_power
    features["spectral_entropy"] = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))

    # ---------- Time-Domain Features ----------
    features["mean"] = signal.mean()
    features["std"] = signal.std()
    features["var"] = signal.var()
    # features["rms"] = rms_feature(signal)
    # features["peak"] = peak_feature(signal)
    features["skewness"] = skewness_feature(signal)
    # features["kurtosis"] = kurtosis_feature(signal)
    # features["crest_factor"] = crest_factor_feature(signal)
    features["shannon_entropy"] = shannon_entropy_feature(signal)
    features["zcr"] = zcr_feature(signal)

    return features

def window_features(window, channel_names, sfreq=128):
    """
    window: shape (window_length, n_channels)
    """
    features = {}

    for ch_idx, ch_name in enumerate(channel_names):
        ch_signal = window[:, ch_idx]
        ch_features = channel_features(ch_signal, sfreq)

        for feat_name, value in ch_features.items():
            features[f"{ch_name}_{feat_name}"] = value

    return features

def features_all_windows(X_windows, channel_names, sfreq=128):
    feature_list = []

    for window in X_windows:
        feature_dict = window_features(window, channel_names, sfreq)
        feature_list.append(feature_dict)

    return pd.DataFrame(feature_list)

