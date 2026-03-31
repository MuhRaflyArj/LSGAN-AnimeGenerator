import numpy as np


def denorm(x):
    return (x + 1.0) / 2.0


def smooth_curve(values, window=25):
    if len(values) < 2:
        return np.array(values, dtype=np.float32)
    window = max(2, min(window, len(values)))
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(np.array(values, dtype=np.float32), kernel, mode="valid")
