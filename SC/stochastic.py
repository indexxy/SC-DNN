from numbers import Number
import numpy as np
from numba import guvectorize
from typing import Union


# To quantize a float or an array of floats
# returns : float or an array of floats
def quantize(x, min_val=-1, max_val=1, precision=16):
    if isinstance(x, np.ndarray):
        x[x <= min_val] = min_val
        x[x >= max_val] = max_val
        q = (max_val - min_val) / precision
        return q * np.round(x / q)

    if x >= max_val:
        return max_val
    elif x <= min_val:
        return min_val
    else:
        q = (max_val - min_val) / precision
        return q * np.round(x / q)


# Calculates the Bipolar encoding probability of a parameter x
def bpe_encode(x):
    # x is expected to be quantized
    return (x + 1) / 2

# Converts bipolar encoded bit-stream(s) to decimal(s)
def bpe_decode(x: np.ndarray, precision, scale=1):
    result = np.empty((x.shape[:-1]), dtype=np.float)
    _bpe_decode(x, precision, scale, result)
    if len(result.shape) == 0:
        return np.float(result)
    return result


@guvectorize(
    ['void(boolean[:], int64, int64, double[:])'],
    '(n),(),()->()'
)
def _bpe_decode(x: np.ndarray, precision, scale, out):
    result = 0
    i = 0
    if scale > 1:
        tmp = (2 * np.count_nonzero(x[i:precision + i]) - precision) / precision
        while scale != 0 and tmp != 0:
            result += tmp
            i += precision
            scale -= 1
            tmp = (2 * np.count_nonzero(x[i:precision + i]) - precision) / precision

    else:
        result = (2 * np.count_nonzero(x) - precision) / precision

    out[0] = result


# Generates a random bit stream given length and number of ones
def random_stream(length, n_ones) -> np.ndarray:
    stream = np.zeros(length, dtype=np.bool)
    stream[:n_ones] = 1
    np.random.shuffle(stream)

    return stream


@guvectorize(
    ['void(double, double, double, double, boolean[:], boolean[:])'],
    '(),(),(),(),(n)->(n)'
)
def _SCNumber(x, min_val, max_val, precision, dummy, stream: np.ndarray):
    # Quantized value
    if x < min_val:
        quantX = min_val
    elif x > max_val:
        quantX = max_val
    else:
        q = (max_val - min_val) / precision
        quantX = q * np.round(x / q)

    prob = (quantX+1)/2  # BPE Encoding probability
    n_ones = int(prob * precision)
    stream[:n_ones] = 1
    np.random.shuffle(stream)


# Converts a decimal/array of decimals to bipolar encoded stochastic bit-stream/array of bit-streams
def SCNumber(x: Union[Number, np.ndarray], min_val=-1, max_val=1, precision=16) -> np.ndarray:
    if isinstance(x, np.ndarray):
        stream = np.zeros(x.shape + (precision,), dtype=np.bool)
        _SCNumber(x, min_val, max_val, precision, stream, stream)
        return stream

    x = quantize(x, min_val, max_val, precision)
    prob = bpe_encode(x)
    n_ones = int(prob * precision)
    return random_stream(precision, n_ones)