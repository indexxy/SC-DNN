from numbers import Number
import numpy as np

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

# Converts bipolar encoded bit-stream to decimal
def bpe_decode(x: np.ndarray, precision, conc=1):
    result = 0
    i = 0
    if conc > 1:
        tmp = (2 * np.count_nonzero(x[i:precision + i]) - precision) / precision
        while conc != 0 and tmp != 0:
            result += tmp
            i += precision
            conc -= 1
            tmp = (2 * np.count_nonzero(x[i:precision + i]) - precision) / precision

    else:
        result = (2 * np.count_nonzero(x) - precision) / precision

    return result


# vectorized version of bpe_decode()
def vect_bpe_decode(x: np.ndarray, precision, conc=1):
    # x is a 2D numpy array holding bit-streams
    n = len(x)
    result = np.empty(n, dtype=np.float)
    for i in range(n):
        result[i] = bpe_decode(x[i], precision, conc=conc)
    return result


def mat_bpe_decode(x: np.ndarray, precision, conc=1):
    # x is a 3D numpy array holding bit-streams
    # returns 2D float numpy array
    result = np.empty((x.shape[0], x.shape[1]), dtype=np.float)
    for i in range(len(x)):
        result[i] = vect_bpe_decode(x[i], precision, conc=conc)
    return result

# Generates a random bit stream given length and number of ones
def random_stream(length, n_ones) -> np.ndarray:
    stream = np.zeros(length, dtype=np.bool)
    stream[:n_ones] = 1
    np.random.shuffle(stream)
    '''
    # a slower way
    stream[0:]=0
    indices = np.random.choice(np.arange(start=0, stop=length,dtype = np.bool), n_ones, replace=False)
    stream.put(indices,1)
    '''
    return stream


# Converts a decimal to bipolar encoded stochastic bit-stream
def SCNumber(x: Number, min_val=-1, max_val=1, precision=16) -> np.ndarray:
    x = quantize(x, min_val, max_val, precision)
    prob = bpe_encode(x)
    n_ones = int(prob * precision)
    return random_stream(precision, n_ones)


# vectorized version of SCNumber
def vect2SC(x: np.ndarray, min_val=-1, max_val=1, precision=16) -> np.ndarray:
    # x : 1D numpy float array
    # returns 2D numpy bol array that contains bit-streams
    n = len(x)
    result = np.empty((n, precision), dtype=np.bool)
    x = quantize(x, min_val, max_val, precision)
    prob: np.ndarray = bpe_encode(x)
    n_ones = (prob * precision).astype(int)
    for i in range(n):
        result[i] = random_stream(precision, n_ones[i])
    return result


def mat2SC(x: np.ndarray, min_val=-1, max_val=1, precision=16) -> np.ndarray:
    # x : 2D numpy float array
    # returns 3D numpy bool array that contains bit-streams
    n = len(x)
    result = np.empty((x.shape[0], x.shape[1], precision), dtype=np.bool)
    for i in range(n):
        result[i] = vect2SC(x[i], min_val, max_val, precision)
    return result