import numpy as np
from numba import guvectorize

@guvectorize(
    ['void(boolean[:], int64, int64, boolean[:])'],
    '(n),(),()->(n)'
)
def _random_bit_flip(x: np.ndarray, n_bits, mode, newX: np.ndarray):
    if mode == 0:
        # cond : Condition
        cond = np.where(x == 0)[0]
        if cond.shape[0] < n_bits:
            n_bits = cond.shape[0]
        indices = np.random.choice(cond, n_bits, replace=False)
    elif mode == 1:
        cond = np.where(x == 1)[0]
        if cond.shape[0] < n_bits:
            n_bits = cond.shape[0]
        indices = np.random.choice(cond, n_bits, replace=False)
    else:
        indices = np.random.choice(x.shape[0], n_bits, replace=False)

    newX[indices] = ~x[indices]


# Modes :
#   0 : Flip zeros to ones
#   1 : Flip ones to zeros
#   2 : Hybrid mode (flip ones to zeros and zeros to ones)
def random_bit_flip(x: np.ndarray, n_bits, mode=2):
    if n_bits > x.shape[-1]:
        raise Exception('Number of bits to be flipped should not exceed the total number of bits')
    newX = x.copy()
    _random_bit_flip(x, n_bits, mode, newX)

    return newX
