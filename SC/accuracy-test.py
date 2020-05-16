import matplotlib.pyplot as plt
import numpy as np

import SC.utils as SC
from SC.utils import quantize, bpe_decode, SCNumber
from SC.math import mul, dot, counter_add

'''
Comparison between SC and Deterministic arithmetic with random float numbers.
I know the code looks like a mess, but it serves its purpose lol.
'''

precision = 1024
n = 0  # number of floats
m = 10  # number of passes
predicted = np.ndarray(n, dtype=np.float)
floats = np.arange(-1, 1 + 2 / precision, step=2 / precision)
for f in range(n):
    x = np.random.choice(floats, 1)
    y = np.random.choice(floats, 1)
    real = np.ndarray(m, dtype=np.float)
    predicted = np.ndarray(m, dtype=np.float)
    quantized = np.ndarray(m, dtype=np.float)
    real[0:] = x * y
    quantized[0:] = quantize(x * y, precision=precision)
    for i in range(m):
        predicted[i] = bpe_decode(mul(SCNumber(x[0], precision=precision), SCNumber(y[0], precision=precision)) ,precision=precision)
    plt.title(str(x[0]) + ' x ' + str(y[0]) + ' SC vs Deterministic comparison')
    plt.plot(real, 'g-', label='Real')
    plt.plot(predicted, 'r.', label='Stochastic')
    plt.plot(quantized, 'bx', label='Quantized')
    plt.grid()
    plt.xlabel('Try')
    plt.ylabel('Result')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

for f in range(n):
    x = np.random.choice(floats, 1)
    y = np.random.choice(floats, 1)
    real = np.ndarray(m, dtype=np.float)
    predicted = np.ndarray(m, dtype=np.float)
    quantized = np.ndarray(m, dtype=np.float)
    real[0:] = x + y
    quantized[0:] = quantize(x + y, precision=precision)
    for i in range(m):
        predicted[i] = bpe_decode(counter_add(SCNumber(x[0], precision=precision), SCNumber(y[0], precision=precision)), precision=precision)
    plt.title(str(x[0]) + ' + ' + str(y[0]) + ' SC vs Deterministic comparison')
    plt.plot(real, 'g-', label='Real')
    plt.plot(predicted, 'r.', label='Predicted')
    plt.plot(quantized, 'bx', label='Quantized')
    plt.grid()
    plt.xlabel('Try')
    plt.ylabel('Result')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


# Testing the absolute difference between np.dot and SC.dot
y_float = -2 * np.random.random((4, 4)) + 1
z_float = -2 * np.random.random((4, 4)) + 1
y = SC.mat2SC(y_float, precision=precision)
z = SC.mat2SC(z_float, precision=precision)
real_result = np.dot(y_float, z_float)
quantized_result = np.dot(SC.mat_bpe_decode(y, precision=precision), SC.mat_bpe_decode(z, precision=precision))
result = SC.mat_bpe_decode(dot(y, z, conc=3), precision=precision, conc=3)
print(np.abs(result - quantized_result))
