from enum import Enum

import numpy as np

RELU_EPSILON = 0.01

"""
Inspired by: https://github.com/volf52/deep-neural-net/
"""

def identity(x: np.ndarray):
    return x


def sigmoid(x: np.ndarray):
    return 1.0 / (1.0 + np.exp(-x))


def tanh(x: np.ndarray):
    return np.tanh(x)


# Expected shape : (num_classes, num_instances)
def softmax(x: np.ndarray):
    a = x - x.max(axis=0)[np.newaxis, :]
    np.exp(a, out=a)
    a /= a.sum(axis=0)[np.newaxis, :]

    return a


def relu(x: np.ndarray):
    a = x.copy()
    a[x < 0] = 0

    return a


def leaky_relu(x: np.ndarray):
    a = x.copy()
    a[x < 0] *= RELU_EPSILON

    return a


def unitstep(x: np.ndarray):
    a = x.copy()

    a[x >= 0] = 1
    a[x < 0] = -1

    return a


class ActivationFuncs(Enum):
    sigmoid = "sigmoid"
    softmax = "softmax"
    relu = "relu"
    tanh = "tanh"
    sign = "unitstep"
    leaky_relu = "leaky_relu"
    identity = "identity"

    def __repr__(self):
        return self.value

    def __str__(self):
        return self.value


ACTIVATION_FUNCTIONS = {
    ActivationFuncs.sigmoid: sigmoid,
    ActivationFuncs.softmax: softmax,
    ActivationFuncs.relu: relu,
    ActivationFuncs.tanh: tanh,
    ActivationFuncs.sign: unitstep,
    ActivationFuncs.leaky_relu: leaky_relu,
    ActivationFuncs.identity: identity,
}