from multiprocessing import Pool, cpu_count

import numpy as np
import cupy as cp

from SC.math import dot_sc
from SC.stochastic import bpe_decode, SCNumber
from dnn.mlpcode.activation import ACTIVATION_FUNCTIONS


class SCNetwork:
    def __init__(
        self, network, hiddenAf, outAf, precision, binarized=False, hidden_scale=1, out_scale=1
    ):
        self.__hiddenAf = ACTIVATION_FUNCTIONS[hiddenAf]
        self.__outAf = ACTIVATION_FUNCTIONS[outAf]

        # In case the Network model was loaded with useGpu=True (although it's not recommended)
        if network.xp == cp:
            weights = [cp.asnumpy(w) for w in network.weights]
            biases = [cp.asnumpy(w) for w in network.biases]
        else:
            weights = network.weights.copy()
            biases = network.biases.copy()

        self.num_layers = network.num_layers

        if binarized:
            weights = [SCNetwork.binarize(w) for w in weights]
            biases = [SCNetwork.binarize(b) for b in biases]

        # Appending biases to weights
        for i in range(self.num_layers):
            weights[i] = np.append(weights[i], biases[i], axis=1)

        self.weights = [SCNumber(wb, precision=precision) for wb in weights]
        self.precision = precision
        self.__activations = [self.__hiddenAf for _ in range(self.num_layers - 1)]
        self.__activations.append(self.__outAf)
        self.__scales = {
            self.__hiddenAf: hidden_scale,
            self.__outAf: out_scale
        }

    def forwardpass(self, x):
        # Expected shape for x : (num_instances, num_features, precision)

        a = x.transpose(1, 0, 2)
        for w, af in zip(self.weights, self.__activations):
            scale = self.__scales[af]
            # Appending ones to the end of each activation layer
            # These ones will be multiplied by the biases appended previously to the weights
            a = np.append(
                a,
                SCNumber(np.ones((1, a.shape[1])), precision=self.precision),
                axis=0
            )
            z = dot_sc(w, a, scale=scale)
            z = bpe_decode(z, precision=self.precision, scale=scale)
            a = SCNumber(af(z), precision=self.precision)
        return a

    def get_accuracy(self, x, y):
        preds = self.forwardpass(x)
        preds = bpe_decode(preds, precision=self.precision)
        preds = preds.argmax(axis=0).reshape(-1, 1)
        return np.count_nonzero(preds == y)

    def evaluate(self, x: np.ndarray, y: np.ndarray, batch_size: int = 50, parallel=False):
        # In case data were loaded with useGpu=True (although it's not recommended)
        if cp.get_array_module(x) == cp:
            x, y = cp.asnumpy(x), cp.asnumpy(y)

        if parallel:
            return self.__evaluateParallel(x, y, batch_size)

        overall_correct = 0
        for i in range(0, x.shape[0], batch_size):
            sX = SCNumber(x[i:i+batch_size], precision=self.precision)
            overall_correct += self.get_accuracy(sX,  y[i:i+batch_size])

        return overall_correct

    def __evaluateParallel(self, x: np.ndarray, y: np.ndarray, batch_size: int):
        processCount = cpu_count()
        num_batches = x.shape[0] // batch_size
        if processCount > num_batches:
            processCount = num_batches

        pool = Pool(processCount)
        input_args = [
            (
                SCNumber(x[i: i + batch_size], precision=self.precision),
                y[i:i + batch_size]
            )
            for i in range(0, x.shape[0], batch_size)
        ]
        output = pool.starmap(self.get_accuracy, input_args)
        overall_correct = sum(output)
        pool.close()

        return overall_correct

    @staticmethod
    def binarize(x):
        newX = np.empty_like(x, dtype=np.int8)
        newX[x >= 0] = 1
        newX[x < 0] = -1
        return newX