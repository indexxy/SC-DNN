from multiprocessing import Pool, cpu_count
from typing import Union, Sequence
from pathlib import Path
import os

import numpy as np
import cupy as cp

from SC.math import dot_sc
from SC.stochastic import bpe_decode, SCNumber
from SC.error import random_bit_flip
from SC.utils import CacheDir, loadModel
from SC.activation import ACTIVATION_FUNCTIONS


class SCNetwork:
    
    def __init__(
            self, modelPath: Path, precision: int, binarized=False, cache=False
    ):
        self.precision = precision
        self.isBinarized = binarized
        self.__batchNormParams = None
        self.cacheDir = None
        self.__weights = None
        self.__H = None

        weights, biases, self.__batchNormParams = loadModel(modelPath)
        if len(biases) == 0:
            self.useBias = False
        else:
            self.useBias = True

        self.num_layers = len(weights)
        if self.useBias:
            # Appending biases to weights
            for i in range(self.num_layers):
                weights[i] = np.append(weights[i], biases[i], axis=1)

        if binarized:
            self.__H = []
            for i in range(self.num_layers):
                self.__H.append(
                    np.sqrt(1.5 / (weights[i].shape[0] + weights[i].shape[1])).astype(
                    np.float32)
                )
                weights[i] = SCNetwork.__binarize(weights[i])

        if cache:
            self.cacheDir = CacheDir(os.getpid())
            print('Writing weights to ' + str(self.cacheDir) + '\\ ...')
            for i, w in enumerate(weights):
                np.save(
                    os.path.join(self.cacheDir.path, str(i)),
                    SCNumber(w, precision=precision)
                )

        else:
            self.__weights = [SCNumber(w, precision=precision) for w in weights]

        self.__activations = None
        self.__scales = None
        self.__hiddenAf = None
        self.__outAf = None
        self.__faultParams = None
        self.isCompiled = False

    def compile(self, hiddenAf, outAf, hidden_scale=5, out_scale=10):
        assert hiddenAf in ACTIVATION_FUNCTIONS
        assert outAf in ACTIVATION_FUNCTIONS

        self.__hiddenAf = ACTIVATION_FUNCTIONS[hiddenAf]
        self.__outAf = ACTIVATION_FUNCTIONS[outAf]
        self.__activations = [self.__hiddenAf for _ in range(self.num_layers - 1)]
        self.__activations.append(self.__outAf)
        self.__scales = {
            self.__hiddenAf: hidden_scale,
            self.__outAf: out_scale
        }
        self.isCompiled = True

    def getWeights(self, layer):
        assert layer <= self.num_layers

        if self.cacheDir is not None:
            w = np.load(os.path.join(self.cacheDir.path, str(layer) + '.npy'))
            overwrite = True
        else:
            w = self.__weights[layer]
            overwrite = False

        if self.__faultParams is not None:
            rate = self.__faultParams['weights_rates'][layer]
            mode = self.__faultParams['mode']
            w = SCNetwork.__injectFault(w, rate, mode, overwrite=overwrite)

        return w

    def __batchNorm(self, z: np.ndarray, layer: int):
        EPSILON = 1e-2
        mu = self.__batchNormParams["mus"][layer]
        sigma = self.__batchNormParams["sigmas"][layer]
        gamma = self.__batchNormParams["gammas"][layer]
        beta = self.__batchNormParams["betas"][layer]
        ztilde = z - mu
        ztilde /= np.sqrt(sigma + EPSILON)
        ztilde *= gamma
        ztilde += beta

        return ztilde

    def forwardpass(self, x):
        # Expected x shape : (num_instances, num_features, precision)
        assert x.ndim == 3

        activation = x.transpose((1, 0, 2))
        layer = 0  # Layer counter
        for af in self.__activations:
            w = self.getWeights(layer)
            scale = self.__scales[af]

            if self.useBias:
                # Appending ones to the end of each activation layer
                # These ones will be multiplied by the biases appended previously to the weights
                ones = np.ones((1, activation.shape[1], self.precision), dtype=np.bool)
                activation = np.append(activation, ones, axis=0)

            z = dot_sc(w, activation, scale=scale)
            z = bpe_decode(z, precision=self.precision, scale=scale)

            if self.isBinarized:
                z *= self.__H[layer]
            if self.__batchNormParams is not None:
                z = self.__batchNorm(z, layer)

            activation = SCNumber(af(z), precision=self.precision)

            if self.__faultParams is not None:
                rate = self.__faultParams['activations_rates'][layer]
                if rate != 0.0:
                    mode = self.__faultParams['mode']
                    SCNetwork.__injectFault(activation, rate, mode, overwrite=True)

            layer += 1

        return activation

    def predict(self, x):
        # Expected x shape: (num_instances, num_features)
        x = self.___preProcess(x)
        preds = self.forwardpass(x)

        # Predictions shape : (num_instances, num_features)
        return bpe_decode(preds, precision=self.precision).T

    def get_accuracy(self, x, y):
        # Expected x shape: (num_instances, num_features)
        # Expected y shape (num_instances, )
        preds = self.predict(x)
        preds = preds.argmax(axis=1)
        return np.count_nonzero(preds == y)

    def ___preProcess(self, x: np.ndarray):
        # Expected x shape: (num_instances, num_features)
        x = SCNumber(x, precision=self.precision)

        if self.__faultParams is not None:
            ir = self.__faultParams['instances_rate']
            fr = self.__faultParams['features_rate']
            mode = self.__faultParams['mode']
            if ir != 0.0 and fr != 0.0:
                # Number of instances (Images)
                n_instances = round(ir * x.shape[0])

                # Number of bits
                n_bits = round(fr * self.precision * x.shape[1])
                instance_indices = np.random.choice(x.shape[0], n_instances, replace=False)

                tmpX = x[instance_indices, :, :].reshape(-1, self.precision * x.shape[1])
                tmpX = random_bit_flip(tmpX, n_bits=n_bits, mode=mode)
                x[instance_indices] = tmpX.reshape(-1, x.shape[1], self.precision)

        return x

    def evaluate(self, x: np.ndarray, y: np.ndarray, batch_size: int = 50, parallel=True, pLimit = 0):
        # Expected x shape : (num_instances, num_features)
        # Expected y shape : (num_instances, )

        assert batch_size >= 0
        if batch_size == 0:
            batch_size = x.shape[0]

        if not self.isCompiled:
            print('The network needs to be compiled first..')
            return -1

        # In case data were loaded with useGpu=True (although it's not recommended)
        if cp.get_array_module(x) == cp:
            x, y = cp.asnumpy(x), cp.asnumpy(y)

        if parallel:
            if pLimit == 0:
                pLimit = cpu_count()
            return self.__evaluateParallel(x, y, batch_size, pLimit)

        correct = 0
        for i in range(0, x.shape[0], batch_size):
            correct += self.get_accuracy(x[i: i + batch_size], y[i: i + batch_size])

        return correct

    def __evaluateParallel(self, x: np.ndarray, y: np.ndarray, batch_size: int, processLimit):
        processCount = cpu_count()
        batchesCount = x.shape[0] // batch_size
        if batchesCount == 0:
            processCount = 1
        else:
            if processCount > processLimit:
                processCount = processLimit
            if processCount > batchesCount:
                processCount = batchesCount
        pool = Pool(processCount)
        input_args = [
            (
                x[i: i + batch_size],
                y[i: i + batch_size]
            )
            for i in range(0, x.shape[0], batch_size)
        ]
        output = pool.starmap(self.get_accuracy, input_args)
        correct = sum(output)

        pool.close()
        return correct

    def faultyEvaluate(
            self,
            x: np.ndarray,
            y: np.ndarray,
            instances_rate: float = 1.0,
            features_rate: float = 0.00,
            weights_rates: Union[float, Sequence[float]] = 0.00,
            activations_rates: Union[float, Sequence[float]] = 0.00,
            mode: int = 2,
            batch_size: int = 50,
            parallel=True,
    ):
        if not self.isCompiled:
            print('The network needs to be compiled first..')
            return -1

        if isinstance(weights_rates, (float, int)):
            weights_rates = [weights_rates for _ in range(self.num_layers)]
        else:
            assert len(weights_rates) == self.num_layers

        if isinstance(activations_rates, (float, int)):
            activations_rates = [activations_rates for _ in self.__activations]
        else:
            assert len(activations_rates) == len(self.__activations)

        self.__faultParams = dict(
            weights_rates=weights_rates,
            instances_rate=instances_rate,
            features_rate=features_rate,
            activations_rates=activations_rates,
            mode=mode
        )

        correct = self.evaluate(x, y, batch_size=batch_size, parallel=parallel)

        # Restoring the non-faulty parameters
        self.__faultParams = None

        return correct

    @staticmethod
    def __binarize(x: np.ndarray) -> np.ndarray:
        newX = x.copy()
        newX[x >= 0] = 1
        newX[x < 0] = -1

        return newX

    @staticmethod
    def __injectFault(x: np.ndarray, rate: float, mode: int, overwrite: bool):
        assert x.ndim == 3
        if not overwrite:
            x = x.copy()

        n_bits = round(rate * x.size)
        x1d = x.reshape(x.size)

        return random_bit_flip(x1d, n_bits, mode=mode).reshape(x.shape)


    def __del__(self):
        if self.cacheDir is not None:
            try:
                if os is not None:
                    self.cacheDir.remove(os.getpid())
            except:
                print('-' * 15)
                print('WARNING: Failed to delete cache folder ' + str(self.cacheDir) + '\nTry to delete it manually..')
                print('-' * 15 + '\n')
