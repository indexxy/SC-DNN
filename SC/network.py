from multiprocessing import Pool, cpu_count
from typing import Union, Sequence
from pathlib import Path
import os

import numpy as np
import cupy as cp

from SC.math import dot_sc
from SC.stochastic import bpe_decode, SCNumber
from SC.error import random_bit_flip
from SC.utils import CacheDir, loadHDF5
from SC.activation import ACTIVATION_FUNCTIONS


class SCNetwork:
    
    def __init__(
            self, modelPath: Path, precision: int, useBias: bool, binarized=False, cache=False
    ):
        self.useBias = useBias
        self.precision = precision
        self.cacheDir = None
        self.__weights = None
        self.isBinarized = binarized

        weights, biases = loadHDF5(modelPath, useBias)

        self.num_layers = len(weights)
        if self.useBias:
            # Appending biases to weights
            for i in range(self.num_layers):
                weights[i] = np.append(weights[i], biases[i], axis=1)
        if binarized:
            weights = [SCNetwork.__binarize(w) for w in weights]

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
        self.__hiddenAf = ACTIVATION_FUNCTIONS[hiddenAf]
        self.__outAf = ACTIVATION_FUNCTIONS[outAf]
        self.__activations = [self.__hiddenAf for _ in range(self.num_layers - 1)]
        self.__activations.append(self.__outAf)
        self.__scales = {
            self.__hiddenAf: hidden_scale,
            self.__outAf: out_scale
        }
        self.isCompiled = True

    def __getWeights(self, layer):
        assert layer <= self.num_layers

        if self.cacheDir is not None:
            w = np.load(os.path.join(self.cacheDir.path, str(layer) + '.npy'))
            overwrite = True
        else:
            w = self.__weights[layer]
            overwrite = False

        if self.__faultParams is not None:
            n_bits = self.__faultParams['n_bits']
            rate = self.__faultParams['weights_rates'][layer]
            mode = self.__faultParams['mode']
            w = SCNetwork.__injectFault(w, n_bits, rate, mode, overwrite=overwrite)

        return w

    def forwardpass(self, x):
        # Expected x shape : (num_instances, num_features, precision)
        assert len(x.shape) == 3

        activation = x.transpose(1, 0, 2)
        layer = 0  # Layer counter
        for af in self.__activations:

            w = self.__getWeights(layer)
            scale = self.__scales[af]

            if self.useBias:
                # Appending ones to the end of each activation layer
                # These ones will be multiplied by the biases appended previously to the weights
                ones = np.ones((1, activation.shape[1], self.precision), dtype=np.bool)
                activation = np.append(activation, ones, axis=0)

            z = dot_sc(w, activation, scale=scale)
            z = bpe_decode(z, precision=self.precision, scale=scale)
            activation = SCNumber(af(z), precision=self.precision)

            if self.__faultParams is not None:
                self.__injectFault(
                    activation,
                    self.__faultParams['n_bits'],
                    self.__faultParams['activations_rates'][layer],
                    self.__faultParams['mode'],
                    overwrite=True
                )
            layer += 1

        return activation

    def predict(self, x):
        # Expected x shape: (num_instances, num_features)
        x = self.___preProcess(x)
        preds = self.forwardpass(x)
        return bpe_decode(preds, precision=self.precision)

    def get_accuracy(self, x, y):
        # Expected x shape: (num_instances, num_features)
        # Expected y shape (num_instances, )
        y = y.reshape(-1, 1)
        preds = self.predict(x)
        preds = preds.argmax(axis=0).reshape(-1, 1)
        return np.count_nonzero(preds == y)

    def ___preProcess(self, x: np.ndarray):
        # Expected x shape: (num_instances, num_features)
        x = SCNumber(x, precision=self.precision)
        if self.__faultParams is not None:
            # Number of instances (Images)
            n_instances = int(np.round(self.__faultParams['instances_rate'] * x.shape[0]))
            # Number of features (Pixels)
            n_features = int(np.round(self.__faultParams['features_rate'] * x.shape[1]))
            instance_indices = np.random.choice(x.shape[0], n_instances, replace=False)
            for i in instance_indices:
                feature_indices = np.random.choice(x.shape[1], n_features, replace=False)
                x[i, feature_indices] = random_bit_flip(
                    x[i, feature_indices], self.__faultParams['n_bits'], self.__faultParams['mode']
                )

        return x

    def evaluate(self, x: np.ndarray, y: np.ndarray, batch_size: int = 50, parallel=False, pLimit = 0):
        # Expected x shape : (num_instances, num_features)
        # Expected y shape : (num_instances, )

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
            instances_rate: float,
            features_rate: float,
            weights_rates: Union[float, Sequence[float]],
            n_bits: int,
            activations_rates: Union[float, Sequence[float]] = 1.00,
            mode: int = 2,
            batch_size: int = 50,
            parallel=False,
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
            n_bits=n_bits,
            activations_rates=activations_rates,
            mode=mode
        )

        correct = self.evaluate(x, y, batch_size=batch_size, parallel=parallel)

        # Restoring the non-faulty parameters
        self.__faultParams = None

        return correct

    @staticmethod
    def __binarize(x: np.ndarray, H=1.0) -> np.ndarray:
        newX = x.copy()
        newX[x >= 0] = 1
        newX[x < 0] = -1
        newX *= H

        return newX

    @staticmethod
    def __injectFault(x: np.ndarray, n_bits: int, rate: float, mode: int, overwrite: bool):
        assert len(x.shape) == 3
        if not overwrite:
            x = x.copy()

        # Number of elements to be manipulated
        n = int(np.round(rate * x.size // x.shape[-1]))
        # Generating random indices
        indices = np.random.choice((x.size // x.shape[-1]), n, replace=False)
        # Remapping
        rows, columns = np.divmod(indices, x.shape[1])
        x[rows, columns] = random_bit_flip(x[rows, columns], n_bits, mode=mode)

        return x

    def __del__(self):
        if self.cacheDir is not None:
            try:
                if os is not None:
                    self.cacheDir.remove(os.getpid())
            except:
                print('-----------')
                print('WARNING: Failed to delete cache folder ' + str(self.cacheDir) + '\nTry to delete it manually..')
                print('-----------\n')
