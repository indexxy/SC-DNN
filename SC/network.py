from multiprocessing import Pool, cpu_count

import numpy as np

from SC.math import dot_sc
from SC.load_utils import loadDataset, DATASETS
from SC.stochastic import bpe_decode, SCNumber
from dnn.mlpcode.activation import ACTIVATION_FUNCTIONS


class SCNetwork:
    def __init__(
        self, network, hiddenAf, outAf, precision, binarized=False, hidden_scale=1, out_scale=1
    ):

        self.__hiddenAf = ACTIVATION_FUNCTIONS[hiddenAf]
        self.__outAf = ACTIVATION_FUNCTIONS[outAf]
        weights = network.weights.copy()
        biases = network.biases.copy()
        self.num_layers = network.num_layers

        if binarized:
            weights = [SCNetwork.binarize(w) for w in weights]
            biases = [SCNetwork.binarize(b) for b in biases]

        for i in range(self.num_layers):
            weights[i] = np.append(weights[i], biases[i], axis=1)

        self.weights = [SCNumber(wb, precision=precision) for wb in weights]
        self.precision = precision
        self.__activations = [self.__hiddenAf for _ in range(self.num_layers - 1)]
        self.__activations.append(self.__outAf)
        self.scales = {
            self.__hiddenAf: hidden_scale,
            self.__outAf: out_scale
        }

    def forwardpass(self, x):
        a = x
        for w, af in zip(self.weights, self.__activations):
            scale = self.scales[af]
            a = np.append(
                a,
                SCNumber(np.ones((1, a.shape[1])), precision=self.precision),
                axis=0
            )
            z = dot_sc(w, a, scale=scale)
            a = SCNumber(
                af(bpe_decode(z, precision=self.precision, scale=scale)),
                precision=self.precision
            )
        return a

    def get_accuracy(self, x, y):
        preds = self.forwardpass(x)
        preds = bpe_decode(preds, precision=self.precision)
        preds = preds.argmax(axis=0).reshape(1, -1)
        return (preds == y).sum()

    def testDataset(self, dataset: DATASETS, num_instances=10000, parallel=False):
        if parallel:
            return self.__testDatasetParallel(dataset, num_instances)

        batch_size = 50
        overall_correct = 0
        for i in range(0, num_instances, 1000):
            x, y, detY = loadDataset(dataset, precision=self.precision, idx=i)
            for j in range(0, 1000, batch_size):
                overall_correct += self.get_accuracy(x[:, j:j + batch_size], detY[:, j:j + batch_size])

        return overall_correct

    def __testDatasetParallel(self, dataset: DATASETS, num_instances=10000):
        batch_size = 50

        # tmp
        processCount = cpu_count()
        if processCount > 20:
            processCount = 20

        pool = Pool(processes=processCount)
        overall_correct = 0
        for i in range(0, num_instances, 1000):
            x, y, detY = loadDataset(dataset, precision=self.precision, idx=i)
            input_args = [(x[:, j: j + batch_size], detY[:, j:j + batch_size]) for j in range(0, 1000, batch_size)]
            output = pool.starmap(self.get_accuracy, input_args)
            overall_correct += sum(output)

        pool.close()
        return overall_correct

    @staticmethod
    def binarize(x):
        newX = np.empty_like(x, dtype=np.int8)
        newX[x >= 0] = 1
        newX[x < 0] = -1
        return newX