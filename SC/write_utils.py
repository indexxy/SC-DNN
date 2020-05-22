from SC.stochastic import SCNumber
import numpy as np
from dnn.mlpcode.utils import loadMnist, loadFashionMnist, loadCifar10, loadMnistC
import os
from SC.load_utils import DATASETS, MNIST_C_DIR, FASHION_MNIST_DIR, MNIST_DIR

LOADING_FUNCS = {
    DATASETS.mnist: loadMnist,
    DATASETS.fashion: loadFashionMnist,
    DATASETS.cifar10: loadCifar10,
}


# noinspection PyTypeChecker
# Converts a dataset to stochastic bit-stream and writes it on the disk
# it's written in 'data' folder in the same directory where the file exists
def dataset2SC(dataset: DATASETS, precision):
    val = str(dataset)
    num_instances = 10000

    # todo : add cifar-10 & affNist

    if val.startswith('mnist_c'):
        write_dir = MNIST_C_DIR / val.split("-")[-1]
        _, _, testX, testY = loadMnistC(dataset, useGpu=False)
    else:
        load_func = LOADING_FUNCS[dataset]
        if val.startswith('fashion'):
            write_dir = FASHION_MNIST_DIR
        else:
            write_dir = MNIST_DIR
        _, _, testX, testY = load_func(useGpu=False)

    if not write_dir.is_dir():
        os.mkdir(write_dir)
    write_dir = write_dir / str(precision)

    if not write_dir.is_dir():
        os.mkdir(write_dir)
        os.mkdir(write_dir / 'labels')
        os.mkdir(write_dir / 'images')

    # taking care of oneHot-encoding
    testY = testY.argmax(axis=1)
    testY = testY.reshape(1, -1)

    # (num_instances, num_features) --> (num_features, num_instances)
    # each column represents an instance
    testX = testX.T

    for i in range(0, num_instances, 1000):
        Y = SCNumber(testY[:, i:i + 1000], precision=precision)
        X = SCNumber(testX[:, i:i + 1000], precision=precision)
        np.save(write_dir / 'labels' / str(i), Y)
        np.save(write_dir / 'images' / str(i), X)

        # Temporary
        np.save(write_dir / 'labels' / (str(i) + '_deterministic'), testY[:, i:i + 1000])