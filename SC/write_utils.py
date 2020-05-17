from SC.stochastic import mat2SC
import numpy as np
import dnn.mlpcode.utils as mlputils
import os

from SC.load_utils import DATASETS, MNIST_C_DIR, FASHION_MNIST_DIR, MNIST_DIR

# noinspection PyTypeChecker
# Converts a dataset to stochastic bit-stream and writes it on the disk
# it's written in 'data' folder in the same directory where the file exists
def dataset2SC(dataset: DATASETS, precision):
    val = str(dataset)
    num_instances = 10000

    # todo : add cifar-10 & affNist

    if val.startswith('mnist_c'):
        write_dir = MNIST_C_DIR / val.split("-")[-1]
    else:
        if val.startswith('fashion'):
            write_dir = FASHION_MNIST_DIR
        else:
            write_dir = MNIST_DIR

    if not write_dir.is_dir():
        os.mkdir(write_dir)
    write_dir = write_dir / str(precision)

    if not write_dir.is_dir():
        os.mkdir(write_dir)
        os.mkdir(write_dir / 'labels')
        os.mkdir(write_dir / 'images')

    _, _, testX, testY = mlputils.loadDataset(dataset, useGpu=False)   # Redundant operation

    # taking care of oneHot-encoding
    testY = testY.argmax(axis=1)
    testY = testY.reshape(1, -1)

    # (num_instances, num_features) --> (num_features, num_instances)
    # each column represents an instance
    testX = testX.T

    for i in range(0, num_instances, 1000):
        Y = mat2SC(testY[:, i:i + 1000], precision=precision)
        X = mat2SC(testX[:, i:i + 1000], precision=precision)
        np.save(write_dir / 'labels' / str(i), Y)
        np.save(write_dir / 'images' / str(i), X)

        # Temporary
        np.save(write_dir / 'labels' / (str(i) + '_deterministic'), testY[:, i:i + 1000])