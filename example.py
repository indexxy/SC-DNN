from SC.activation import ActivationFuncs as af
from SC.network import SCNetwork

if __name__ == "__main__":
    from dnn.mlpcode.utils import MODELDIR, DATASETS, loadDataset

    """
    Testing accuracy of an SCNetwork on a specific dataset
    """
    # Parameters :
    dataset = DATASETS.mnist  # dataset to be tested
    _, _, testX, testY = loadDataset(dataset, useGpu=False)

    precision = 1024        # Stochastic bit-stream's length.
    binarized = False       # Whether to binarize the weights and biases or not.
    hiddenAf = af.sigmoid   # Activation function of the hidden layers.
    outAf = af.softmax      # Activation function of the output layer.
    parallel = True         # Divide the dataset into batches and evaluate them simultaneously.
    hidden_scale = 6        # The scaling factors should be chosen
    out_scale = 100         # depending on the activation function used at each layer.

    # Write the weights to a cache folder on the disk
    # and load them layer by layer during forward propagation
    # Although it's slower, this can be used in large networks to prevent memory issues.
    cache = False

    # number of instances to be tested
    num_instances = 1000
    testX = testX[:num_instances]
    testY = testY[:num_instances]

    # Path of a trained DNN model
    modelPath = MODELDIR / "150-300_1590860460.6402.hdf5"

    # initializing an SCNN using a pre-trained DNN
    scnn = SCNetwork(
        modelPath, useBias=True, precision=precision, binarized=binarized, cache=False
    )

    # Compiling the network
    scnn.compile(hiddenAf=hiddenAf, outAf=outAf, hidden_scale=hidden_scale, out_scale=out_scale)
    # Number of correct predictions
    correct = scnn.evaluate(testX, testY, batch_size=50, parallel=parallel)
    print("{:s}: {:.3f}%".format(dataset, (correct * 100) / num_instances))

    """
    Evaluation with fault injection
    """
    # Parameters :
    instances_rate = 0.1    # The rate of instances (Images) to be manipulated.
    features_rate = 0.1     # The rate of features (Pixels) to be manipulated.
    # The rates of weights to be manipulated at each layer (This can be a fixed value too)
    # eg. : weights_rates = 0.3.
    weights_rates = [0.2, 0.1, 0.4]

    # The rates of activations to be manipulated at each layer (The default value is 1.0)
    # it also can be a fixed value, eg. : activations_rates = 0.5.
    activations_rates = [0.5, 0.4, 0.9]

    n_bits = 50     # Number of bits to be manipulated, This should not exceed the total number of bits in a stream.
    mode = 2        # 0 (flip 0s to 1s), 1 (flip 1s to 0s), 2 (Hybrid)

    correct = scnn.faultyEvaluate(
        testX, testY, instances_rate=instances_rate, features_rate=features_rate,
        weights_rates=weights_rates, activations_rates=activations_rates,
        n_bits=n_bits, mode=mode, parallel=parallel
    )
    print("{:s}-faulty evaluation: {:.3f}%".format(dataset, (correct * 100) / num_instances))