from dnn.mlpcode.activation import ActivationFuncs as af
from SC.load_utils import DATASETS
from SC.network import SCNetwork

if __name__ == "__main__":
    from dnn.mlpcode.network import Network
    from dnn.mlpcode.utils import MODELDIR

    """
    Testing accuracy of an SCNetwork on a specific dataset
    """
    # Parameters
    dataset = DATASETS.mnist  # dataset to be tested
    precision = 64  # Stochastic bit-stream's length
    binarized = True  # Whether the network is binarized or not
    hiddenAf = af.sign  # Activation function of the hidden layers
    outAf = af.softmax  # Activation function if the output layer
    parallel = True  # Divide the dataset to batches and test them in parallel

    # number of instances to be tested (This should be a multiple of 1000)
    num_instances = 10000

    # Path of a trained DNN model
    modelPath = MODELDIR / "NAME_OF_YOUR_MODEL.npz"
    # Initializing a DNN using a pre-trained model
    nn = Network.fromModel(
        modelPath,
        useGpu=False,
        binarized=binarized
    )

    # initializing an SC-DNN using a pre-trained DNN
    sc_nn = SCNetwork(
        network=nn,
        hiddenAf=hiddenAf,
        outAf=outAf,
        precision=precision,
        binarized=binarized,
        hidden_scale=10,
        out_scale=256
    )

    # number of correct predictions
    correct = sc_nn.testDataset(dataset, num_instances=num_instances, parallel=parallel)
    print("{:s}: {:.2f}%".format(dataset, (correct * 100) / num_instances))
