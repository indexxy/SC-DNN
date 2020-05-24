from dnn.mlpcode.activation import ActivationFuncs as af
from SC.network import SCNetwork

if __name__ == "__main__":
    from dnn.mlpcode.network import Network
    from dnn.mlpcode.utils import MODELDIR, DATASETS, loadDataset

    """
    Testing accuracy of an SCNetwork on a specific dataset
    """
    # Parameters
    dataset = DATASETS.mnist  # dataset to be tested
    _, _, testX, testY = loadDataset(dataset, useGpu=False)
    precision = 64      # Stochastic bit-stream's length
    binarized = True    # Whether the network is binarized or not
    hiddenAf = af.sign  # Activation function of the hidden layers
    outAf = af.softmax  # Activation function of the output layer
    parallel = True     # Divide the dataset into batches and evaluate them simultaneously

    # number of instances to be tested
    num_instances = 10000

    # Path of a trained DNN model
    modelPath = MODELDIR / "NAME_OF_MODEL.npz"
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
        hidden_scale=1,     # The scaling factors should be chosen
        out_scale=256       # depending on the activation function used at each layer
    )

    # number of correct predictions
    correct = sc_nn.evaluate(testX[:num_instances], testY[:num_instances], batch_size=25, parallel=True)
    print("{:s}: {:.2f}%".format(dataset, (correct * 100) / num_instances))
