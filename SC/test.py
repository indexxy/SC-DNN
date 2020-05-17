from dnn.mlpcode.network import Network
from SC.network import SCNetwork
from dnn.mlpcode.loss import LossFuncs as lf
from dnn.mlpcode.activation import ActivationFuncs as af
from dnn.mlpcode.utils import DATASETS, MODELDIR


if __name__ == "__main__":

    modelPath = MODELDIR / 'mnist_bnn_32.npz'

    num_instances = 1000
    precision = 8

    lr = 0.07
    hiddenAf = af.sigmoid
    outAf = af.softmax
    lossFunc = lf.cross_entropy

    useGpu = False
    binarized = True

    network = Network.fromModel(
        modelPath,
        useGpu=useGpu,
        binarized=binarized
    )

    SC_NN = SCNetwork(network, hiddenAf=hiddenAf, outAf=outAf, precision=precision, binarized=binarized)

    datasets = [
        DATASETS.mnistc_identity,
        DATASETS.mnistc_shot_noise,
        DATASETS.mnistc_impulse_noise,
        DATASETS.mnistc_glass_blur,
        DATASETS.mnistc_motion_blur,
        DATASETS.mnistc_shear,
        DATASETS.mnistc_scale,
        DATASETS.mnistc_rotate,
        DATASETS.mnistc_brightness,
        DATASETS.mnistc_translate,
        DATASETS.mnistc_stripe,
        DATASETS.mnistc_fog,
        DATASETS.mnistc_spatter,
        DATASETS.mnistc_dotted_line,
        DATASETS.mnistc_zigzag,
        DATASETS.mnistc_canny_edges,
    ]

    for dataset in datasets:
        correct = SC_NN.testDataset(dataset, num_instances=num_instances, parallel=True)
        print("{:s}: {:.2f}%".format(dataset, (correct * 100) / num_instances))
