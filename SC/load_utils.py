from pathlib import Path
import numpy as np
from enum import Enum


DATA_DIR: Path = Path(__file__).parent / "data"
assert DATA_DIR.exists()

MNIST_DIR: Path = DATA_DIR / "mnist"
FASHION_MNIST_DIR: Path = DATA_DIR / "fashion-mnist"
MNIST_C_DIR: Path = DATA_DIR / "mnist-c"


class DATASETS(Enum):
    mnist = "mnist"
    fashion = "fashion-mnist"
    mnistc_brightness = "mnist_c-brightness"
    mnistc_canny_edges = "mnist_c-canny_edges"
    mnistc_dotted_line = "mnist_c-dotted_line"
    mnistc_fog = "mnist_c-fog"
    mnistc_glass_blur = "mnist_c-glass_blur"
    mnistc_identity = "mnist_c-identity"
    mnistc_impulse_noise = "mnist_c-impulse_noise"
    mnistc_motion_blur = "mnist_c-motion_blur"
    mnistc_rotate = "mnist_c-rotate"
    mnistc_scale = "mnist_c-scale"
    mnistc_shear = "mnist_c-shear"
    mnistc_shot_noise = "mnist_c-shot_noise"
    mnistc_spatter = "mnist_c-spatter"
    mnistc_stripe = "mnist_c-stripe"
    mnistc_translate = "mnist_c-translate"
    mnistc_zigzag = "mnist_c-zigzag"
    cifar10 = "cifar-10"
    # affnist = 'affNIST'

    def __repr__(self):
        return self.value

    def __str__(self):
        return self.value


# noinspection PyTypeChecker
def loadDataset(dataset: DATASETS, precision, idx):
    dataset = str(dataset)
    if dataset.startswith('fashion'):
        load_dir = FASHION_MNIST_DIR

    elif dataset.startswith("mnist_c"):
        load_dir = MNIST_C_DIR / dataset.split("-")[-1]

    else:
        load_dir = MNIST_DIR

    load_dir = load_dir / str(precision)

    x = np.load(load_dir / 'images' / (str(idx) + '.npy'))
    y = np.load(load_dir / 'labels' / (str(idx) + '.npy'))
    deterministic_y = np.load(load_dir / 'labels' / (str(idx) + '_deterministic.npy'))

    return x, y, deterministic_y