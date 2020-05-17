from SC.write_utils import dataset2SC
from SC.load_utils import DATASETS


if __name__ == "__main__":
    """
    Converting a dataset to stochastic and storing it
    """
    # It will be written to CURRENT_FOLDER / 'data' / DATASET_NAME / PRECISION /
    dataset = DATASETS.mnist
    precision = 8
    dataset2SC(dataset, precision=precision)
