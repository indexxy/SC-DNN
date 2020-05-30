import tempfile
import shutil
import os
import atexit
from pathlib import Path
import h5py
import numpy as np

class CacheDir:
    def __init__(self, pid):
        path = tempfile.mkdtemp(dir=os.getcwd())
        self.path = path
        self.pid = pid

        def exitCleanup():
            shutil.rmtree(path)
        atexit.register(exitCleanup)

    def __repr__(self):
        return self.path

    def __str__(self):
        return self.path

    def remove(self, pid):
        if pid == self.pid:
            shutil.rmtree(self.path)
        else:
            return 0


"""
Inspired by: https://github.com/volf52/deep-neural-net/
"""
def loadHDF5(filePath: Path, useBias: bool):
    assert filePath.exists()

    with h5py.File(filePath, "r") as fp:
        fpKeys = fp.keys()
        assert "units" in fpKeys
        assert "weights" in fpKeys
        assert "useBias" in fpKeys
        weights = []
        biases = []
        ws = fp["weights"]
        for w in ws.values():
            weights.append(np.array(w[()], dtype=np.float32).T)

        if useBias:
            assert fp["useBias"]
            assert "biases" in fpKeys
            bs = fp["biases"]
            for b in bs.values():
                biases.append(np.array(b[()], dtype=np.float32).reshape(-1, 1))

    # Weights shape : (layer_units, input_units)
    # Biases shape : (layer_units, 1)
    return weights, biases

