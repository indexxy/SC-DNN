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
def loadModel(filePath: Path):
    assert filePath.exists()
    # print(f"\nLoading model from {filePth}\n")

    with h5py.File(filePath, "r") as fp:
        fpKeys = fp.keys()
        assert "units" in fpKeys
        assert "weights" in fpKeys
        assert "useBias" in fpKeys
        assert "useBatchNorm" in fpKeys

        useBias = bool(fp["useBias"][()])
        useBatchNorm = bool(fp["useBatchNorm"][()])
        if useBatchNorm and useBias:
            print("Setting useBias to false cause of batchnorm")
            useBias = False

        weights = []
        biases = []
        ws = fp["weights"]
        batchNormParams = None
        for w in ws.values():
            weights.append(np.array(w[()], dtype=np.float32).T)

        if useBias:
            assert "biases" in fpKeys
            bs = fp["biases"]
            for b in bs.values():
                biases.append(np.array(b[()], dtype=np.float32).reshape(-1, 1))

        if useBatchNorm:
            assert "gammas" in fpKeys
            assert "betas" in fpKeys
            assert "mus" in fpKeys
            assert "sigmas" in fpKeys

            gammas = []
            betas = []
            mus = []
            sigmas = []

            gs = fp["gammas"]
            bts = fp["betas"]
            muDS = fp["mus"]
            sigmaDS = fp["sigmas"]

            for gamma in gs.values():
                gammas.append(np.array(gamma[()], dtype=np.float32).reshape(-1, 1))

            for beta in bts.values():
                betas.append(np.array(beta[()], dtype=np.float32).reshape(-1, 1))

            for mu in muDS.values():
                mus.append(np.array(mu[()], dtype=np.float32).reshape(-1, 1))

            for sigma in sigmaDS.values():
                sigmas.append(np.array(sigma[()], dtype=np.float32).reshape(-1, 1))

            batchNormParams = dict(
                sigmas=sigmas,
                gammas=gammas,
                mus=mus,
                betas=betas
            )

        # Weights shape : (layer_units, input_units)
        # Biases shape : (layer_units, 1)

        return weights, biases, batchNormParams
