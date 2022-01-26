import numpy as np
import torch as T

class XYDataSet:
    def __init__(self, X, Y, ratio=0.9):
        self.X = X
        self.Y = Y
        self.N = len(X)
        self.ratio = ratio
        rnd_indeces = np.arange(self.N)
        np.random.shuffle(rnd_indeces)
        self.trn_indeces = rnd_indeces[:int(self.N * ratio)]
        self.val_indeces = rnd_indeces[int(self.N * ratio):]

    def get_random_batch(self, batchsize, tensor=True):
        rnd_indeces = np.random.choice(self.trn_indeces, batchsize, replace=False)
        x = self.X[rnd_indeces]
        y = self.Y[rnd_indeces]
        if tensor:
            x = T.tensor(x)
            y = T.tensor(y)
        return x, y

    def get_val_data(self, tensor=True):
        x = self.X[self.val_indeces]
        y = self.Y[self.val_indeces]
        if tensor:
            x = T.tensor(x)
            y = T.tensor(y)
        return x, y