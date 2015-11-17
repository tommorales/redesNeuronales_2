__author__ = 'tmorales'

import numpy as np

from activationFunction import *


class Backpropagatin:
    """
    Backpropagation Network
    """
    def __init__(self, layers, activation="TanSig"):
        if activation == "TanSig":
            activation = TanSig()
        if activation == "LogSig":
            activation = LogSig()
        # Set weights
        self.weights = []
        # layer = [2,2,1]
        # range of weights (-1,1)
        # input and hidden layer -
        for i in range(1, len(layers)-1):
            r = 2 * np.random.random((layers[i-1]+1, layers[i]+1)) -1
            self.weights.append(r)



    def fit(self):
        pass

    def predict(self):
        pass


class Hopfield:
    """
    Hopfiel Network
    """
    def __init__(self, patterns, activation="HardLims"):
        if activation == "HardLims":
            self.sgn = HardLims()
        if activation == "HardLim":
            self.sgn = "HardLim"
        # Patterns
        self.patterns = patterns


    def fit(self):
        r, c = self.patterns.shape
        W = np.zeros((c,c))
        for p in self.patterns:
            W = W + np.outer(p,p)
        W[np.diag_indices(c)]=0
        return W/r


    def recall(self, W, patterns, steps=5):
        for step in xrange(steps):
            patterns = self.sgn(np.dot(patterns, W))
        return patterns



