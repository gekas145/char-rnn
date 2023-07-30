import theano.tensor as T
import numpy as np

class Loss:

    def __init__(self):
        pass

    # use for symbolic ops
    def __call__(self, predicted, real):
        pass
    
    # use for real ops
    def calculate(self, predicted, real):
        pass

class SparseCategoricalCrossentropy(Loss):

    def __call__(self, predicted, real):
        return -T.sum(T.log(predicted) * real)
    
    def calculate(self, predicted, real):
        return -np.sum(np.log(predicted) * real)



