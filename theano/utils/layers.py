import theano
import theano.tensor as T
import numpy as np

class Layer:

    def __init__(self, dim, activation, dropout, seq2seq):
        self.dim = dim
        self.weights = None
        self.bias = None
        self.activation = activation # pass theano function if want to train, numpy function otherwise
        self.dropout = 1. - dropout if dropout else None
        self.seq2seq = seq2seq
        self.param_names = []

    def feedforward(self, inputs):
        pass

    def predict(self, input_step):
        pass

    def reset_state(self):
        pass

    def _step(self, input_step):
        pass

    def _set_params(self, prev_dim):
        pass

    def _get_params(self):
        pass

    def to_dict(self):
        d = {}

        for name in self.param_names:
            d[name] = getattr(self, name).get_value().tolist()
        
        return d
    
    def inference_on(self):
        for name in self.param_names:
            setattr(self, name, getattr(self, name).get_value())

    def from_dict(self, d):
        weights, bias = self._get_params()
        params = weights + bias
        for i in range(len(self.param_names)):
            params[i].set_value(d[self.param_names[i]])


class FullyConnectedLayer(Layer):
    
    def __init__(self, dim, activation, dropout=None, seq2seq=False):
        super(FullyConnectedLayer, self).__init__(dim, activation, dropout, seq2seq)
        self.param_names = ["weights", "bias"]
    
    def feedforward(self, inputs):
        if self.seq2seq:
            outputs, _ = theano.scan(fn=self._step,
                                     sequences=inputs)
            return outputs
        
        return self._step(inputs)
    
    def predict(self, x, temp=1.0):
        return self.activation((np.dot(x, self.weights) + self.bias)/temp)
    
    def _step(self, input_step):
        return self.activation(T.dot(input_step, self.weights) + self.bias)

    def _set_params(self, prev_dim):
        self.weights = theano.shared(value=np.random.randn(prev_dim, self.dim) * 0.01)

        self.bias = theano.shared(value=np.random.randn(self.dim) * 0.01)
                                  
    def _get_params(self):
        return [self.weights], [self.bias]
    
class LSTMCell(Layer):

    def __init__(self, dim, nbatch, seq2seq=True):
        super().__init__(dim, None, dropout=None, seq2seq=seq2seq)

        self.Wf, self.bf = None, None # forget gate params
        self.Wi, self.bi = None, None # input gate params
        self.Wc, self.bc = None, None # input gate params
        self.Wo, self.bo = None, None # output gate params

        self.h, self.C = None, None # for later inference

        self.nbatch = nbatch

        self.param_names = ["Wf", "Wi", "Wc", "Wo", "bf", "bi", "bc", "bo"]

    def feedforward(self, inputs):
        outputs, _ = theano.scan(fn=self._step,
                                 sequences=inputs,
                                 outputs_info=[np.zeros((self.nbatch, self.dim)),
                                               np.zeros((self.nbatch, self.dim))])
        # get rid of C sequence
        outputs = outputs[0]

        if self.seq2seq:
            return outputs
        
        return outputs[-1]
    
    def predict(self, x):
        xh = np.hstack([x, self.h])
        # forget gate
        f = LSTMCell.sigmoid(np.dot(xh, self.Wf) + self.bf)
        # input gate
        i = LSTMCell.sigmoid(np.dot(xh, self.Wi) + self.bi)
        C_new = LSTMCell.tanh(np.dot(xh, self.Wc) + self.bc)
        self.C = f * self.C + i * C_new
        # output gate
        o = LSTMCell.sigmoid(np.dot(xh, self.Wo) + self.bo)
        self.h = o * LSTMCell.tanh(self.C)

        return self.h
    
    def reset_state(self):
        self.h, self.C = np.zeros(self.dim), np.zeros(self.dim)
    
    def _step(self, input_step, h, C):
        x = T.concatenate([input_step, h], axis=1)
        # forget gate
        f = T.nnet.sigmoid(T.dot(x, self.Wf) + self.bf)
        # input gate
        i = T.nnet.sigmoid(T.dot(x, self.Wi) + self.bi)
        C_new = T.tanh(T.dot(x, self.Wc) + self.bc)
        C = f * C + i * C_new
        # output gate
        o = T.nnet.sigmoid(T.dot(x, self.Wo) + self.bo)
        h = o * T.tanh(C)

        return h, C

    def _set_params(self, prev_dim):
        self.Wf = theano.shared(value=np.random.randn(self.dim + prev_dim, self.dim) * 0.01)
        self.bf = theano.shared(value=np.random.randn(self.dim) * 0.01)

        self.Wi = theano.shared(value=np.random.randn(self.dim + prev_dim, self.dim) * 0.01)
        self.bi = theano.shared(value=np.random.randn(self.dim) * 0.01)

        self.Wc = theano.shared(value=np.random.randn(self.dim + prev_dim, self.dim) * 0.01)
        self.bc = theano.shared(value=np.random.randn(self.dim) * 0.01)

        self.Wo = theano.shared(value=np.random.randn(self.dim + prev_dim, self.dim) * 0.01)
        self.bo = theano.shared(value=np.random.randn(self.dim) * 0.01)

    def _get_params(self):
        return [self.Wf, self.Wi, self.Wc, self.Wo], [self.bf, self.bi, self.bc, self.bo]
    
    @staticmethod
    def sigmoid(x):
        return 1/(1 + np.exp(-x))
    
    @staticmethod
    def tanh(x):
        return 2 * LSTMCell.sigmoid(2*x) - 1
    
