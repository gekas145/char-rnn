import theano
import theano.tensor as T
import numpy as np
import time
import json

class NeuralNetwork:

    def __init__(self, input_dim, loss, metric=None, nepochs=10, nbatch=4, alpha=0.001, alpha_decay=1.0, beta1=0.9, beta2=0.999, lmbda=0.0, grad_clip=5, embedding=None):
        # theano function for mini bach training step
        self._train = None
        # # theano function for predicting
        # self._predict = None

        # list of all network layers
        self.layers = []
        # input dimensionality
        self.input_dim = input_dim

        # loss function
        self.loss = loss

        # fitness metric, e.g. accuracy
        self.metric = metric

        # lists of all network weights and biases
        self.weight_params = []
        self.bias_params = []

        # training hyperparameters
        self.nbatch = nbatch # number of samples in mini batch
        self.nepochs = nepochs # number of epochs
        self.alpha = theano.shared(alpha) # learning rate
        self.alpha_decay = alpha_decay # learning rate reduction
        self.beta1 = beta1 # first moment momentum
        self.beta2 = beta2 # second moment momentum
        self.lmbda = lmbda # L2 regularization parameter
        self.grad_clip = grad_clip # maximum value for abs(grad[i])

        # input embedding
        self.embedding = embedding

    def add_layer(self, layer):
        self.layers.append(layer)
        prev_dim = self.input_dim if len(self.layers) == 1 else self.layers[-2].dim
        self.layers[-1]._set_params(prev_dim)
        weights, bias = self.layers[-1]._get_params()
        self.weight_params += weights
        self.bias_params += bias

    def compile(self):
        # single_input = T.tensor3()
            
        # prediction = self._construct_predict(single_input)
        # self._predict = theano.function([single_input], prediction)

        batch_inputs = T.tensor3()
        batch_outputs = T.tensor3() if self.layers[-1].seq2seq else T.matrix()

        mini_batch_loss = self._construct_mini_batch_loss(batch_inputs, batch_outputs)
        param_updates, moments_updates = self._get_params_updates(mini_batch_loss)

        train_params = [batch_inputs, batch_outputs]
        
        self._train_dampening = theano.function(train_params, 
                                                mini_batch_loss,
                                                updates=moments_updates)

        self._train_param = theano.function(train_params, 
                                            mini_batch_loss,
                                            updates=param_updates)

    def predict(self, inputs, n=300, temp=1.0):
        self.reset_state()

        y, inputs = inputs[-1], inputs[:-1]
        for x in inputs:
            self._predict(x)
        
        output = []
        for i in range(n):
            pred = np.random.choice(range(self.input_dim), p=self._predict(y, temp=temp))
            y = np.zeros(self.input_dim)
            y[pred] = 1.0
            output.append(pred)
        
        return output
    
    def reset_state(self):
        for layer in self.layers:
            layer.reset_state()
    
    def train(self, X_train, y_train, X_valid=None, y_valid=None):
        n = X_train.shape[0]
        n_valid = X_valid.shape[0]
        loss_valid = []
        metric_scores_valid = []
        loss_train = []
        metric_scores_train = []

        X_valid, y_valid = self.embedding(X_valid), self.embedding(y_valid)
        X_valid = np.array([X_valid[:, i, :] for i in range(X_valid.shape[1])])
        y_valid = np.array([y_valid[:, i, :] for i in range(y_valid.shape[1])])

        X_train_valid, y_train_valid = self.embedding(X_train[0:n_valid, :]), self.embedding(y_train[0:n_valid, :])
        X_train_valid = np.array([X_train_valid[:, i, :] for i in range(X_train_valid.shape[1])])
        y_train_valid = np.array([y_train_valid[:, i, :] for i in range(y_train_valid.shape[1])])

        for epoch in range(self.nepochs):
            m = 0
            for i in range(n//self.nbatch + int(n%self.nbatch > 0)):
                if m + self.nbatch >= n:
                    # wrap around if last batch has fewer samples than self.nbatch
                    X_batch = np.vstack((X_train[m:n, :], X_train[0:self.nbatch + m - n, :]))
                    y_batch = np.vstack((y_train[m:n, :], y_train[0:self.nbatch + m - n, :]))
                else:
                    X_batch, y_batch = X_train[m:m + self.nbatch, :], y_train[m:m + self.nbatch, :]
                
                m += self.nbatch

                if self.embedding:
                    X_batch = self.embedding(X_batch)
                    y_batch = self.embedding(y_batch)

                X_batch = np.array([X_batch[:, j, :] for j in range(X_batch.shape[1])])
                y_batch = np.array([y_batch[:, j, :] for j in range(y_batch.shape[1])])

                # start_time = time.time()

                self._train_dampening(X_batch, y_batch)

                self._train_param(X_batch, y_batch)

                # print(f"Seconds passed {time.time() - start_time}")

                if i%300 == 0:
                    predictions_valid = self.predict(X_valid)
                    predictions_train = self.predict(X_train_valid)

                    current_valid = self.loss.calculate(predictions_valid, y_valid)/n_valid
                    current_train = self.loss.calculate(predictions_train, y_train_valid)/n_valid
                    
                    message = f"current train loss: {current_train}, current valid loss: {current_valid}"
                    print(message)
            
            if X_valid is not None and y_valid is not None:
                predictions_valid = self.predict(X_valid)
                predictions_train = self.predict(X_train_valid)

                loss_valid.append(self.loss.calculate(predictions_valid, y_valid)/n_valid)
                loss_train.append(self.loss.calculate(predictions_train, y_train_valid)/n_valid)
                
                message = f"[Epoch {epoch + 1}] train loss: {loss_train[-1]}, valid loss: {loss_valid[-1]}"
                print(message)
            
                # make a checkpoint
                self.save(f"checkpoints/net_{epoch + 5}_{self.layers[0].dim}_loss_{loss_valid[-1]:.3f}.json")
            
            # decay the learning rate
            self.alpha.set_value(self.alpha.get_value() * self.alpha_decay)

        return loss_train, loss_valid, metric_scores_train, metric_scores_valid
    
    def _get_params_updates(self, loss):
        moments_updates = []
        param_updates = []
        eps = 10**(-8)

        correction1 = theano.shared(self.beta1)
        correction2 = theano.shared(self.beta2)
        moments_updates.append((correction1, correction1 * self.beta1))
        moments_updates.append((correction2, correction2 * self.beta2))

        # add weights updates
        for weights in self.weight_params:
            grad = T.grad(loss, weights)
            
            if self.lmbda > 0.0:
                grad += self.lmbda * weights
            
            first_moment = theano.shared(np.zeros(weights.shape.eval()))
            second_moment = theano.shared(np.zeros(weights.shape.eval()))

            moments_updates.append((first_moment, self.beta1*first_moment + (1.0 - self.beta1)*grad))
            moments_updates.append((second_moment, self.beta2*second_moment + (1.0 - self.beta2)*grad**2))

            param_updates.append((weights, weights - self.alpha * T.sqrt(1 - correction2)/(1 - correction1) * first_moment/T.sqrt(second_moment + eps)))
        
        # add bias updates
        for bias in self.bias_params:
            grad = T.grad(loss, bias)
            
            first_moment = theano.shared(np.zeros(bias.shape.eval()))
            second_moment = theano.shared(np.zeros(bias.shape.eval()))

            moments_updates.append((first_moment, self.beta1*first_moment + (1.0 - self.beta1)*grad))
            moments_updates.append((second_moment, self.beta2*second_moment + (1.0 - self.beta2)*grad**2))

            param_updates.append((bias, bias - self.alpha * T.sqrt(1 - correction2)/(1 - correction1) * first_moment/T.sqrt(second_moment + eps)))
        
        return param_updates, moments_updates
    
    def _predict(self, x, temp=None):
        y = x
        for i in range(len(self.layers) - 1):
            y = self.layers[i].predict(y)

        y = self.layers[-1].predict(y, temp) if temp is not None else self.layers[-1].predict(y)
        
        return y
    
    def _construct_train_predict(self, inputs):
        y = inputs
        for i in range(len(self.layers)):
            y = self.layers[i].feedforward(y)
        
        return y
    
    def _construct_mini_batch_loss(self, batch_inputs, batch_outputs):
        return self.loss(self._construct_train_predict(theano.gradient.grad_clip(batch_inputs, -self.grad_clip, self.grad_clip)), 
                         batch_outputs)/self.nbatch
    
    def save(self, path):
        d = []

        for layer in self.layers:
            d.append(layer.to_dict())

        with open(path, "w") as f:
            json.dump(d, f)


    def load(self, path):
        with open(path, "r") as f:
            d = json.load(f)

        for i in range(len(d)):
            self.layers[i].from_dict(d[i])


    def inference_on(self):
        for layer in self.layers:
            layer.inference_on()       




