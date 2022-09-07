import numpy as np


class AdalineGD:
    """
    Adaptive Linear Neuron Classifier

    params:
    h: learning rate in [0,1]
    n: # iterations
    random_state: integer to keep results equivalent

    attributes:
    w: 1d-array
    b: bias scalar
    losses: list of mse loss function values in each epoch
    """
    def __init__(self, h = 0.1, n = 50, random_state = 1):
        self.h = h
        self.n = n
        self.random_state = random_state
    
    def fit(self, X, y):
        """
        Fit training data
        """
        rgen = np.random.RandomState(self.random_state)
        
        self.w = rgen.normal(loc = 0.0, scale = 0.01, size = X.shape[1])
        self.b = np.float_(0.)
        self.losses = []

        for _ in range(self.n):
            # calculate linear function
            net_input = self.net_input(X)
            # find output of above and calculate errors
            output = self.activation(net_input)
            errors = (y - output)
            # full batch gradient descent derivative to update params
            self.w += 2*self.h*(X.T.dot(errors))/X.shape[0]
            self.b += 2*self.h*errors.mean()
            loss = (errors**2).mean()
            self.losses.append(loss)
        
        return self
    
    def net_input(self, X,):
        """
        Calculate net input
        """
        return (np.dot(X,self.w) + self.b)
    
    def activation(self, X):
        """
        Compute linear activation
        """
        return X
    
    def predict(self, X):
        """
        Return class label after unit step
        """
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1,0)

    def test(self):
        pass
    
