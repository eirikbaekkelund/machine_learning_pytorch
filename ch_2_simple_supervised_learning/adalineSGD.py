import numpy as np


class AdalineSGD:
    """
    Adaptive Linear Neuron Classifier

    params:
    h: learning rate in [0,1]
    n: # iterations
    shuffle: bool (default: True) - shuffles training data every epoch if True to prevent cycles.
    random_state: integer to keep results equivalent

    attributes:
    w: 1d-array
    b: bias scalar
    losses: list of mse loss function values in each epoch
    """
    def __init__(self, h = 0.1, n = 50, shuffle = True, random_state = 1):
        self.h = h
        self.n = n
        self.shuffle = shuffle
        self.w_initialized = False
        self.random_state = random_state
    
    def fit(self, X, y):
        """
        Fit training data
        """
        self.init_weights(X.shape[1])
        self.losses = []

        for _ in range(self.n):
            if self.shuffle:
                X, y = self.shuffle_func(X,y)
            
            losses = []

            for xi, target in zip(X,y):
                losses.append(self.update_weights(xi, target))
            avg_loss = np.mean(losses)
            self.losses.append(avg_loss)
        
        return self
    
    def partial_fit(self, X, y):
        """
        Fit training data without reinitializing the weights
        """
        if not self.w_initialized:
            self.init_weights(X.shape[1])
        
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X,y):
                self.update_weights(xi, target)
        else:
            self.update_weights(X,y)
        
        return self

    def shuffle_func(self, X, y):
        """
        Shuffle training data
        """
        r = self.rgen.permutation(len(y))

        return X[r], y[r]
    
    def init_weights(self, m):
        """
        Initialize weights to small random numbers
        """
        self.rgen = np.random.RandomState(self.random_state)
        self.w = self.rgen.normal(loc = 0., scale = 0.01, size = m)
        self.b = np.float_(0.)
        self.w_initialized = True
    
    def update_weights(self, xi, target):
        """
        Apply standard Adaline learning rule to update weights
        """
        output = self.activation(self.net_input(xi))
        error = (target - output)
        
        self.w += self.h*2*xi*error
        self.b += self.h*2*error

        loss = error**2
        return loss

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
    
