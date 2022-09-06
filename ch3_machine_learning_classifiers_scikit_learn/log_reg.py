import numpy as np
# Logistic regression class using full-batch gradient descent

class LogisticRegressionGD:
    """
    Gradient descent-based logistic regression classifier
    
    Parameters:
    --------------
    h - learning rate (float between 0 and 1)
    n_iter - number of iterations over the training set (int)
    random_state - random number generator seed for random weight initialization (int)
    
    Attributes
    --------------
    w_ - weights after traininig (1d-array)
    b_ - bias unit after fitting (scalar)
    losses_ - mean squared errror loss function values in eaxh epoch (list)
    
    """

    def __init__(self, h=0.01, n_iter=50, random_state=1):
        self.h = h
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit training data

        Parameters:
        --------------
        X - Training vectors, where n_examples is the number
        of examples and n_features is the number of features.
        (array-like, shape = [n_examples, n_features])
        y - Target values, (array-like, shape = [n_examples])

        Returns:
        --------------
        self : Instance of LogisticRegressionGD
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        self.losses = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)

            errors = (y - output)
            self.w_ += self.h*2.0*X.T.dot(errors)/X.shape[0]
            self.b_ += self.h*2.0*errors.mean()

            loss = self.loss(X=X, y=y, output=output)
            self.losses.append(loss)
        
        return self
    
    def net_input(self, X):
        """
        Calculate net input
        """
        return np.dot(X,self.w_) + self.b_

    def activation(self,z):
        """
        Compute logistic sigmoid activation
        """
        return 1 / (1 + np.exp(-np.clip(z,-250,250)))
    
    def loss(self, X, y, output):
        """
        Calculate loss function
        """
        return (-y.dot(np.log(output)))-((1-y).dot(np.log(1-output))) / X.shape[0]

    def predict(self, X):
        """
        Return class label after unit step
        """
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
    
