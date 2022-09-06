# creating a perceptron class for classification algorithm
import numpy as np

class Perceptron:
    """
    Perceptron classifier algorithm

    Parameters:
    h: learning rate in [0,1]
    n: integer for number of times to loop over algorithm
    random_state: integer to set random generator seed for the random weight initialization
    
    Attributes:
    w: 1d array for weights after fitting
    b: bias unit after fitting

    errors: list of number of misclassifications (updates) in each epoch
    """

    def __init__(self, h = 0.1, n = 50, random_state = 1):
        self.h = h
        self.n = n
        self.random_state = random_state
    
    def fit(self, X, y):
        """
        Fit the perceptron decision boundary to data
        ----------
        Parameters:
        X: matrix with n examples and m features
        y: n target/response values

        Returns:
        self: object
        """
        random_gen = np.random.RandomState(self.random_state)
        self.w = random_gen.normal(loc= 0.0, scale = 0.01, size= X.shape[1])
        self.b = np.float_(0.0)
        self.errors = []

        for _ in range(self.n):
            error = 0
            for x_i, y_hat in zip(X, y):
                update = self.h * (y_hat - self.predict(x_i))
                
                self.w += update * x_i
                self.b += update

                error += int(update != 0.0)
            self.errors.append(error)
        
        return self
    
    def net_input(self, X):
        """
        Calculates the vector of X*w + b for input to be used in the prediction function

        """
        return(np.dot(X,self.w) + self.b)
    
    def predict(self, X):
        return (np.where(self.net_input(X) >= 0.0, 1, 0))
    
    

