from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class SBS:
    def __init__(self, estimator, k_features, scoring=accuracy_score, test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state
    
    def fit(self, X, y):
        """
        
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        dimensions = X_train.shape[1]
        self.indices = tuple(range(dimensions))
        self.subsets = [self.indices]
        score = self.calc_score(X_train, X_test, y_train, y_test, self.indices)

        self.scores = [score]

        while dimensions > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices, r= dimensions -1):
                score = self.calc_score(X_train, X_test, y_train, y_test, p)
                scores.append(score)
                subsets.append(p)
            
            best = np.argmax(scores)
            self.indices = subsets[best]
            self.subsets.append(self.indices)
            
            dimensions -= 1

            self.scores.append(scores[best])
        
        self.k_score = self.scores[-1]

        return self

    def calc_score(self, X_train, X_test, y_train, y_test, indices):
        """
        
        """
        self.estimator.fit(X_train[:,indices], y_train)
        y_pred = self.estimator.predict(X_test[:,indices])
        
        return self.scoring(y_pred,y_test)
    
    def transform(self, X):
        return X[:, self.indices]
