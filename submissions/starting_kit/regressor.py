from sklearn.base import BaseEstimator
from sklearn import linear_model

class Regressor(BaseEstimator):
    def __init__(self):
        pass #self.reg = linear_model.BayesianRidge()

    def fit(self, X, y = None):
        pass #self.reg.fit(X, y)

    def predict(self, X):
        y_pred = X
        return y_pred #self.reg.predict(X)
