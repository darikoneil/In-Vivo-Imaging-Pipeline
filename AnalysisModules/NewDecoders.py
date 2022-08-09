import numpy as np
from sklearn import linear_model


class LogisticRegressionDecoder(object):

    def __init__(self):
        return

    def fit(self, x, y, **kwargs):
        _penalty = kwargs.get('penalty', 'l1')
        _solver = kwargs.get('solver', 'liblinear')
        _max_iter = kwargs.get('max_iter', 100000)
        self.model = linear_model.LogisticRegression(penalty=_penalty, solver=_solver, max_iter=_max_iter)
        self.model.fit(x, y)

    def predict(self, x, **kwargs):
        y_predicted = self.model.predict(x)
        return y_predicted



