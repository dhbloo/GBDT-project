import numpy as np
from sklearn.tree import DecisionTreeRegressor
from .Loss import MeanSquareLoss

losses = {'mse': MeanSquareLoss}


class GBDTRegressor():
    """ Gradient Boosting Decision Tree Regressor  """
    """ based on sklearn.DecisionTreeRegressor     """
    def __init__(self, n_estimators=100, learning_rate=1, loss='mse', tree_params={}):
        assert n_estimators > 0
        assert loss in losses

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = losses[loss]()
        self.estimator_ = []
        self.lambdas = []
        self.tree_params = tree_params

    def fit(self, X, y):
        self.f0 = self.loss.init_f_0(y)
        fm = self.f0.copy()

        for m in range(self.n_estimators):
            residuals = self.loss.compute_residual(fm, y)
            
            tree = DecisionTreeRegressor(**self.tree_params)
            tree.fit(X, residuals)

            pred = tree.predict(X)
            lambda_ = self.loss.compute_lambda(pred, residuals)
            fm += self.learning_rate * lambda_ * pred

            self.estimator_.append(tree)
            self.lambdas.append(lambda_)

    def predict(self, X):
        fm = self.f0.copy()

        for m in range(self.n_estimators):
            fm += self.learning_rate * self.lambdas[m] * self.estimator_[m].predict(X)

        return fm


if __name__ == "__main__":
    x = np.array([[1, 5, 20], [2, 7, 30], [3, 21, 70], [4, 30, 60]])
    y = np.array([0, 0, 1, 1])

    Gbdt = GBDTClassifier(10, 1, 'mse', {'max_depth': 3})
    Gbdt.fit(x, y)
    pred = Gbdt.predict(x)
    print(pred)