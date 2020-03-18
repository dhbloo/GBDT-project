import numpy as np
from math import exp, log
from sklearn import svm
from sklearn import tree


class AdaBoost():
    def __init__(self, base_estimator, n_estimators, learning_rate):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimator_weights_ = None
        self.miss_pred = None

    #compute the error_rate
    def error_rate(self, y, pred):
        return sum(y != pred)/len(y)

    def train_clf(self, X, Y):
        for i in range(self.n_estimators):
            D = self.estimator_weights_ * len(X)
            self.base_estimator.fit(X, Y, sample_weight=D)

            pred = self.base_estimator.predict(X)
            miss_label = [i for i in (Y != pred)]

            error_rate_m = np.dot(self.estimator_weights_ * miss_label)
            self.miss_pred = [x if x == 1 else -1 for x in miss_label]

        return error_rate_m


    def fit(self, X, Y):
        N = len(X)
        self.estimator_weights_ = np.ones(N)/N

        for i in range(self.n_estimators):
            error_rate = train_clf(X, Y)

            alpha = float(0.5 * log((1 - error_rate)/max(error_rate, 1e-16)))

            Z_m = np.dot(self.estimator_weights_ * exp(- alpha * self.miss_pred))
            self.estimator_weights_ = self.estimator_weights_ * exp(- alpha * self.miss_pred) / Z_m

            
        return 0
