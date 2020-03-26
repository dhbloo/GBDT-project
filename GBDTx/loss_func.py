import numpy as np
from cmath import exp, log

class BinomialDeviance():
    def __init__(self):
        self.pred = None
        self.residual = None

    def initialize(self, Y):
        pos = np.sum(Y)
        neg = len(Y) - pos
        self.pred = np.ones(Y.shape) * log(pos/neg)
        return self.pred

    def compute_residual(self, Y, pred):
        self.residual = (Y - 1/(1 + np.exp(-pred)))
        return self.residual

    def update_leaf_values(self, f_m, y):
        numerator = f_m.sum()
        if numerator == 0:
            return 0.0
        denominator = ((y - f_m) * (1 - y + f_m)).sum()
        if abs(denominator) < 1e-150:
            return 0.0
        else:
            return numerator / denominator

    def update_f_m(self, f_m_1, tree, learning_rate):   
        f_m = np.zeros(len(f_m_1)) + f_m_1
        for leaf_node in tree.leaf_nodes:
            f_m[leaf_node.data_index] += learning_rate * leaf_node.predict_value
        return f_m


class MultinomialDeviance():
    def __init__(self):
        self.classes = None
        self.residual = None

    def initialize(self, Y):
        self.classes = Y.shape[1]
        f_0 = np.ones(Y.shape) * Y.sum(axis=0) / len(Y)
        return f_0

    def compute_residual(self, Y, pred):
        pred_exp = np.exp(pred)
        pred_sum_exp = pred_exp.sum(axis=1)

        residuals = []
        for i in range(Y.shape[0]):
            residuals.append(Y[i] - pred_exp[i] / pred_sum_exp[i])
        
        self.residual = np.array(residuals)
        return self.residual

    def update_leaf_values(self, f_m, y):
        numerator = f_m.sum()
        if numerator == 0:
            return 0.0
        numerator *= (self.classes - 1)/self.classes
        denominator = ((y - f_m) * (1 - y - f_m)).sum()

        if abs(denominator) < 1e-150:
            return 0.0
        else:
            return numerator / denominator

    def update_f_m(self, f_m_1, tree, learning_rate):   
        f_m = np.zeros(np.shape(f_m_1)) + f_m_1
        for leaf_node in tree.leaf_nodes:
            f_m[leaf_node.data_index] += learning_rate * leaf_node.predict_value
        return f_m