import numpy as np
from sklearn import svm
from cmath import exp, log
from .mytree import DecisionTreeClassifier

class BinomialDeviance():
    def __init__(self, loss='deviance'):
        self.pred = None
        self.loss = loss
        self.residual = None

    def initialize(self, Y):
        pos = np.sum(Y)
        neg = len(Y) - pos
        self.pred = np.ones(Y.shape) * log(pos/neg)
        return self.pred, log(pos/neg)

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
    def __init__(self, loss='deviance'):
        self.C = None
        self.loss = loss

    def initialize(self, Y):
        self.C = - log((len(Y)/np.sum(Y)) - 1)
        return self.C


class GradientBoostClassifier():
    ''' Gradient Boosting for classification. '''
    #   Parameters
    #   ----------
    #   loss : {'deviance', 'exponential'}

    def __init__(self, loss='deviance', n_estimators=50, learning_rate=1, max_depth=10):
        self.c = None
        self.f_m = None
        self.loss = None
        self.pred = []
        self.loss_type = loss
        self.estimator_ = {}
        self.f_m_estimator = {}
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.feature_importances_ = None

    def fit(self, X, Y):
        N = len(X)
        index = [True] * N
        n_classes = len(np.unique(Y))
        self.loss = BinomialDeviance(self.loss_type) if n_classes == 2 else MultinomialDeviance(self.loss_type)
        pred, self.c = self.loss.initialize(Y)

        for m in range(self.n_estimators):
            residuals = self.loss.compute_residual(Y, pred)

            tree = DecisionTreeClassifier(X=X, Y=residuals, label=Y, index=index, loss=self.loss, max_depth=self.max_depth)

            self.estimator_[m] = tree

            pred = self.loss.update_f_m(pred, tree, self.learning_rate)
            self.f_m_estimator[m] = pred

        self.f_m = pred

        prob = 1/(1 + np.exp(-pred))
        for i in range(len(X)):
            if prob[i] >= 0.5:
                self.pred.append(1)
            else:
                self.pred.append(0)


    def predict_proba(self, X):
        f_m_estimator = {}
        f_m_estimator[0] = np.zeros(len(X))
        for i in range(self.n_estimators):
            f_m_estimator[i+1] = f_m_estimator[i] 
            for x in range(len(X)):
                f_m_estimator[i+1][x] += self.learning_rate * self.estimator_[i].root.get_predict_value(X[x])

        y_pred = 1/(1 + np.exp(-f_m_estimator[self.n_estimators]))
        return y_pred

    def predict(self, X):
        y_pred = []
        y_prob = self.predict_proba(X)
        for i in range(len(X)):
            if y_prob[i] >= 0.5:
                y_pred.append(1)
            else:
                y_pred.append(0)
        return y_pred




if __name__ == "__main__":
    x = np.array([[1, 5, 20], [2, 7, 30], [3, 21, 70], [4, 30, 60]])
    y = np.array([0, 0, 1, 1])
    xt = np.array([[5, 25, 65]])

    Gbdt = GradientBoostClassifier()
    Gbdt.fit(x, y)
    pred = Gbdt.predict(x)
    print(pred)

