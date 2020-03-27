import math
import numpy as np
from cmath import exp, log
from .mytree import DecisionTreeClassifier
from .loss_func import BinomialDeviance, MultinomialDeviance


class GradientBoostClassifier():
    ''' Gradient Boosting for classification. '''
    # --------------
    #   Parameters
    # --------------
    #   loss : {'deviance', 'exponential'}

    def __init__(self, n_estimators=50, learning_rate=1, max_depth=10):
        self.pred = []
        self.loss = None
        self.n_classes = None
        self.feature_num = None
        self.train_score = None
        self.estimator_ = {}
        self.f_m_estimator = {}
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.feature_importance = None


    def fit(self, X, Y):
        N = len(X)
        Y_label = Y
        index = [True] * N
        n_classes = len(np.unique(Y))

        self.feature_num = np.shape(X)[1]
        self.n_classes = n_classes
        self.loss = BinomialDeviance() if n_classes == 2 else MultinomialDeviance()
        #transform the label into onehot encoding
        if n_classes > 2:
            Y = np.eye(N, n_classes)[Y]

        pred = self.loss.initialize(Y)

        for m in range(self.n_estimators):
            residuals = self.loss.compute_residual(Y, pred)
            if n_classes == 2:
                tree = DecisionTreeClassifier(X=X, Y=residuals, label=Y, index=index, \
                                                loss=self.loss, max_depth=self.max_depth)

                self.estimator_[m] = tree

                pred = self.loss.update_f_m(pred, tree, self.learning_rate)
                self.f_m_estimator[m] = pred
            else:
                self.estimator_[m] = {}
                for i in range(n_classes):
                    tree = DecisionTreeClassifier(X=X, Y=residuals[:, i], label=Y[:, i], index=index, \
                                                    loss=self.loss, max_depth=self.max_depth)

                    self.estimator_[m][i] = tree

                    pred[:, i] = self.loss.update_f_m(pred[:, i], tree, self.learning_rate)
                self.f_m_estimator[m] = pred

        if self.n_classes == 2:
            prob = 1/(1 + np.exp(-pred))
            self.pred = (prob >= 0.5).astype('int')
        else:
            pred_exp = np.exp(pred)
            pred_sum_exp = pred_exp.sum(axis=1)
            prob_list = []
            for i in range(pred_exp.shape[0]):
                prob_list.append(pred_exp[i] / pred_sum_exp[i])
            prob = np.array(prob_list)

            self.pred = prob.argmin(axis=1)

        self.train_score = np.sum(self.pred==Y_label)/len(Y_label)
        self.feature_importance = self.get_feature_importances_()

    def predict_proba(self, X):
        f_m_estimator = {}
        if self.n_classes == 2:
            f_m_estimator[0] = np.zeros(len(X))
        else:
            f_m_estimator[0] = np.zeros((len(X), self.n_classes))

        for i in range(self.n_estimators):
            f_m_estimator[i+1] = f_m_estimator[i] 

            if self.n_classes == 2:
                for x in range(len(X)):
                    f_m_estimator[i+1][x] += self.learning_rate * self.estimator_[i].root.get_predict_value(X[x])
            else:
                for c in range(self.n_classes):
                    for x in range(len(X)):
                        f_m_estimator[i+1][x][c] += self.learning_rate * self.estimator_[i][c].root.get_predict_value(X[x])

        if self.n_classes == 2:
            y_pred = 1/(1 + np.exp(- f_m_estimator[self.n_estimators]))
        else:
            pred_exp = np.exp(f_m_estimator[self.n_estimators])
            pred_sum_exp = pred_exp.sum(axis=1)

            prob_list = []
            for i in range(pred_exp.shape[0]):
                prob_list.append(pred_exp[i] / pred_sum_exp[i])
            y_pred = np.array(prob_list)

        return y_pred

    def predict(self, X):
        y_prob = self.predict_proba(X)

        if self.n_classes == 2:
            y_pred = (y_prob >= 0.5).astype('int')
        else:
            y_pred = y_prob.argmin(axis=1)

        return y_pred

    def get_feature_importances_(self):
        if self.n_classes == 2:
            feat_importance = np.zeros(self.feature_num)
        else:
            feat_importance = np.zeros((self.n_classes, self.feature_num))
        
        for i in range(self.n_estimators):
            if self.n_classes == 2:
                feat_importance += self.estimator_[i].root.get_feature_importance()
            else:
                for c in range(self.n_classes):
                    feat_importance[c] += self.estimator_[i][c].root.get_feature_importance()
        self.feature_importance = feat_importance / self.n_estimators

        return feat_importance / self.n_estimators

    def score(self, X, Y):
        y_pred = self.predict(X)
        score = np.sum(Y == y_pred)/len(Y)
        return score



if __name__ == "__main__":
    x = np.array([[1, 5, 20], [2, 7, 30], [3, 21, 70], [4, 30, 60]])
    y = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    yt = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    xt = np.array([[5, 25, 65]])

    # Gbdt = GradientBoostClassifier()
    # Gbdt.fit(x, y)
    # pred = Gbdt.predict(x)
    # print(pred)
    q = MultinomialDeviance()
    f = q.compute_residual(y, yt)
    print(f)
