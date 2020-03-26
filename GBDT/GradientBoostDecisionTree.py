import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, check_random_state
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import LabelBinarizer
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from .Loss import MeanSquareLoss, LogisticLoss
from sklearn.model_selection import train_test_split
from sklearn.metrics import zero_one_loss

losses = {'ls': MeanSquareLoss, 'deviance': LogisticLoss}


class GBDTRegressor(BaseEstimator, RegressorMixin):
    """ Gradient Boosting Decision Tree Regressor  """
    """ based on sklearn.DecisionTreeRegressor     """
    def __init__(self,
                n_estimators=100,
                learning_rate=0.1,
                loss='ls',
                subsample=1.0,
                tree_params=None,
                random_state=None,
                tol=None,
                n_iter_no_change=2):
        assert n_estimators > 0
        assert loss in losses

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.subsample = subsample
        self.tree_params = tree_params
        self.random_state = random_state

    def fit(self, X, y):
        if self.tol is not None and self.subsample==1.0:
            self.subsample = 0.8
        # Ensure input is dense nparray
        X, y = check_X_y(X, y)

        # Get loss function and tree params and random state
        loss = losses[self.loss]()
        tree_params = self.tree_params if self.tree_params else {}
        random = check_random_state(self.random_state)

        # Initialize f_0 by minimize loss function
        self.f0_ = loss.init_f_0(y)
        fm = np.zeros(len(y)) + self.f0_

        self.estimator_ = np.empty(self.n_estimators, dtype=DecisionTreeRegressor)
        self.gammas_ = np.zeros(self.n_estimators)
        self.train_score_ = np.zeros(self.n_estimators)
        self.feature_importances_ = np.zeros(X.shape[1])
        self.oob_improvement_ = np.zeros(self.n_estimators) if self.subsample < 1.0 else None
        subsample_count = int(np.ceil(len(X) * self.subsample))
        last_oob = loss.compute_loss(fm, y)

        tol_init = self.tol
        no_change_itr = 0
        former_oob = last_oob

        for m in range(self.n_estimators):
            # Calc residuals at this iteration
            residuals = loss.compute_residual(fm, y)

            # Create decision tree
            tree = DecisionTreeRegressor(random_state=random, **tree_params)

            # Choose a subsample of dataset and fit the tree
            if self.subsample < 1.0:
                sub_index = random.choice(range(len(X)), subsample_count)
                tree.fit(X[sub_index], residuals[sub_index])
            else:
                tree.fit(X, residuals)

            # Add up feature importance
            self.feature_importances_ += tree.feature_importances_

            # Update f_m and gamma
            pred = tree.predict(X)
            self.gammas_[m] = self.learning_rate * loss.compute_gamma(pred, residuals)
            fm += self.gammas_[m] * pred

            # Record train score
            self.train_score_[m] = loss.compute_loss(fm, y)

            # Calculate out-of-bag error
            if self.subsample < 1.0:
                test_index = list(set(range(len(X))) - set(sub_index))
                cur_oob = loss.compute_loss(fm[test_index], y[test_index])
                self.oob_improvement_[m] = last_oob - cur_oob
                former_oob = last_oob
                last_oob = cur_oob

            #early stopping
            if m % 10 == 0 and m > 300 and self.tol is not None:
                if cur_oob > (former_oob + self.tol):
                    no_change_itr += 1
                    self.tol /= 2
                else:
                    no_change_itr = 0
                    self.tol = tol_init
                
                if no_change_itr == 2:
                    self.n_estimators = m
                    print("early stopping in round {}, best round is {}, M = {}".format(m, m - 20, self.n_estimators))
                    # print("loss: ", later_loss)
                    break
                former_oob = cur_oob
            # Append estimator
            self.estimator_[m] = tree

        # Get average feature importance
        self.feature_importances_ /= self.n_estimators

        return self

    def predict(self, X):
        X = check_array(X)

        fm = np.zeros(len(X)) + self.f0_

        for m in range(self.n_estimators):
            fm += self.gammas_[m] * self.estimator_[m].predict(X)

        return fm

    def staged_predict(self, X):
        """ Return a generator for prediction at each iteration """
        X = check_array(X)

        fm = np.zeros(len(X)) + self.f0_

        for m in range(self.n_estimators):
            yield fm
            fm += self.gammas_[m] * self.estimator_[m].predict(X)

        return fm


class GBDTClassifier(BaseEstimator, ClassifierMixin):
    """ Gradient Boosting Decision Tree Classifier  """
    """ based on sklearn.DecisionTreeClassifier     """
    def __init__(self,
                 n_estimators=100,
                 learning_rate=0.1,
                 loss='deviance',
                 subsample=1.0,
                 tree_params=None,
                 random_state=None,
                 tol=None,
                 n_iter_no_change=2):
        assert n_estimators > 0
        assert loss in losses

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.subsample = subsample
        self.tree_params = tree_params
        self.random_state = random_state

    def fit(self, X, y):
        if self.tol is not None and self.subsample==1.0:
            self.subsample = 0.8
        # Ensure input is dense nparray
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        n_classes = 1 if len(self.classes_) == 2 else len(self.classes_)

        encoder = LabelBinarizer()
        encoder.fit(y)
        y_classes = encoder.transform(y).T

        # Get loss function and tree params and random state
        loss = losses[self.loss]()
        tree_params = self.tree_params if self.tree_params else {}
        random = check_random_state(self.random_state)

        # Initialize f_0 by minimize loss function
        self.f0_ = np.array([loss.init_f_0(y) for y in y_classes])
        fm = np.zeros((n_classes, len(y))) + self.f0_.reshape(n_classes, -1)

        self.estimator_ = np.empty((self.n_estimators, n_classes), dtype=DecisionTreeRegressor)
        self.gammas_ = np.zeros((self.n_estimators, n_classes))
        self.train_score_ = np.zeros(self.n_estimators)
        self.feature_importances_ = np.zeros(X.shape[1])
        self.oob_improvement_ = np.zeros(self.n_estimators) if self.subsample < 1.0 else None
        subsample_count = int(np.ceil(len(X) * self.subsample))
        last_oob = loss.compute_loss(fm, y)

        tol_init = self.tol
        no_change_itr = 0
        former_oob = last_oob

        for m in range(self.n_estimators):
            # Choose a subsample of dataset
            if self.subsample < 1.0:
                sub_index = random.choice(range(len(X)), subsample_count)
                test_index = list(set(range(len(X))) - set(sub_index))
                cur_oob = 0.0

            for i_class in range(n_classes):
                # Calc residuals at this iteration
                residuals = loss.compute_residual(fm[i_class], y_classes[i_class])

                # Create decision tree
                tree = DecisionTreeRegressor(random_state=random, **tree_params)

                # Fit the tree
                if self.subsample < 1.0:
                    tree.fit(X[sub_index], residuals[sub_index])
                else:
                    tree.fit(X, residuals)

                # Add up feature importance
                self.feature_importances_ += tree.feature_importances_

                # Update f_m and gamma
                pred = tree.predict(X)
                self.gammas_[m][i_class] = self.learning_rate * loss.compute_gamma(pred, residuals)
                fm[i_class] += self.gammas_[m][i_class] * pred

                # Record train score
                self.train_score_[m] += loss.compute_loss(fm[i_class], y_classes[i_class])

                # Append estimator for this class
                self.estimator_[m][i_class] = tree

                # Calculate out-of-bag error
                if self.subsample < 1.0:
                    cur_oob += loss.compute_loss(fm[i_class][test_index], y_classes[i_class][test_index])

            # Record out-of-bag error
            if self.subsample < 1.0:
                self.oob_improvement_[m] = last_oob - cur_oob
                former_oob = last_oob
                last_oob = cur_oob
            
            #early stopping
            if m % 10 == 0 and m > 300 and self.tol is not None:
                if cur_oob > (former_oob + self.tol):
                    no_change_itr += 1
                    self.tol /= 2
                else:
                    no_change_itr = 0
                    self.tol = tol_init
                
                if no_change_itr == 2:
                    self.n_estimators = m
                    print("early stopping in round {}, best round is {}, M = {}".format(m, m - 20, self.n_estimators))
                    # print("loss: ", later_loss)
                    break
                former_oob = cur_oob



        # Get average feature importance
        self.feature_importances_ /= (self.n_estimators * n_classes)

        return self

    def _get_y_class(self, fm):
        if len(fm) == 1:
            return np.apply_along_axis(lambda x: self.classes_[0 if x < 0.5 else 1], 0, fm)
        else:
            # softmax
            fm = np.exp(fm)
            fm = fm / fm.sum(axis=0)

            index = np.argmax(fm, axis=0)
            return np.apply_along_axis(lambda i: self.classes_[i], 0, index)

    def predict_log_proba(self, X):
        X = check_array(X)
        n_classes = 1 if len(self.classes_) == 2 else len(self.classes_)

        fm = np.zeros((n_classes, len(X))) + self.f0_.reshape(n_classes, -1)

        for m in range(self.n_estimators):
            for i_class in range(n_classes):
                fm[i_class] += self.gammas_[m][i_class] * self.estimator_[m][i_class].predict(X)

        return fm.T

    def predict_proba(self, X):
        fm = self.predict_log_proba(X)

        return np.exp(fm) / (1 + np.exp(fm))

    def predict(self, X):
        fm = self.predict_proba(X)

        return self._get_y_class(fm.T)

    def staged_predict_log_proba(self, X):
        """ Return a generator for prediction at each iteration """
        X = check_array(X)
        n_classes = 1 if len(self.classes_) == 2 else len(self.classes_)

        fm = np.zeros((n_classes, len(X))) + self.f0_.reshape(n_classes, -1)

        for m in range(self.n_estimators):
            yield fm.T
            for i_class in range(n_classes):
                fm[i_class] += self.gammas_[m][i_class] * self.estimator_[m][i_class].predict(X)

        return fm.T

    def staged_predict_proba(self, X):
        """ Return a generator for prediction at each iteration """
        X = check_array(X)
        n_classes = 1 if len(self.classes_) == 2 else len(self.classes_)

        fm = np.zeros((n_classes, len(X))) + self.f0_.reshape(n_classes, -1)

        for m in range(self.n_estimators):
            yield (np.exp(fm) / (1 + np.exp(fm))).T
            for i_class in range(n_classes):
                fm[i_class] += self.gammas_[m][i_class] * self.estimator_[m][i_class].predict(X)

        return (np.exp(fm) / (1 + np.exp(fm))).T

    def staged_predict(self, X):
        """ Return a generator for prediction at each iteration """
        X = check_array(X)
        n_classes = 1 if len(self.classes_) == 2 else len(self.classes_)

        fm = np.zeros((n_classes, len(X))) + self.f0_.reshape(n_classes, -1)

        for m in range(self.n_estimators):
            yield self._get_y_class(np.exp(fm) / (1 + np.exp(fm)))
            for i_class in range(n_classes):
                fm[i_class] += self.gammas_[m][i_class] * self.estimator_[m][i_class].predict(X)

        return self._get_y_class(np.exp(fm) / (1 + np.exp(fm)))


if __name__ == "__main__":
    x = np.array([[1, 5, 20], [2, 7, 30], [3, 21, 70], [4, 30, 60]])
    y = np.array([0, 0, 1, 1])

    Gbdt = GBDTRegressor(100, 0.1, 'ls', tree_params={'max_depth': 3})
    Gbdt.fit(x, y)
    pred = Gbdt.predict(x)
    print(pred)