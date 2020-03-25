import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, check_random_state
from sklearn.utils.multiclass import unique_labels
from sklearn.tree import DecisionTreeRegressor
from .Loss import MeanSquareLoss

losses = {'ls': MeanSquareLoss}


class GBDTRegressor(BaseEstimator, RegressorMixin):
    """ Gradient Boosting Decision Tree Regressor  """
    """ based on sklearn.DecisionTreeRegressor     """
    def __init__(self,
                 n_estimators=100,
                 learning_rate=0.1,
                 loss='ls',
                 subsample=1.0,
                 tree_params=None,
                 random_state=None):
        assert n_estimators > 0
        assert loss in losses

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss 
        self.subsample = subsample
        self.tree_params = tree_params
        self.random_state = random_state

    def fit(self, X, y):
        # Ensure input is dense nparray
        X, y = check_X_y(X, y)

        # Get loss function and tree params and random state
        loss = losses[self.loss]()
        tree_params = self.tree_params if self.tree_params else {}
        random = check_random_state(self.random_state)

        # Initialize f_0 by minimize loss function
        self.f0_ = loss.init_f_0(y)
        fm = np.zeros(len(y)) + self.f0_

        self.estimator_ = []
        self.lambdas_ = []
        self.train_score_ = np.zeros(self.n_estimators)
        self.feature_importances_ = np.zeros(X.shape[1])
        self.oob_improvement_ = np.zeros(self.n_estimators) if self.subsample < 1.0 else None
        subsample_count = int(np.ceil(len(X) * self.subsample))
        last_oob = loss.compute_loss(fm, y)

        for m in range(self.n_estimators):
            # Record train score and calc residuals at this iteration
            self.train_score_[m] = loss.compute_loss(fm, y)
            residuals = loss.compute_residual(fm, y)

            # Create decision tree
            tree = DecisionTreeRegressor(random_state=random.randint(2147483647), **tree_params)

            # Choose a subsample of dataset and fit the tree
            if self.subsample < 1.0:
                sub_index = random.choice(range(len(X)), subsample_count)
                tree.fit(X[sub_index], residuals[sub_index])
            else:
                tree.fit(X, residuals)

            # Add up feature importance
            self.feature_importances_ += tree.feature_importances_

            # Update f_m
            pred = tree.predict(X)
            lambda_ = self.learning_rate * loss.compute_lambda(pred, residuals)
            fm += lambda_ * pred

            # Calculate out-of-bag error
            if self.subsample < 1.0:
                test_index = list(set(range(len(X))) - set(sub_index))
                cur_oob = loss.compute_loss(fm[test_index], y[test_index])
                self.oob_improvement_[m] = last_oob - cur_oob
                last_oob = cur_oob

            # Append estimator and lambda
            self.estimator_.append(tree)
            self.lambdas_.append(lambda_)

        # Get average feature importance
        self.feature_importances_ /= self.n_estimators

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        fm = np.zeros(len(X)) + self.f0_

        for m in range(self.n_estimators):
            fm += self.lambdas_[m] * self.estimator_[m].predict(X)

        return fm

    def staged_predict(self, X):
        """ Return a generator for prediction at each iteration """
        check_is_fitted(self)
        X = check_array(X)

        fm = np.zeros(len(X)) + self.f0_

        for m in range(self.n_estimators):
            yield fm
            fm += self.lambdas_[m] * self.estimator_[m].predict(X)

        return fm


if __name__ == "__main__":
    x = np.array([[1, 5, 20], [2, 7, 30], [3, 21, 70], [4, 30, 60]])
    y = np.array([0, 0, 1, 1])

    Gbdt = GBDTClassifier(100, 0.1, 'ls', tree_params={'max_depth': 3})
    Gbdt.fit(x, y)
    pred = Gbdt.predict(x)
    print(pred)