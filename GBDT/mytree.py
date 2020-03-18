import numpy as np


class Node:
    def __init__(self, split_feature=None, split_value=None, data_index=None, loss=None, depth=None):
        # Loss function object
        self.loss = loss

        # Depth of this node
        self.depth = depth

        # Indices of feature subset
        self.data_index = data_index

        # Index of feature to split and its value at split point
        self.split_value = split_value
        self.split_feature = split_feature

        # Child nodes
        self.left_child = None
        self.right_child = None

        # Prediction value in this region (only valid for leaf node)
        self.predict_value = None

    def is_leaf(self):
        """ Return whether this is a leaf node """
        return self.left_child is None and self.right_child is None

    def update_predict_value(self, f_m, y):
        """ Update prediction value? """
        self.predict_value = self.loss.update_leaf_values(f_m, y)

    def get_predict_value(self, X):
        """ Recursively get prediction at leaf node """
        if self.is_leaf():
            return self.predict_value
        if X[self.split_feature] <= self.split_value:
            return self.left_child.get_predict_value(X)
        else:
            return self.right_child.get_predict_value(X)


class DecisionTreeClassifier:
    def __init__(self, X=None, Y=None, label=None, max_depth=10, min_samples_split=2, loss=None):
        assert len(X) == len(Y)

        # Loss function object
        self.loss = loss

        # Tree hyper-parameters(max depth, min number of samples to split)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        # Keep track of all the leaf nodes
        self.leaf_nodes = []

        # Fit the tree
        self.root = self.fit(X, Y, label, [True] * len(X))

    def fit(self, Xtrain, Ytrain, label, index, depth=0):
        X = Xtrain[index]
        Y = Ytrain[index]
        label_n = label[index]

        # Stop condition
        if (depth < self.max_depth - 1) and (len(Y) >= self.min_samples_split) and (len(np.unique(Y)) > 1):
            Gini = None
            split_value = None
            split_feature = None
            features = np.shape(X)[1]

            for feature in range(features):
                feat_X = X[:, feature]
                feat_values = np.unique(feat_X)

                for feat_val in feat_X:
                    left_index = list(feat_X <= feat_val)
                    right_index = list(feat_X > feat_val)  #~left_index

                    left_Gini = gini(Y[left_index])
                    right_Gini = gini(Y[right_index])
                    Gini_i = np.sum((feat_X <= feat_val) + 0) * left_Gini + np.sum(
                        (feat_X > feat_val) + 0) * right_Gini
                    Gini_i = Gini_i / len(Y)

                    if Gini is None or Gini_i < Gini:
                        Gini = Gini_i
                        split_value = feat_val
                        split_feature = feature
                        left_index_n = left_index
                        right_index_n = right_index

            node = Node(split_feature, split_value, loss=self.loss, data_index=index, depth=depth)

            new_left_index = update_index(index, left_index_n)
            new_right_index = update_index(index, right_index_n)

            node.left_child = self.fit(Xtrain, Ytrain, label, new_left_index, depth + 1)
            node.right_child = self.fit(Xtrain, Ytrain, label, new_right_index, depth + 1)

            return node
        else:
            node = Node(loss=self.loss, data_index=index, depth=depth)
            node.update_predict_value(Y, label_n)
            self.leaf_nodes.append(node)
            return node


def calcGini(Y):
    classes = np.unique(Y)
    Gini = 0
    for label in classes:
        p = np.sum(Y[Y == label]) / len(Y)
        Gini += p * p
    return 1 - Gini

# Maybe faster?
def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    if len(array) == 0:
        return 0.0

    array = array.flatten()  #all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array)  #values cannot be negative

    array += 0.0000001  #values cannot be 0
    array = np.sort(array)  #values must be sorted
    index = np.arange(1, array.shape[0] + 1)  #index per array element
    n = array.shape[0]  #number of array elements

    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))  #Gini coefficient


def update_index(index, partial_index):
    new_index = []
    for i in index:
        if i:
            if partial_index[0]:
                new_index.append(True)
                del partial_index[0]
            else:
                new_index.append(False)
                del partial_index[0]
        else:
            new_index.append(False)
    return new_index