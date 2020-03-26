import numpy as np 

class Node:
    def __init__(self, logger=None, split_feature=None, split_value=None, data_index=None, 
    is_leaf=False, loss=None, depth=None, feature_importance=None):
        self.loss = loss
        self.is_leaf = is_leaf
        self.max_depth = depth
        self.left_child = None
        self.right_child = None
        self.predict_value = None
        self.data_index = data_index
        self.split_value = split_value
        self.split_feature = split_feature
        self.feature_importance = feature_importance
        #self.logger = logger

    def get_feature_importance(self):
        feature_importances = np.zeros(len(self.feature_importance))
        if self.is_leaf:
            return self.feature_importance
        else:
            feature_importances += self.left_child.get_feature_importance()
            feature_importances += self.right_child.get_feature_importance()
            return self.feature_importance + feature_importances



    def update_predict_value(self, f_m, y):
        self.predict_value = self.loss.update_leaf_values(f_m, y)

    def get_predict_value(self, X):
        if self.is_leaf:
            return self.predict_value
        if X[self.split_feature] <= self.split_value:
            return self.left_child.get_predict_value(X)
        else:
            return self.right_child.get_predict_value(X)


class DecisionTreeClassifier:
    def __init__(self, X=None, Y=None, label=None, index=None, depth=0, logger= None, max_depth=10, min_samples_split=2, loss=None):
        self.loss = loss
        self.leaf_nodes = []
        self.logger = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = self.fit(X, Y, label, index)

    def fit(self, Xtrain, Ytrain, label, index, depth=0):
        X = Xtrain[index]
        Y = Ytrain[index]
        label_n = label[index]

        if (depth < self.max_depth - 1) and (len(Y) >= self.min_samples_split) and (len(np.unique(Y)) > 1):
            Gini = None
            split_value = None
            split_feature = None
            features = np.shape(X)[1]
            feature_importance = []
            Gini_o = calcGini(Y)

            for feature in range(features):
                Gini_feature = None
                feat_X = X[ :, feature]
                # calculate the Gini for each features
                for feat_val in feat_X:
                    left_index = list(feat_X <= feat_val)
                    right_index = list(feat_X > feat_val)

                    left_Gini = calcGini(Y[left_index])
                    right_Gini = calcGini(Y[right_index])
                    Gini_i = np.sum((feat_X <= feat_val)+0)*left_Gini + np.sum((feat_X > feat_val)+0)*right_Gini
                    Gini_i = Gini_i / len(Y)

                    if Gini_feature is None or Gini_i< Gini_feature:
                        Gini_feature = Gini_i

                    if Gini is None or Gini_i < Gini:
                        Gini = Gini_i
                        split_value = feat_val
                        split_feature = feature
                        left_index_n = left_index
                        right_index_n = right_index

                feature_importance.append((Gini_o - Gini_feature))
            feature_importance = np.array(feature_importance)

            node = Node(self.logger, split_feature, split_value, loss=self.loss, data_index=index, depth=depth, feature_importance=feature_importance)

            new_left_index = update_index(index, left_index_n)
            new_right_index = update_index(index, right_index_n)

            node.left_child = self.fit(Xtrain, Ytrain, label, new_left_index, depth+1)
            node.right_child = self.fit(Xtrain, Ytrain, label, new_right_index, depth+1)

            return node
        else:
            feature_importance = np.zeros(np.shape(X)[1])
            node = Node(self.logger, is_leaf=True, loss=self.loss, data_index=index, depth=depth, feature_importance=feature_importance)
            node.update_predict_value(Y, label_n)
            self.leaf_nodes.append(node)
            return node

def calcGini(Y):
    classes = np.unique(Y)
    Gini = 0
    for label in classes:
        p = np.sum((Y == label) + 0) / len(Y)
        Gini += p * p
    Gini = 1 - Gini
    return Gini

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