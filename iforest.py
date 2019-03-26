# coding: utf-8

# In[87]:


import numpy as np
import random
import time
import pandas as pd
from mpmath import *
from sklearn.metrics import confusion_matrix
from lolviz import *
from scipy.stats import kurtosis


class LeafNode:
    def __init__(self, size):
        self.size = size


class InnerNode:
    def __init__(self, left_child, right_child, split_attribute, split_value, min_split_value, max_split_value):
        self.left_child = left_child
        self.right_child = right_child
        self.split_attribute = split_attribute
        self.split_value = split_value
        self.min_split_value = min_split_value
        self.max_split_value = max_split_value


# # Isolation Tree Class

# In[89]:


class IsolationTree:
    def __init__(self, height_limit=0):
        self.height_limit = height_limit
        self.n_nodes = 0
        self.root = None

    def growTree(self, X: np.ndarray, height_limit, current_tree_height, min_value = None, max_value=0):

        if current_tree_height >= height_limit or len(X) <= 1 or (min_value == max_value):  # reach the limit (or if run out of data)
            self.n_nodes += 1
            return LeafNode(size=len(X))

        else:
            split_attribute = np.random.randint(0, self.num_attributes)  # inclusive and exclusive
            column = X[:, split_attribute]
            min_value = column.min()
            max_value = column.max()
            mean_value = column.mean()
            #sd = np.std(column)
            #print("min: {}, max: {}, mean: {}, sd: {}".format(min_value, max_value, mean_value, sd))
            split_value = np.random.uniform(mean_value, max_value)
            #split_value = np.random.normal(mean_value, sd, 1)
            left_index = X[:, split_attribute] < split_value
            right_index = np.invert(left_index)
            left_data = X[left_index]
            right_data = X[right_index]

            self.n_nodes += 1
            return InnerNode(left_child=self.growTree(X=left_data, height_limit=height_limit,
                                                      current_tree_height=current_tree_height + 1, min_value=min_value, max_value=max_value),
                             right_child=self.growTree(X=right_data, height_limit=height_limit,
                                                       current_tree_height=current_tree_height + 1, min_value=min_value, max_value=max_value),
                             split_attribute=split_attribute,
                             split_value=split_value,
                             min_split_value=min_value,
                             max_split_value=max_value)

    def fit(self, X: np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.
        """
        self.num_attributes = len(X[0])
        self.root = self.growTree(X, height_limit=np.ceil(np.log2(256)), current_tree_height=0)
        return self


# # Outside Functions

# In[90]:

def preprocess_data(X):
    blacklist = []
    mean_kurtosis = np.mean(kurtosis(X))
    for header in list(X):

        if 'noise' in header:
            blacklist.append(header)
        else:
            if kurtosis(X[header]) < mean_kurtosis:
                blacklist.append(header)
            #else: print("{}: {:.5}".format(header, kurtosis(X[header])))
    X  = X.drop(blacklist, axis=1)
    return X

def walk(row, node):
    current_edges = 0
    while isinstance(node, LeafNode) == False:
        current_edges += 1
        if row[node.split_attribute] < node.split_value:
            node = node.left_child
        else:
            node = node.right_child
    else:  # is leaf
        current_edges = current_edges + compute_c(node.size)
        return current_edges


def compute_c(sample_size):
    if sample_size > 2:
        return 2 * (np.log(sample_size - 1) + 0.5772156649) - 2 * ((sample_size - 1) / sample_size)
    elif sample_size == 2:
        return 1
    else:
        return 0


# # Isolation Tree Ensemble Class

# In[91]:


class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=10):
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.c = compute_c(sample_size)

    def fit(self, X: np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        self.trees = []
        X = preprocess_data(X)
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.X = X  # to keep it in pandas version (odd)

        for i in range(self.n_trees):
            X_sample = X[np.random.randint(low=0, high=len(X), size=self.sample_size), :]  # take a sample
            iTree = IsolationTree(height_limit=np.ceil(np.log2(256))).fit(X_sample)
            self.trees.append(iTree)

        return self  # do we really need to return anything?

    def path_length(self, X: np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """
        avg_path_lengths = []
        for x_i in self.X:

            xi_lengths = []
            for t in self.trees:
                xi_lengths.append(walk(x_i, t.root))
            avg_path_lengths.append(np.average(xi_lengths))

        self.avg_path_lengths = avg_path_lengths
        return avg_path_lengths

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        h = self.path_length(X)  # returns a vector of size num_rows of X
        score = 2 ** -(h / self.c)  # divide each by c, the normalizing constant
        return score

    def predict_from_anomaly_scores(self, scores: np.ndarray, threshold: float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        return [1 if score >= threshold else 0 for score in scores]

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        "A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."
        return self.predict_from_anomaly_scores(self.anomaly_score(X), threshold)


# # Find TPR Threshold

# In[92]:
def find_TPR_threshold(y, scores, desired_TPR):
    """
    Start at score threshold 1.0 and work down until we hit desired TPR.
    Step by 0.01 score increments. For each threshold, compute the TPR
    and FPR to see if we've reached to the desired TPR. If so, return the
    score threshold and FPR.
    """
    for threshold in [i / 100 for i in range(100, 0, -1)]:  # needs this funky thing coz range only takes ints. Step from 10 to zero in steps of -1
        # google vectorize
        prediction = [1 if score >= threshold else 0 for score in scores]  # get 1 if score >= threshold
        TN, FP, FN, TP = confusion_matrix(y, prediction, labels=[0, 1]).flat  # extract confusion matrix
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        if TPR >= desired_TPR: return threshold, FPR