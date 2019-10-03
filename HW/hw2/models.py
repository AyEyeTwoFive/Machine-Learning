"""
Keep model implementations in here.

This file is where you will write most of your code!
"""

import numpy as np

class RegressionTree(object):
    def __init__(self, nfeatures, max_depth):
        self.num_input_features = nfeatures
        self.max_depth = max_depth
        self.nodes = []

    def fit(self, *, X, y):
        """ Fit the model.
                Args:
                X: A of floats with shape [num_examples, num_features].
                y: An array of floats with shape [num_examples].
                max_depth: An int representing the maximum depth of the tree
        """

        # TODO: Implement this!
        self.nodes.append(RegTreeNode(X, y, 0))
        for i in self.nodes:
            if i.leaf == 0: # node is not a leaf
                d, t = optimizeSplit(i.X, i.y)
                L, R, yL, yR = split(i.X, i.y, d, t)
                left = RegTreeNode(i.X[L, :], yL, i.depth + 1)
                right = RegTreeNode(i.X[R, :], yR, i.depth + 1)
                left.parent = i
                right.parent = i
                i.left = left
                i.right = right
                if (left.depth > self.max_depth or left.X.shape[0] <= 1 or left.zeroVariance()):
                    left.leaf = 1
                    left.pred = np.mean(left.y)
                if (right.depth > self.max_depth or right.X.shape[0] <= 1 or right.zeroVariance()):
                    right.leaf = 1
                    right.pred = np.mean(right.y)

        raise Exception("You must implement this method!")

    class RegTreeNode(object):
        def __init__(self, X, y, depth):
            self.X = X
            self.y = y
            self.depth = depth
            self.parent = None
            self.left = None
            self.right = None
            self.leaf = 0
            self.pred = 0

        def zeroVariance(self):
            allZeroVar = 1
            for i in self.X.shape[1]:
                if (np.var(X[:,i]) == 0):
                    allZeroVar = 0
            return allZeroVar

        # def setParent(self, obj):
        #     self.parent = obj
        #
        # def setLeft(self, obj):
        #     self.left = obj
        #
        # def setRight(self, obj):
        #     self.right = obj

    def computeSSE(self, L, R, y):
        """ Compute  residual sum of squared error
                        Args:
                        L: left group
                        R: right group
                        y: An array of floats with shape [num_examples].
        """
        sum_y_L = 0
        sum_y_R = 0
        for i in range(len(y)):
            if i in L:
                sum_y_L += y[i]
            if i in R:
                sum_y_R += y[i]
        muL = sum_y_L / len(L)
        muR = sum_y_R / len(R)

        sse = 0
        for i in range(len(y)):
            if i in L:
                sse += (y[i] - muL)**2
            if i in R:
                sse += (y[i] - muR)**2
        return SSE

    def split(self, X, y, d, t):
        """ Get the split indices
                                Args:
                                X: A of floats with shape [num_examples, num_features].
                                d: A feature dimension.
                                t: threshold
        """
        L = []
        R = []
        yL = []
        yR = []
        for i in range(X.shape[0]):
            if X[i,d] < t:
                L.append[i]
                yL.append(y[i])
            else:
                R.append[i]
                yR.append(y[i])
        return L, R, yL, yR

    def optimizeSplit(self, X,y):
        """ Find d and t for best split
                        Args:
                        X: A of floats with shape [num_examples, num_features].
                        y: An array of floats with shape [num_examples].
                        max_depth: An int representing the maximum depth of the tree
        """
        minSSE = 99999999
        dstar = 0
        tstar = 0
        Lstar = []
        Rstar = []
        for d in range(self.num_input_features):
            for i in range(X.shape[0]):
                t = X[i][d]
                L,R, yL, yR = split(X, y, d, t)
                sse = computeSSE(L,R,y)
                if sse < minSSE:
                    minSSE = sse
                    dstar = d
                    tstar = t
                    Lstar = L
                    Rstar = R

        return dstar, tstar

    def predict(self, X):
        """ Predict.
        Args:
                X: A  matrix of floats with shape [num_examples, num_features].

        Returns:
                An array of floats with shape [num_examples].
        """
        # TODO: Implement this!
        raise Exception("You must implement this method!")



class GradientBoostedRegressionTree(object):
    def __init__(self, nfeatures, max_depth, n_estimators, regularization_parameter):
        self.num_input_features = nfeatures
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.regularization_parameter = regularization_parameter
    def fit(self, *, X, y):
        """ Fit the model.
                Args:
                X: A of floats with shape [num_examples, num_features].
                y: An array of floats with shape [num_examples].
                max_depth: An int representing the maximum depth of the tree
                n_estimators: An int representing the number of regression trees to iteratively fit
        """
        # TODO: Implement this!
        raise Exception("You must implement this method!")

    def predict(self, X):
        """ Predict.
        Args:
                X: A  matrix of floats with shape [num_examples, num_features].

        Returns:
                An array of floats with shape [num_examples].
        """
        # TODO: Implement this!
        raise Exception("You must implement this method!")
