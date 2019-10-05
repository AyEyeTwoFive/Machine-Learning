"""
Keep model implementations in here.

This file is where you will write most of your code!
"""

import numpy as np
import cProfile

class RegTreeNode(object):
    def __init__(self, X, y, depth):
        self.X = X
        self.y = y
        self.d = 0
        self.t = 0
        self.depth = depth
        self.parent = None
        self.left = None
        self.right = None
        self.leaf = 0
        self.pred = 0

    def zeroVariance(self):
        allZeroVar = 1
        for i in range(self.X.shape[1]):
            if (np.var(self.X[:, i]) != 0):
                allZeroVar = 0
        return allZeroVar

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
        root = RegTreeNode(X, y, 0)
        if (root.depth > self.max_depth or root.X.shape[0] <= 1 or root.zeroVariance()):
            root.leaf = 1
            root.pred = np.mean(root.y)
        self.nodes.append(root)
        for i in self.nodes:
            if i.leaf == 0 and i.left == None and i.right == None: # node is not a leaf but no children
                d, t = self.optimizeSplit(i.X, i.y)
                i.d = d
                i.t = t
                L, R, yL, yR = self.split(i.X, i.y, d, t)
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
                self.nodes.append(left)
                self.nodes.append(right)


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
        if (len(L) > 0):
            muL = sum_y_L / len(L)
        else:
            muL = 0
        if (len(R) > 0):
            muR = sum_y_R / len(R)
        else:
            muR = 0
        sse = 0
        for i in range(len(y)):
            if i in L:
                sse += (y[i] - muL)**2
            if i in R:
                sse += (y[i] - muR)**2
        return sse

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
                L.append(i)
                yL.append(y[i])
            else:
                R.append(i)
                yR.append(y[i])
        return L, R, yL, yR

    def optimizeSplit(self, X,y):
        """ Find d and t for best split
                        Args:
                        X: A of floats with shape [num_examples, num_features].
                        y: An array of floats with shape [num_examples].
                        max_depth: An int representing the maximum depth of the tree
        """
        minSSE = float('inf')
        tCheck = []
        for d in range(self.num_input_features):
            for i in range(X.shape[0]):
                if ((d, i) not in tCheck):
                    tCheck.append((d, i))
                    t = X[i][d]
                    L,R, yL, yR = self.split(X, y, d, t)
                    sse = self.computeSSE(L,R,y)
                    if sse < minSSE:
                        minSSE = sse
                        dstar = d
                        tstar = t
        return dstar, tstar

    def predict(self, X):
        """ Predict.
        Args:
                X: A  matrix of floats with shape [num_examples, num_features].

        Returns:
                An array of floats with shape [num_examples].
        """
        # TODO: Implement this!
        y_hat = []
        for i in range(X.shape[0]):
            node = self.nodes[0]
            while (node.leaf == 0):
                if (X[i, node.d] < node.t):
                    node = node.left
                else:
                    node = node.right
            y_hat.append(node.pred)
        return y_hat




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
        F = []
        F0 = np.mean(y)
        F = F0 * np.zeros(X.shape[0])
        for i in range(n_estimators):
            g = []
            for j in range(X.shape[0]):
                g.append(y[j] - F)


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
