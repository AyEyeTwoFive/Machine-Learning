"""
Keep model implementations in here.

This file is where you will write most of your code!
"""

import numpy as np

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
                break
        return allZeroVar

class RegressionTree(object):
    def __init__(self, nfeatures, max_depth):
        self.num_input_features = nfeatures
        self.max_depth = max_depth
        self.nodes = []

    #@profile
    def fit(self, X, y):
    #def fit(self, *, X, y):
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
            if (i.leaf == 0 and i.left == None and i.right == None): # node is not a leaf but no children
                d, t = self.optimizeSplit(i.X, i.y)
                i.d = d
                i.t = t
                L, R, yL, yR = self.split(i.X, i.y, d, t)
                left = RegTreeNode(L, yL, i.depth + 1)
                right = RegTreeNode(R, yR, i.depth + 1)
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

    #@profile
    def computeSSE(self, yL, yR):
        """ Compute  residual sum of squared error
                        Args:
                        L: left group
                        R: right group
                        y: An array of floats with shape [num_examples].
        """
        if (len(yL) > 0):
            muL = np.mean(yL)
        else:
            muL = 0
        if (len(yR) > 0):
            muR = np.mean(yR)
        else:
            muR = 0
        sse = sum([(x - muL)**2 for x in yL]) + sum([(x - muR)**2 for x in yR])
        return sse

    #@profile
    def split(self, X, y, d, t):
        """ Get the split indices
                                Args:
                                X: A of floats with shape [num_examples, num_features].
                                d: A feature dimension.
                                t: threshold
        """
        # L = X[:, d] < t
        # return X[L], X[~L], y[L], y[~L]
        L = []
        R = []
        yL = []
        yR = []
        for i in range(X.shape[0]):
            if X[i, d] < t:
                L.append(i)
                yL.append(y[i])
            else:
                R.append(i)
                yR.append(y[i])
        return X[L, :], X[R, :], yL, yR

    def splitSorted(self, X, y, d, t):
        ind = np.where(X[:,d] == t)[0][0]
        return X[0:ind, :], X[ind:X.shape[0]], y[0:ind], y[ind:X.shape[0]]

    #@profile
    def optimizeSplit(self, X,y):
        """ Find d and t for best split
                        Args:
                        X: A of floats with shape [num_examples, num_features].
                        y: An array of floats with shape [num_examples].
                        max_depth: An int representing the maximum depth of the tree
        """
        minSSE = float('inf')
        for d in range(self.num_input_features):
            XCheck = np.unique(X[:,d])
            for t in XCheck:
                X_L, X_R, yL, yR = self.split(X, y, d, t)
                sse = self.computeSSE(yL, yR)
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
        self.F0 = None
        self.F = None
        self.h = None

    def fit(self, X, y):
    #def fit(self, *, X, y):
        """ Fit the model.
                Args:
                X: A of floats with shape [num_examples, num_features].
                y: An array of floats with shape [num_examples].
                max_depth: An int representing the maximum depth of the tree
                n_estimators: An int representing the number of regression trees to iteratively fit
        """
        # TODO: Implement this!
        self.F0 = np.mean(y)
        F = np.zeros((X.shape[0], self.n_estimators + 1))
        F[:,0] = self.F0 * np.ones(X.shape[0])
        h = []
        for i in range(1, self.n_estimators + 1):
            g = []
            for j in range(X.shape[0]):
                g.append(y[j] - F[j,i - 1])
            h.append(RegressionTree(self.num_input_features, self.max_depth))
            h[i - 1].fit(X=X, y=g)
            hpred = h[i - 1].predict(X=X)
            print("Iteration:", i)
            for k in range(X.shape[0]):
                F[k,i] = F[k, i - 1] + self.regularization_parameter * hpred[k]
        self.F = F
        self.h = h

    def predict(self, X):
        """ Predict.
        Args:
                X: A  matrix of floats with shape [num_examples, num_features].

        Returns:
                An array of floats with shape [num_examples].
        """
        # TODO: Implement this!
        y_hat = self.F0 * np.ones(X.shape[0])
        for j in range(self.n_estimators):
            hpred = self.h[j].predict(X)
            for i in range(X.shape[0]):
                y_hat[i] += self.regularization_parameter * hpred[i]
        return y_hat
