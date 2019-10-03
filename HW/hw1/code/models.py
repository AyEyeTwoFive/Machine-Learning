""" 
Keep model implementations in here.

This file is where you will write most of your code!
"""

import numpy as np


class Model(object):
    """ Abstract model object.

    Contains a helper function which can help with some of our datasets.
    """

    def __init__(self, nfeatures):
        self.num_input_features = nfeatures

    def fit(self, *, X, y, lr):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            y: A dense array of ints with shape [num_examples].
            lr: A float, the learning rate of this fit step.
        """
        raise NotImplementedError()

    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()

    def _fix_test_feats(self, X):
        """ Fixes some feature disparities between datasets.
        Call this before you perform inference to make sure your X features
        match your weights.
        """
        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]
        return X


class MCModel(Model):
    """ A multiclass model abstraction.
    It wants to know, up front:
        - How many features in the data
        - How many classes in the data
    """

    def __init__(self, *, nfeatures, nclasses):
        super().__init__(nfeatures)
        self.num_classes = nclasses


class MCPerceptron(MCModel):

    def __init__(self, *, nfeatures, nclasses):
        super().__init__(nfeatures=nfeatures, nclasses=nclasses)
        self.W = np.zeros((nclasses, nfeatures), dtype=np.float)

    def fit(self, *, X, y, lr):
        # TODO: Implement this!
        y_hat = np.zeros(X.shape[0])
        for i in range(X.shape[0]): #for every row (example) in X
            score = []
            for j in range(self.W.shape[0]): # for every w vector
                score.append(self.W[j] @ np.transpose(X[i])) # compute scores
            y_hat[i] = int(np.argmax(score))
            if (y_hat[i]) != y[i]: # update w
                self.W[int(y_hat[i])] = self.W[int(y_hat[i])] - (lr * X[i])
                self.W[y[i]] = self.W[y[i]] + (lr * X[i])


    def predict(self, X):
        X = self._fix_test_feats(X)
        # TODO: Implement this!
        y_hat = np.zeros(X.shape[0])
        for i in range(X.shape[0]):  # for every row (example) in X
            score = []
            for j in range(self.W.shape[0]):  # for every w vector
                score.append(self.W[j] @ np.transpose(X[i]))  # compute scores
            y_hat[i] = int(np.argmax(score))
        return y_hat

    def score(self, X):
        return self.W[1] @ np.transpose(X)


class MCLogistic(MCModel):

    def __init__(self, *, nfeatures, nclasses):
        super().__init__(nfeatures=nfeatures, nclasses=nclasses)
        self.W = np.zeros((nclasses, nfeatures), dtype=np.float)

    def fit(self, *, X, y, lr):
        # TODO: Implement this!
        for i in range(X.shape[0]):  # for every row (example) in X
            logits = self.W @ np.transpose(X[i, :]) # compute logits
            softmaxes = self.softmax(logits) # apply softmax
            for j in range(self.W.shape[0]):  # for every w vector
                if j == y[i]:
                    self.W[j, :] += lr * (1 - softmaxes[j]) * X[i, :]
                else:
                    self.W[j, :] -= lr * softmaxes[j] * X[i, :]

    def predict(self, X):
        X = self._fix_test_feats(X)
        # TODO: Implement this!
        y_hat = np.zeros(X.shape[0])
        for i in range(X.shape[0]): # for every row (example) in X
            score = self.W @ np.transpose(X[i, :]) # compute scores for W vectors
            y_hat[i] = np.argmax(score) #find class with highest score
        return y_hat

    def softmax(self, logits):
        # TODO: Implement this!
        g_star = max(logits) # get g_star
        logits -= g_star
        softmax = np.exp(logits) # exponentiate logits
        softmax /= sum(softmax) # divide by sum
        return softmax

    def score(self, X):
        return self.W[1] @ np.transpose(X)



class OneVsAll(Model):

    def __init__(self, *, nfeatures, nclasses, model_class):
        super().__init__(nfeatures)
        self.num_classes = nclasses
        self.model_class = model_class
        self.models = [model_class(nfeatures=nfeatures, nclasses=2) for _ in range(nclasses)]

    def fit(self, *, X, y, lr):
        # TODO: Implement this!
        for i in range(self.num_classes):
            ynew = y.copy()
            for j in range(len(ynew)):
                if ynew[j] == i:
                    ynew[j] = 1
                else:
                    ynew[j] = 0

            self.models[i].fit(X=X, y=ynew, lr=lr)

        # raise Exception("You must implement this method!")


    def predict(self, X):
        X = self._fix_test_feats(X)
        # TODO: Implement this!
        y_hat = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            scores = np.zeros(self.num_classes)
            for j in range(self.num_classes):
                scores[j] = self.models[j].score(X[i])

            y_hat[i] = np.argmax(scores)

        return y_hat


