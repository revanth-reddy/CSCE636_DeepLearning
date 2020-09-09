import numpy as np
import sys
import math

"""This script implements a two-class logistic regression model.
"""

class logistic_regression(object):

    def __init__(self, learning_rate, max_iter):
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit_GD(self, X, y):
        """Train perceptron model on data (X,y) with GD.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
        n_samples, n_features = X.shape

		### YOUR CODE HERE
        self.W = np.zeros(n_features)

        for _ in range(self.max_iter):
            grad = 0
            for i in range(0,n_samples):
                grad += self._gradient(X[i],y[i])

            grad = grad/n_samples
            self.W -= self.learning_rate * grad

		### END YOUR CODE
        return self

    def fit_BGD(self, X, y, batch_size):
        """Train perceptron model on data (X,y) with BGD.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)
        for _ in range(self.max_iter):
            for i in range(0, n_samples//batch_size):
                grad = 0
                for j in range(i * batch_size, (i+1) * batch_size):
                    if j >= n_samples:
                        break
                    grad += self._gradient(X[j],y[j])

                grad = grad/batch_size
                # print("LR gradients:",grad)
                self.W -= self.learning_rate * grad

		### END YOUR CODE
        return self

    def fit_BGD_Convergence(self, X, y, batch_size):
        """Train perceptron model on data (X,y) with BGD.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)
        for _ in range(self.max_iter):
            for i in range(0, n_samples//batch_size):
                grad = 0
                for j in range(i * batch_size, (i+1) * batch_size):
                    if j >= n_samples:
                        break
                    grad += self._gradient(X[j],y[j])

                grad = grad/batch_size
                if np.linalg.norm(grad*1./batch_size) < 0.0005:
                    return
                self.W -= self.learning_rate * grad

		### END YOUR CODE
        return self

    def fit_SGD(self, X, y):
        """Train perceptron model on data (X,y) with SGD.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)
        for _ in range(self.max_iter):
            for i in range(0,n_samples):
                grad = self._gradient(X[i],y[i])
                self.W -= self.learning_rate * grad
		### END YOUR CODE
        return self

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: An integer. 1 or -1.

        Returns:
            _g: An array of shape [n_features,]. The gradient of
                cross-entropy with respect to self.W.
        """
		### YOUR CODE HERE
        _g = - _y * _x / ( 1 + math.exp(_y * np.dot(self.W,_x)))
        return _g
		### END YOUR CODE

    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features,].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W

    def predict_proba(self, X):
        """Predict class probabilities for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds_proba: An array of shape [n_samples, 2].
                Only contains floats between [0,1].
        """
		### YOUR CODE HERE

        z = np.dot(X, self.W)

        def sigmoid(p):
            return 1/(1+ math.exp(-p))

        ans = np.vectorize(sigmoid)(z)
        return np.concatenate((np.reshape(ans,(-1,1)), np.reshape(1-ans,(-1,1))), axis = 1)

		### END YOUR CODE


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 1 or -1.
        """
		### YOUR CODE HERE
        z = np.dot(X, self.W)

        def sigmoid(p):
            return 1/(1+ math.exp(-p))

        ans = np.vectorize(sigmoid)(z)

        ans[ans>=0.5] = 1
        ans[ans<0.5] = -1

        return ans
		### END YOUR CODE

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. y.
        """
		### YOUR CODE HERE
        pred_y = self.predict(X)
        acc = sum(pred_y == y) / y.shape[0]
        return acc
		### END YOUR CODE

    def assign_weights(self, weights):
        self.W = weights
        return self

