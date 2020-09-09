#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 12:00:48 2019

@author:
"""

import numpy as np
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression_multiclass(object):

    def __init__(self, learning_rate, max_iter, k):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.k = k

    def fit_BGD(self, X, labels, batch_size):
        """Train perceptron model on data (X,y) with BGD.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,].  Only contains 0,..,k-1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.

        Hint: the labels should be converted to one-hot vectors, for example: 1----> [0,1,0]; 2---->[0,0,1].
        """

		### YOUR CODE HERE
        import pandas as pd

        n_samples, n_features = X.shape
        y = pd.get_dummies(labels).values

        self.W = np.zeros((n_features, self.k))

        for _ in range(self.max_iter):
            for i in range(0, n_samples//batch_size):
                grad = 0
                for j in range(i * batch_size, (i+1) * batch_size):
                    if j >= n_samples:
                        break
                    grad += self._gradient(X[j],y[j])

                grad = grad/batch_size
                # print("LRM gradients:",grad)
                self.W -= self.learning_rate * grad

		### END YOUR CODE

    def fit_BGD_Convergence(self, X, labels, batch_size):
        """Train perceptron model on data (X,y) with BGD untill convergence.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,].  Only contains 0,..,k-1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.

        Hint: the labels should be converted to one-hot vectors, for example: 1----> [0,1,0]; 2---->[0,0,1].
        """

		### YOUR CODE HERE
        import pandas as pd

        n_samples, n_features = X.shape
        y = pd.get_dummies(labels).values

        self.W = np.zeros((n_features, self.k))

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


    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: One_hot vector.

        Returns:
            _g: An array of shape [n_features,]. The gradient of
                cross-entropy with respect to self.W.
        """
		### YOUR CODE HERE
        h = self.softmax(np.dot(_x, self.W))
        grad = _x.reshape(-1,1) * (h - _y).reshape(1,-1)
        return grad
		### END YOUR CODE

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        ### You must implement softmax by youself, otherwise you will not get credits for this part.

		### YOUR CODE HERE
        x = x - np.max(x)
        x_softmax = (np.exp(x) / np.sum(np.exp(x))).T
        return x_softmax
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


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 0,..,k-1.
        """
		### YOUR CODE HERE
        pred_one_hot = np.dot(X,self.W)
        pred = np.argmax(pred_one_hot,axis=1)
        return pred
		### END YOUR CODE


    def score(self, X, labels):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,]. Only contains 0,..,k-1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. labels.
        """
		### YOUR CODE HERE
        pred_y = self.predict(X)
        acc = sum(pred_y == labels) / labels.shape[0]
        return acc
		### END YOUR CODE

