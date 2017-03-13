# coding: utf-8
"""
Description:
-----------
    The implementation of back-propagation neural networks(BPNN)

Including:
-----------
    1. basic multi-layer perceptron
    2. training with batch gradient descent(GD)
    3. L2 weights constrain is supported

Usage:
-----------
    # import sklearn.datasets as dt
    # from sklearn.preprocessing import normalize
    # from sklearn.cross_validation import train_test_split
    # from sklearn.metrics import accuracy_score
    # iris = dt.load_iris()
    # X, y = iris.get('data'), iris.get('target') + 1 # start with 0
    # X = normalize(X)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6)
    # bpnn_1 = BPNN([4, 20, 3])
    # bpnn_1.fit(X_train, y_train, learn_rate=0.05, max_iter=1000, batch=10, min_error=0, regular=None)
    # labels_1 = bpnn_1.predict(X_test)
    # print accuracy_score(labels_1, y_test)
"""

import numpy as np
from numpy import exp
import math

class BPNN:
    parameters = []
    X = np.empty(0)
    y = np.empty(0)
    W = np.empty(0)
    bias = np.empty(0)
    y_expected = np.empty(0)
    __nclass,__nfeature, __nsample = 0,0,0
    classes = np.empty(0)
    total_error = []

    def __init__(self, parameters):
        self.parameters = parameters

    def __init_network(self):
        # initialize network by setting initial weights and biases
        self.__nsample, self.nfeature = self.X.shape
        self.classes = np.unique(self.y)
        self.nclass = self.classes.shape[0]
        # set the initial weights for networks
        w, b = [], []
        for layer in range(len(self.parameters) - 1):
            w.append(np.random.uniform(-1., 1., size=(self.parameters[layer], self.parameters[layer + 1])))
            b.append(np.random.uniform(-1., 1., self.parameters[layer + 1]))
        self.W, self.bias = np.array(w), np.array(b)
        # convert labels to 0/1 sequence
        e = np.eye(self.nclass)
        self.y_expected = np.zeros((self.__nsample, self.nclass))
        for label in self.classes:
            indices = np.nonzero(self.y == label)
            self.y_expected[indices] = e[np.nonzero(self.classes == label)]

    def __sigmoid(self, x):
        """
        vectorized sigmoid function
        :param x: array_like
        :return: array_like
        """
        return 1 / (1 + exp(-x))

    def __feed_forword(self, X, layer=None):
        # calculate nets,Xi(sigmoid) and outputs
        if layer == 0:
            return X
        if layer == None:
            layer = len(self.parameters) - 1
        output = np.empty(0)
        for i in np.arange(layer):
            net = np.dot(X, self.W[i]) + self.bias[i]
            output = self.__sigmoid(net)
            X = output
        return  output

    def __error(self, o_actual, o_calculated):
        return 0.5 * np.sum((o_actual-o_calculated)**2, axis=1)

    def __backprop(self, X, y_excepted, learn_rate, regularization=False):
        """
        train the networks model with back propagation algorithm
        :param X: batch samples
        :param learn_rate:
        :return:
        """
        outputs = self.__feed_forword(X) # N*k
        sum_of_error = (outputs - y_excepted) # N*k
        for layer in np.arange(1, len(self.parameters))[::-1]:
            net_error = sum_of_error * outputs * (1 - outputs) # N*k
            outputs_last_layer = self.__feed_forword(X,layer = layer - 1) # N*j
            w = self.W[layer - 1]
            if regularization is True:
                self.W[layer - 1] = (1 - learn_rate * self.regular) * self.W[layer - 1] - \
                                    learn_rate * np.dot(outputs_last_layer.T, net_error)  # j*N * N*k =j*k
                self.bias[layer - 1] -= learn_rate * np.average(net_error, axis=0)
            else:
                self.W[layer - 1] -= learn_rate * np.dot(outputs_last_layer.T, net_error) # j*N * N*k =j*k
                self.bias[layer - 1] -= learn_rate * np.average(net_error, axis=0)
            outputs = outputs_last_layer
            # B = np.random.random(w.shape).transpose()
            # sum_of_error = np.dot(net_error, B)
            sum_of_error = np.dot(net_error, w.T)

    def fit(self, X, y, learn_rate=0.02, max_iter=10000, batch=None, min_error=0.01, regular=None):
        """
        fit the hyperplane using back propagation with batch gradient decent, to set regular to improv. generalization ability
        :param X:
        :param y:
        :param learn_rate:
        :param max_iter:
        :param batch:
        :param min_error:
        :param regular:None or float
        :return:
        """
        self.X, self.y = X, y
        self.regular = regular
        self.__init_network()
        self.train_error = []
        if batch == None:
            batch = self.__nsample
        for iter in range(max_iter):
            self.train_error.append(np.average(self.__error(self.y_expected, self.__feed_forword(X_train))))
            for b in np.arange(int (math.ceil(self.__nsample / float(batch)))):
                if (b+1)*batch >= self.__nsample - 1:
                    batch_X = self.X[b * batch:]
                    batch_y_excepted = self.y_expected[b * batch : ]
                else:
                    batch_X = self.X[b*batch:(b+1)*batch]
                    batch_y_excepted = self.y_expected[b*batch:(b+1)*batch]
                output = self.__feed_forword(batch_X)
                total_error = np.average(self.__error(batch_y_excepted,output))
                print total_error
                self.total_error.append(total_error)
                if total_error <= min_error:
                    # when error less than the min error stop training
                    print 'iterator = ', iter
                    return
                if regular != None:
                    self.__backprop(batch_X, batch_y_excepted, learn_rate, regularization=True)
                self.__backprop(batch_X, batch_y_excepted, learn_rate)

    def predict(self, X):
        """
        :param X:
        :return:
        """
        output = self.__feed_forword(X)
        index = output.argmax(axis=1)
        labels = np.random.randint(1, 2, X.shape[0])
        for c in np.arange(self.nclass):
            labels[ np.nonzero(index == c)] = self.classes[c]
        return labels

    def get_train_error(self):
        return np.array(self.train_error)

