# coding: utf-8
"""
The implementation of back-propagation neural networks(BPNN)
1. signal feed forward;
2. error feed backward.
"""
import numpy as np
from numpy import sqrt, exp

class BPNN:
    nlayer,nhidden = 2,2
    X = np.empty(0)
    y = np.empty(0)
    W = np.empty(0)
    bias = np.empty(0)
    y_expected = np.empty(0)
    __nclass,__nfeature = 0,0
    total_error = []
    def __init__(self,nlayer=2,nhidden=3):
        self.nlayer,self.nhidden=nlayer,nhidden

    def __init_network(self):
        # initialize network by setting initial weights and biases
        nsample,self.nfeature = self.X.shape
        self.nclass = np.unique(self.y).shape[0]
        # set the initial weights for networks
        self.W = np.array([ np.random.uniform(-0.1,0.1,size=(self.nfeature,self.nhidden)),# i (input)rows j (hidden)colums
                np.random.uniform(-0.1,0.1,size=(self.nhidden,self.nclass) ) ] ) # j (hidden)rows,k (output) columns

        # convert labels to 0/1 sequence
        e = np.eye(self.nclass)
        self.y_expected = np.zeros((nsample, self.nclass))
        for i in range(self.nclass):
            indices = np.nonzero(self.y == i+1)
            self.y_expected[indices] = e[i]

        # initialize biases
        self.bias = np.array([np.random.rand(self.nhidden),np.random.rand(self.nclass)])
        # print self.bias

    def __sigmoid(self, x):
        """
        vectoring the sigmoid function
        :param x: array_like
        :return: array_like
        """
        return 1 / (1 + exp(-x))

    def __feed_forword(self,X,layer=2):
        # calculate nets,Xi(sigmoid) and outputs
        output = np.empty(0)
        for i in range(layer):
            net = np.dot(X, self.W[i]) + self.bias[i]
            output = self.__sigmoid(net)
            X = output
        return  output

    def __error(self,o_actual,o_calculated):
        return np.sum((o_actual-o_calculated)**2, axis=1)

    def __backprop(self,rate):
        for sample in range(self.y.shape[0]):
            # update the weights for hidden-output layer
            theta_output = [] # the error of output layer
            o_actual = self.__feed_forword(self.X)
            for k in range(self.nclass):
                theta_output.append((o_actual[sample][k] - self.y_expected[sample][k]) * o_actual[sample][k] * (1 - o_actual[sample][k]))
                self.bias[1][k] -= rate * theta_output[k] # update the bias for hidden-output layer
            for j in range(self.nhidden):
                for k in range(self.nclass):
                    self.W[1][j][k] -= rate * theta_output[k] * self.X[sample][k]
            # get error of hidden layer
            theta_hidden = []
            output_hidden = self.__feed_forword(self.X)
            for j in range(self.nhidden):
                theta_arr = np.array(theta_output)
                theta_hidden.append(((theta_arr * self.W[1][j])* output_hidden[sample]*(1-output_hidden[sample])).sum())
                self.bias[0][j] = theta_hidden[j] # update the bias for input-hidden layer

            for i in range(self.nfeature):
                for j in range(self.nhidden):
                    self.W[0][i][j] = theta_hidden[j] * self.X[sample][i]

    def fit(self,X,y,learn_rate = 0.02,iters = 2,min_error = 0.01):
        self.X, self.y = X, y
        self.__init_network()
        for iter in range(iters):
            output = self.__feed_forword(self.X)
            total_error = 0.5*np.sum(self.__error(self.y_expected,output))/self.y.shape[0]
            self.total_error.append(total_error)
            if total_error <= min_error:
                # when error less than the min error stop training
                break
            # print self.W,'\n\n'
            self.__backprop(learn_rate)

    def predict(self,X):
        output = self.__feed_forword(X)
        labels = output.argmax(axis=1)
        return labels+1


import sklearn.datasets as dt
from sklearn.preprocessing import normalize
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
iris = dt.load_iris()
X, y = iris.get('data'),iris.get('target') + 1 # start with 0
X = normalize(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

bpnn = BPNN(nhidden=2)
bpnn.fit(X_train,y_train,iters=100)
labels = bpnn.predict(X_test)

print accuracy_score(y_test,labels)
print 'predicted labels:',labels
print 'actual labels:',y_test

print bpnn.total_error
# print bpnn.W

