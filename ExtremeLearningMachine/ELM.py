"""
Description:
    implementation of basic Extreme Learning Machine with sklearn
    including:
        1. basic ELM: HB = T
        2. weighted ELM: arg min || HB - T||2 * W
        ------
        future: L2 constrained ELM: arg min || HB - T||2 s.t. 1/2||B||2

Ref.
    [1]	G. B. Huang, H. Zhou, X. Ding, and R. Zhang, "Extreme Learning Machine for Regression and Multiclass Classification," IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), vol. 42, pp. 513-529, 2012.
    [2]	G.-B. Huang, Q.-Y. Zhu, and C.-K. Siew, "Extreme learning machine: theory and applications," Neurocomputing, vol. 70, pp. 489-501, 2006.

"""

import numpy as np
from numpy import linalg
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import AdaBoostClassifier


class ELM(BaseEstimator, ClassifierMixin):
    """
    Parameters
    ----------
    n_hidden : the number of hidden nodes


    Attributes
    ----------
    upper_bound : upper bound of hidden weight, default 1
    lower_bound : lower bound of hidden weight, default -1
    classes_:
    n_classes_:
    W: hidden weights, initialized randomly
    b: hidden biases, initialized randomly
    B: mathematically computed
    sample_weight: None if default, or N_samples array
    """

    upper_bound = 1
    lower_bound = -1

    def __init__(self, n_hidden, C=1000):
        self.n_hidden = n_hidden
        self.C = C
        # self.constraint = constraint

    def fit(self, X, y, sample_weight=None):
        """

        :param X: training features
        :param y: training class labels.
                   Note: it will be converted to binary matrix if input 1-D array. Label start with 0
        :param sample_weight:
        :return:
        """
        # check label has form of 2-dim array
        self.sample_weight = sample_weight
        if y.shape.__len__() != 2:
            self.classes_ = np.unique(y)
            self.n_classes_ = self.classes_.__len__()
            y = self.one2array(y, self.n_classes_)
        else:
            self.classes_ = np.arange(y.shape[1])
            self.n_classes_ = self.classes_.__len__()
        self.W = np.random.uniform(self.lower_bound, self.upper_bound, size=(X.shape[1], self.n_hidden))
        self.b = np.random.uniform(self.lower_bound, self.upper_bound, size=self.n_hidden)
        H = expit(np.dot(X, self.W) + self.b)
        if sample_weight is not None:
            self.sample_weight = sample_weight / sample_weight.sum()
            extend_sample_weight = np.diag(self.sample_weight)
            inv_ = linalg.pinv(np.dot(
                np.dot(H.transpose(), extend_sample_weight), H))
            self.B = np.dot(np.dot(np.dot(inv_, H.transpose()), extend_sample_weight), y)
        else:
            self.B = np.dot(linalg.pinv(H), y)

    def one2array(self, y, n_dim):
        """
        convert label to binary matrix, like [[0,0,1,0],[0,1,0,0],...]
        :param y:
        :param n_dim: num of classes
        :return:
        """
        y_expected = np.zeros((y.shape[0], n_dim))
        for i in range(y.shape[0]):
            y_expected[i][y[i]] = 1
        return y_expected

    def predict(self, X):
        H = expit(np.dot(X, self.W) + self.b)
        output = np.dot(H, self.B)
        return output.argmax(axis=1)

    def get_params(self, deep=True):
        self.params = {'n_hidden': self.n_hidden, 'C': self.C}
        return self.params

    def set_params(self, **parameters):
        return self


#
# '''
# ----------
# ELM test
# ----------
# '''
# import sklearn.datasets as dt
# from sklearn.preprocessing import normalize
# from sklearn.cross_validation import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn import preprocessing
# iris = dt.load_iris()
# X, y = iris.get('data'), iris.get('target')  # start with 0
# X = normalize(X)
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6)
#
# lb = preprocessing.LabelBinarizer()
# Y_train = lb.fit_transform(y_train)
# # Y_test = lb.fit_transform(y_test)
#
# eml = ELM(10, C=1000)
# eml.fit(X_train, Y_train, sample_weight=None)
# labels = eml.predict(X_test)
#
# print 'Accuracy:', accuracy_score(y_test, labels)
# print 'predicted labels:', labels
# print 'actual labels:', y_test-labels

#
# elm_ab = AdaBoostClassifier(ELM(10), algorithm="SAMME", n_estimators=100)
# elm_ab.fit(X_train, y_train)
# y_pre_elm_ab = elm_ab.predict(X_test)
# print 'AdBoost ELM:', accuracy_score(y_test, y_pre_elm_ab)


# '''
# --------
# compare with HPELM
# --------
# '''
# from hpelm import ELM as HPELM
# model = HPELM(4, 3)
# model.add_neurons(10, "sigm")
# model.train(X_train, Y_train)
# Y_pre = model.predict(X_test)
# y = Y_pre.argmax(axis=1)
# print 'HPELM acc:', accuracy_score(y_test, y)
