# coding: utf-8

"""
Description:
    1.the implementation of kernel-based k-nearest-neighbors algorithm.
    2.focus on processing multiple features classification using weighted composite kernel function.
Requirement:
    python 2.7
    numpy
    scipy
    sklearn-1.8
"""
import numpy as np
from sklearn.preprocessing import normalize
class KNeighbors:

    def __init__(self, weights, k=3, gamma=1):
        self.k = k
        self.weights = np.asarray(weights)
        self.gamma = gamma

    def fit(self, X_arr, y):
        self.X_arr = X_arr
        self.y = y

    def predict(self, X_arr):
        """
        conventional kNN prediction that use weighted summation distance
        :param X_arr:
        :return:
        """
        from sklearn.neighbors import DistanceMetric as DM
        dis = DM.get_metric('euclidean')
        distances = []
        for i in range(X_arr.__len__()):
            X_arr[i], self.X_arr[i] = normalize(X_arr[i]), normalize(self.X_arr[i])  # force convert into range
            distances.append(dis.pairwise(X_arr[i], self.X_arr[i]))
        distances = np.array(distances)
        multi_dis = np.zeros((distances.shape[1], distances.shape[2]))
        for w, d in zip(self.weights, distances):
            multi_dis += w * d  # weighted distances
        sorted_ss_indicies = multi_dis.argsort()  # sort dis
        k_neighbors_lables = self.y[sorted_ss_indicies][:, :self.k]
        from scipy.stats import mode
        y_predicted, t = mode(k_neighbors_lables, axis=1)
        return y_predicted.reshape(-1)

    def predict_stack_vector(self, X_arr):
        """
        calculate distance using stacked feature vectors
        :param X_arr:
        :return:
        """
        from sklearn.neighbors import DistanceMetric as DM
        dis = DM.get_metric('euclidean')
        distances = []
        for i in range(X_arr.__len__()):
            distances.append(dis.pairwise(X_arr[i], self.X_arr[i]))
        distances = np.array(distances)
        multi_dis = np.zeros((distances.shape[1], distances.shape[2]))
        for w, d in zip(self.weights, distances):
            multi_dis += w**2 * d*d  # weighted distances
        sorted_ss_indicies = multi_dis.argsort()  # sort dis
        k_neighbors_lables = self.y[sorted_ss_indicies][:, :self.k]
        from scipy.stats import mode
        y_predicted, t = mode(k_neighbors_lables, axis=1)
        return y_predicted.reshape(-1)

    def kernel_predict(self, X_arr):
        """
        kernel distance will be calculated for multi-features
        :param X_arr:
        :return:
        """
        # TODO: edited 2017.03.08
        import sklearn.metrics.pairwise as pw
        n_tx, n_tr = X_arr[0].shape[0], self.X_arr[0].shape[0]
        D = np.zeros((n_tx, n_tr))
        for i in range(X_arr.__len__()):
            X_arr[i], self.X_arr[i] = normalize(X_arr[i]), normalize(self.X_arr[i])  # force convert into range
            # of (0, 1), if the value of ||X-Y|| is too big will cause kernel value approximate to 0
            # TODO: rbf
            D += 2 * pw.pairwise_kernels(X_arr[i], self.X_arr[i], metric='rbf', gamma=self.gamma) * self.weights[i]
        multi_dis = 2 - D
        sorted_ss_indicies = multi_dis.argsort()  # sort
        k_neighbors_lables = self.y[sorted_ss_indicies][:, :self.k]
        from scipy.stats import mode
        y_predicted, t = mode(k_neighbors_lables, axis=1)
        return y_predicted.reshape(-1)

    def kernel_predict_big(self, X_arr, split_size=4000):
        """
        predict large samples if memory error arisen.
        :param X_arr:
        :param split_size: predicted size one time
        :return:
        """
        n_tx = X_arr[0].shape[0]
        import math
        n_parts = int(math.ceil(float(n_tx) / split_size))
        y_predicted = np.empty(0)
        for i in range(n_parts):
            print 'split:', i + 1
            if (i+1) * split_size >= n_tx:
                X_arr_i = [X_arr[0][i * split_size:],
                           X_arr[1][i * split_size:],
                           X_arr[2][i * split_size:]]
            else:
                X_arr_i = [X_arr[0][i * split_size:(i+1) * split_size],
                           X_arr[1][i * split_size:(i+1) * split_size],
                           X_arr[2][i * split_size:(i+1) * split_size]]
            y_predicted = np.append(y_predicted, self.kernel_predict(X_arr_i))
        return y_predicted





