"""
Description:
    Implementation of local mean-based nearest neighbors classifier(LMNN).
    LMNN compute local mean vector V_i from training samples of peer class, then the label classified according to 
    minimal residual.
    
Formulas:
    V_i = 1/k * sum(X_i)
    class(y) = arg min ||y - Vi||
    
Ref.
    [1]	Y. Mitani and Y. Hamamoto, "A local mean-based nonparametric classifier," Pattern Recognition Letters, vol. 27, 
    pp. 1151-1159, 7/15/ 2006.
"""
from sklearn.metrics import pairwise_distances, pairwise_kernels
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import copy


class LMNN(object):
    def __init__(self, n_neighbor=3):
        self.n_neighbor = n_neighbor

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        self.X = copy.deepcopy(X)
        self.y = copy.deepcopy(y)

    def predict_kernel(self, X):
        n_test = X.shape[0]
        distance_matrix = 2 - 2 * pairwise_kernels(X, self.X, metric='rbf', gamma=10)
        mean_vector = np.zeros((n_test, self.n_classes_, X.shape[1]))  # [n_X, n_class, n_feature]

        for c in range(self.n_classes_):
            c_index = np.nonzero(self.y == self.classes_[c])
            dis_c = distance_matrix[:, c_index[0]]
            X_c = self.X[c_index]
            sorted_index = dis_c.argsort()
            nearest_neighbor_c = X_c[sorted_index][:, :self.n_neighbor, :]
            mean_vector[:, c, :] = nearest_neighbor_c.mean(axis=1)
        results = np.zeros(n_test)
        for i in range(n_test):
            dis = 2 - 2 * pairwise_kernels(X[i].reshape(1, X.shape[1]), mean_vector[i, :, :], metric='rbf', gamma=1).flatten()
            results[i] = self.classes_[np.argmin(dis)]
        return results

    def predict(self, X):
        n_test = X.shape[0]
        distance_matrix = pairwise_distances(X, self.X)
        mean_vector = np.zeros((n_test, self.n_classes_, X.shape[1]))  # [n_X, n_class, n_feature]
        for c in range(self.n_classes_):
            c_index = np.nonzero(self.y == self.classes_[c])
            dis_c = distance_matrix[:, c_index[0]]
            X_c = self.X[c_index]
            sorted_index = dis_c.argsort()
            nearest_neighbor_c = X_c[sorted_index][:, :self.n_neighbor, :]
            mean_vector[:, c, :] = nearest_neighbor_c.mean(axis=1)
        results = np.zeros(n_test)
        for i in range(n_test):
            dis = pairwise_distances(X[i].reshape(1, X.shape[1]), mean_vector[i, :, :]).flatten()
            results[i] = self.classes_[np.argmin(dis)]
        return results



import sklearn.datasets as dt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
iris = dt.load_iris()
X, y = iris.get('data'), iris.get('target')  # start with 0
mms = MinMaxScaler()
X = mms.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1)


lmnn = LMNN(n_neighbor=3)
lmnn.fit(X_train, y_train)
labels = lmnn.predict(X_test)
labels_kernel = lmnn.predict_kernel(X_test)


print accuracy_score(y_test, labels)
print accuracy_score(y_test, labels_kernel)

