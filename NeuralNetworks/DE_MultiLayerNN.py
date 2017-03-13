"""
Description
-----------
    train neural networks with differential evolutionary(DE) algorithm.
    DE fitness: min Loss(W, b, X, Y)

Requirement
-----------
    Keras; neural networks model
    Scipy; differential evolution

Usage
-----------

Notes
-----------
    The small-parameter network is suggested.
    If NN parameters are too many the convergence will get slow or arise memory error.
"""

from scipy.optimize import differential_evolution
import numpy as np
import datetime
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Flatten, Activation, Deconvolution2D
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import np_utils
from keras.metrics import categorical_crossentropy
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from Toolbox.Preprocessing import Processor
from scipy.linalg import norm

def create_model(X_tr, nb_clz):
    layers = [
        Dense(10, input_dim=X_tr.shape[1], activation='relu'),
        Dense(10, activation='relu'),
        Dense(10, activation='relu'),
        Dense(nb_clz, activation='softmax')
    ]
    model = Sequential(layers)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    return model

def train(model, X_train, Y_train):

    print 'random weights scc:', model.evaluate(X_train, Y_train)
    size = 0
    for w in model.get_weights(): size += w.size

    print 'parameters size:', size
    bounds = np.array([[-1, 1], ] * size)
    result = differential_evolution(fitness, bounds, args=(model, X_train, Y_train), popsize=20, maxiter=100, disp=True)
    print 'training is done:\n', result.fun
    trained_model = format_weights(result.x, model)
    trained_model.save_weights('de_mlnn.h5')
    return trained_model

def fitness(W, model, X_train, Y_train):
    # model = create_model(X_test, Y_train.shape[1])
    model = format_weights(W, model)
    Y_pre = model.predict(X_train)
    # score = model.evaluate(X_train, Y_train, verbose=0)
    # regularization item
    error = mean_squared_error(Y_train, Y_pre)
    error += 0.05 * norm(W, ord=1)
    # print error
    return error
    # print 'score:', score
    # return 1 - score[1]

def format_weights(W, model):
    start = 0
    weights = []
    model_weights = model.get_weights()
    for i in range(model_weights.__len__()):
        shape = model_weights[i].size
        stop = start + shape
        w = W[start:stop].reshape(model_weights[i].shape)
        weights.append(w)
        start = stop
    model.set_weights(weights)
    return model

'''
-----------
test
-----------
# '''
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# X_train, X_test = X_train.reshape(X_train.shape[0], 28 * 28), X_test.reshape(X_test.shape[0], 28 * 28)
# p = Processor()
# X_train = p.pca_transform(10, X_train)
# X_test = p.pca_transform(10, X_test)
# print 'pca trans. done.'
# # iris = load_iris()
# # X, y = iris['data'], iris['target']
# # X_train, X_test, y_train, y_test = train_test_split(
# #             X, y, test_size=0.4, random_state=42)
# nb_clz = np.unique(y_train).__len__()
# Y_train = np_utils.to_categorical(y_train, nb_clz)
# Y_test = np_utils.to_categorical(y_test, nb_clz)
# model = create_model(X_train, nb_clz)
# # trained_model = train(model, X_train, Y_train)
# model.load_weights('mnist_de_mlnn_100.h5')
# print 'test acc:', model.evaluate(X_train, Y_train, verbose=0)



