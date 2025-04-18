import numpy as np
import math
from utils import *

alpha = 1e-4
epsilon = 1e-8
xtrain = "x_train.csv"
ytrain = "y_train.csv"
xtest = "x_test.csv"
ypred = "y_pred.csv"

def S(array):
    ret_array = np.clip(array, -500, 500)
    return 1 / (1 + np.exp(-ret_array))

def magnitude(vec):
    sum = 0.0
    for x in vec:
        sum += (x * x)
    return math.sqrt(sum)

X, y, w, X_test = initialize(xtrain, ytrain, xtest)

X_T = np.transpose(X)

w_update = w + alpha * np.dot(X_T, y - S(np.dot(X,w)))

while(magnitude(w_update - w) > epsilon):
    w = w_update
    w_update = w + alpha * np.dot(X_T, y - S(np.dot(X,w)))
    print(magnitude(w_update - w))
w = w_update

ypredictions = S(np.dot(X_test,w))

outputter(ypredictions, ypred)
