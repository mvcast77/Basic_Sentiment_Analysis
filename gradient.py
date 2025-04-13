import numpy as np
import math
from utils import *

alpha = 10e-4
xtrain = "x_train.csv"
ytrain = "y_train.csv"

def S(array):
    ret_array = np.ones_like(array)
    for spot, element in zip(array, ret_array):
        spot = 1 / (1 + math.exp(-element))
    return ret_array


X, y, w = initialize(xtrain, ytrain)

X_T = np.transpose(X)

# while("""condition"""):
#     w = w_update
#     w_update = w + alpha * X_transposed * (y_result - S(matrix * w))
