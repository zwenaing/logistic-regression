import numpy as np
from scipy.optimize import fmin_bfgs
import math


def convert_numpy_array(file_path):
    data = []
    with open(file_path, 'r') as input_file:
        for line in input_file.readlines():
            current_arr = []
            words = line.strip().split(' ')
            for w in words:
                w = float(w)
                current_arr.append(w)
            data.append(current_arr)
    data = np.array(data)
    n = data.shape[0]
    data = np.hstack((np.ones((n, 1)), data))
    data_X = data[:, :-1]
    data_y = data[:, -1].reshape(-1, 1)

    return data_X, data_y


# def NLL(w, X, y):
#     exp = np.exp(- y * np.dot(X, w))
#     log = np.log(1 + exp)
#     sum = np.sum(log)
#     return sum


# def NLL(w, X, y):
#     pred = np.dot(X, w)
#     e = np.exp(- 1 * np.dot(np.transpose(y), pred))
#     return np.log(1 + e)

def NLL(w, X, Y):
    sum = 0
    for x, y in zip(X, Y):
        sum += np.log(1 + np.exp(- np.dot(np.transpose(w), x) * y))
    return sum


def NLL_regularization(w, X, y, lbda):
    nll = NLL(w, X, y)
    w_without_bias = w[1:]
    reg = lbda * np.dot(np.transpose(w_without_bias), w_without_bias)
    return nll + reg


def NLL_gradient(X, y, w):
    first = np.dot(np.transpose(X), y)
    second = sigmoid(- np.dot(np.transpose(y), np.dot(X, w)))
    return - np.dot(first, second)


def NLL_regular_gradient(X, y, w):
    pred = np.dot(X, w)
    return np.dot(np.transpose(X), pred - y)


def NLL_regularization_gradient(X, y, w, lbda):
    bias = w[0]
    reg = 2 * lbda * w
    reg[0] = bias
    return NLL_gradient(X, y, w) + reg


def logistic_gradient_descent(X, Y, w, step=0.001, threshold=0.001, lbda=0, max_iterations=float("Inf")):
    count = 0  # count the number of times below threshold
    i = 0  # count the number of iterations
    while i <= max_iterations and count < 2:
        grad = NLL_regularization_gradient(X, Y, w, lbda)
        w = w - step * grad
        if np.max(np.abs(grad)) < threshold:  # Use the gradient as convergence criteria
            count += 1
        i += 1
    print("Number of Iterations: ", i)
    print("Optimized weights: ", w)
    return w


def logistic_fmin(X, y, w, lbda=0):
    res = fmin_bfgs(NLL_regularization, w, args=(X, y, lbda), disp=False)
    return res


def sigmoid(z):
    return 1 / (1 + np.exp(- z))


def predict(x, w):
    pred = sigmoid(np.dot(np.transpose(w), x))
    return pred


def logistic_regression_empirical_error(X, y, w, threshold=0.5):
    count = 0
    for x, y in zip(X, y):
        posterior = predict(x, w)
        if posterior > threshold:
            pred = 1
        else:
            pred = -1
        if pred != y:
            count = count + 1
    return count


def second_order_polynomial(X):
    d = X.shape[1]
    X_copy = np.copy(X)
    for i in range(1, d):
        for j in range(i, d):
            res = X[:, i] * X[:, j]
            X_copy = np.hstack((X_copy, res.reshape(-1, 1)))
    return X_copy


# TODO:
# Empirical Error?
# Why separation line becomes more vertical in ls_data and does not move in nls_data and nonlin_data
# Can we just use J(w) = X.T (mu - y) Fix that.
# C must be large to see difference?
