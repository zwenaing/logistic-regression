import numpy as np
from HW2.plotBoundary import plotDecisionBoundary
from HW2.logistic_regression import *

# import your LR training code

# parameters
data = 'ls'
print('======Training======')
# load data from csv files
train = np.loadtxt('data/data_' + data + '_train.csv')
X = train[:, 0:2]
Y = train[:, 2:3]

# Carry out training.
w = np.random.rand((3, 1))
w = logistic_gradient_descent(X, Y, w)

# Define the predictLR(x) function, which uses trained parameters
predictLR = np.dot(X, w)

# plot training results
plotDecisionBoundary(X, Y, erro, [0.5], title='LR Train')

print('======Validation======')
# load data from csv files
validate = np.loadtxt('data/data_' + data + '_validate.csv')
X = validate[:, 0:2]
Y = validate[:, 2:3]

# plot validation results
plotDecisionBoundary(X, Y, predictLR, [0.5], title='LR Validate')
