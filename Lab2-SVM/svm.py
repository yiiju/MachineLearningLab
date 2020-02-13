import random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
import math
import test

C = 10000
X = 3
Y = 3
KERNELTYPE = 'rbf'
P = 3
GAMMA = 2


'''Kernel Functions'''
def kernel(x, y, type=KERNELTYPE):
    if type == 'linear':
        return np.dot(x, np.array(y).T)
    elif type == 'polynomial':
        return np.power(np.dot(x, np.array(y).T) + 1, P)
    elif type == 'rbf':
        # Only one point of x and y, looks like x = [X_x, X_y] and y = [Y_x, Y_y]
        if len(x) == 2 and len(y) == 2:
            li = np.power(x - y, 2)
            dist = -sum(li) / (2 * math.pow(GAMMA,2))
            k = math.exp(dist)
        # Only one point of x or y
        elif len(x) == 2 or len(y) == 2:
            # If the length of y is 2, swap x and y
            if len(y) == 2:
                x, y = y, x
            li = np.power([x - j for j in y], 2)
            dist = -np.array([sum(li[i]) for i in range(len(li))]) / (2 * math.pow(GAMMA,2))
            k = [math.exp(dist[i]) for i in range(len(dist))]
        # Multiple x and y, 
        # looks like x = [[X0_x, X0_y], [X1_x, X1_y], ..., ] and y = [[Y0_x, Y0_y], [Y1_x, Y1_y], ..., ]
        else:
            li = np.power([[x[i] - j for j in y] for i in range(len(x))], 2)
            dist = -np.array([[sum(li[i][j]) for j in range(len(li[1]))] for i in range(len(li[0]))]) / (2 * math.pow(GAMMA,2))
            k = [[math.exp(dist[i][j]) for j in range(len(dist[1]))] for i in range(len(dist[0]))]
        return k
    else:
        print('wrong type in kernel')
        return 0

def zerofun(alpha):
    return np.sum(np.dot(alpha, test.targets))

def objective(alpha):
    return np.dot(alpha, np.dot(alpha, Pnn)) / 2 - np.sum(alpha)

def b():
    bsum = np.sum(np.multiply(np.multiply(nonzero_alpha, nonzero_targets), kernel(nonzero_alpha_list[0][1], nonzero_inputs, KERNELTYPE)))
    return bsum - nonzero_alpha_list[0][2]

def indicator(x, y):
    total = 0
    for value in nonzero_alpha_list:
        total += value[0] * value[2] * kernel([x, y], value[1], KERNELTYPE)
    return total - b()

def plot():  
    plt.plot([p[0] for p in test.classA], [p[1] for p in test.classA], 'b.' )
    plt.plot([p[0] for p in test.classB], [p[1] for p in test.classB], 'r.')
    plt.axis('equal') # Force same scale on both axes

    xgrid = np.linspace(-X, X)
    ygrid = np.linspace(-Y, Y)
    grid = np.array([[indicator(x, y) for x in xgrid] for y in ygrid])
    plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))
    plt.savefig('svmplot.pdf') # Save a copy in a file
    plt.show()

if __name__== "__main__":
    # K = kernel(test.inputs, test.inputs, KERNELTYPE, p=3)
    # K = kernel(test.inputs, test.inputs, KERNELTYPE, gamma=0.3)
    K = kernel(test.inputs, test.inputs, KERNELTYPE)
    Pnn = np.multiply([[test.targets[i] * j for j in test.targets] for i in range(test.N)], K)

    B = [(0, C) for b in range(test.N)]
    ret = minimize(objective, np.zeros(test.N), bounds=B, constraints={'type':'eq', 'fun':zerofun})
    alpha = ret['x']
    nonzero_alpha_list = [(alpha[i], test.inputs[i], test.targets[i]) for i in range(test.N) if abs(alpha[i]) > math.pow(10, -5)]

    nonzero_alpha = []
    nonzero_inputs = []
    nonzero_targets = []
    for x in nonzero_alpha_list:
        nonzero_alpha.append(x[0])
        nonzero_inputs.append(x[1])
        nonzero_targets.append(x[2])

    plot()