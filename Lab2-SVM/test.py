import numpy as np
import random
import matplotlib.pyplot as plt

np.random.seed(100)

# Original
classA = np.concatenate((np.random.randn(10, 2) * 0.5 + [1.5, 0.5], 
                        np.random.randn(10, 2)  * 0.5 + [-1.5, 0.5]))
classB = np.random.randn(20, 2) * 0.2 + [0.0, -0.5]

# Assignment 1 & 2 - easier classifier
classA = np.random.randn(20, 2) * 0.5 + [1.5, 0.5]
classB = np.random.randn(20, 2) * 0.2 + [0.0, -0.5]

# Assignment 1 & 2 - harder classifier
classA = np.concatenate((np.random.randn(10, 2) * 0.5 + [1.5, 0.5], 
                        np.random.randn(10, 2)  * 0.5 + [-1.5, 0.5]))
classB = np.concatenate((np.random.randn(20, 2) * 0.5 + [0.0, -0.5], 
                        np.random.randn(20, 2)  * 0.5 + [-2.0, -1.0]))

# Assignment 3 - try parameters in kernel
classA = np.concatenate((np.random.randn(20, 2) * 0.8 + [1.5, 0.5], 
                        np.random.randn(20, 2)  * 0.8 + [-1.5, 0.5]))
classB = np.concatenate((np.random.randn(20, 2) * 0.8 + [0.0, -0.5], 
                        np.random.randn(20, 2)  * 0.8 + [-2.0, -1.0]))

# Assignment 4 - the slack parameter C linear
classA = np.concatenate((np.random.randn(10, 2) * 0.5 + [1.5, 0.5], 
                        np.random.randn(5, 2)  * 0.5 + [-1.5, 0.5]))
classB = np.random.randn(15, 2) * 0.2 + [0.0, -0.5]

# Assignment 4 - the slack parameter C of poly & RBF
classA = np.concatenate((np.random.randn(20, 2) * 0.5 + [1.5, 0.5], 
                        np.random.randn(10, 2)  * 0.5 + [-1.5, 0.5]))
classB = np.concatenate((np.random.randn(10, 2) * 0.2 + [0.0, -0.5], 
                        np.random.randn(10, 2)  * 0.2 + [-2.0, -1.0],
                        np.random.randn(10, 2)  * 0.2 + [1.0, 0.5]))

# # Circle data point
# from math import pi, cos, sin
# def point(h, k, r):
#     theta = np.random.randn(1) * 2 * pi
#     return [h + cos(theta) * r, k + sin(theta) * r]

# classA = np.array([point(1,2,1) for _ in range(50)])
# classB = np.array([point(1,2,2) for _ in range(30)])

inputs = np.concatenate((classA, classB))
targets = np.concatenate ((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))

# Randomly reorder the samples
N = inputs.shape[0] # Number of rows ( samples )
permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute,:]
targets = targets[permute]