import numpy as np
import random
import matplotlib.pyplot as plt

np.random.seed(100)

classA = np.concatenate((np.random.randn(10, 2) * 0.2 + [1.5, 0.5], 
                        np.random.randn(10, 2)  * 0.2 + [-1.5,0.5]))
classB = np.random.randn(20, 2) * 0.2 + [0.0, -0.5]

inputs = np.concatenate((classA, classB))
targets = np.concatenate ((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))

# Randomly reorder the samples
N = inputs.shape[0] # Number of rows ( samples )
permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute,:]
targets = targets[permute]