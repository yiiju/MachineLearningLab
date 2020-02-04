import monkdata as m
import dtree
import matplotlib.pyplot as plt
import statistics
import numpy as np
import random
import pandas as pd

# Assignment7 - Pruning
def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

def prunedAccuracy(data, testdata, fraction, time):
    accuracyMonk = []
    while time:
        monktrain, monkval = partition(data, fraction)
        monk_t = dtree.buildTree(monktrain, m.attributes)
        # Get the best pruning from validation set
        maxPruned = dtree.check(monk_t, monkval)
        bestTree = monk_t
        for p_tree in dtree.allPruned(monk_t):
            if dtree.check(p_tree, monkval) > maxPruned:
                maxPruned = dtree.check(p_tree, monkval)
                bestTree = p_tree
        
        # Test on test set
        accuracyMonk.append(dtree.check(bestTree, testdata))
        time = time - 1
    return accuracyMonk

fraction = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
meanMock1 = []
stdevMock1 = []
meanMock3 = []
stdevMock3 = []
times = 1000
for f in fraction:
    accuracyMonk1 = prunedAccuracy(m.monk1, m.monk1test, f, times)
    meanMock1.append(statistics.mean(accuracyMonk1))
    stdevMock1.append(statistics.stdev(accuracyMonk1))
    accuracyMonk3 = prunedAccuracy(m.monk3, m.monk3test, f, times)
    meanMock3.append(statistics.mean(accuracyMonk3))
    stdevMock3.append(statistics.stdev(accuracyMonk3))

index = np.arange(len(fraction))
width = 0.35
plt.bar(index - width/2, meanMock1, width, yerr = stdevMock1, error_kw = {'ecolor' : '0.2', 'capsize' :6}, label = 'Monk1')
plt.bar(index + width/2, meanMock3, width, yerr = stdevMock3, error_kw = {'ecolor' : '0.2', 'capsize' :6}, label = 'Monk3')
plt.xticks(index, fraction)
plt.legend(loc=2)
plt.title('Accuracy with different fraction (Mean of '+ str(times) + ' times)')
plt.xlabel('Fraction')
plt.ylabel('Accuracy')
plt.show()