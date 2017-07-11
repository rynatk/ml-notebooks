"""Softmax."""
from __future__ import division
import math

# scores = [[3, 5], [1, 3], [3, 2]]
# scores = [3.0, 1.0, 0.1, 4.5]

import numpy as np

scores = [[1, 2, 3, 6], [2, 4, 5, 6], [3, 8, 7, 6]]


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    
    # assume 2-dimension is greatest dimension
    twoDimensional = len(np.array(x).shape) == 2

    exp_x = []
    if twoDimensional:
        for q in range(0, len(x)):
            exp_x.append([])
            for w in range(0, len(x[q])):
                exp_x[q].append(math.exp(x[q][w]))
    else:
        exp_x = [math.exp(q) for q in x]
    
    totals = []
    # sum the values of the columns
    if twoDimensional:
        for a in range(0, len(exp_x)):
            for b in range(0, len(exp_x[a])):
                if a == 0:
                    totals.append(0) # create the placeholder list value
                totals[b] += exp_x[a][b]
    else:
        totals.append(np.sum(exp_x))

    outputOuter = []
    # create the p-values
    for i in range (0, len(exp_x)):
        if twoDimensional:
            outputInner = []
            for j in range (0, len(exp_x[i])):
                outputInner.append(exp_x[i][j]/totals[j])
            outputOuter.append(outputInner)
        else:
            outputOuter.append(exp_x[i]/totals[0])

    # return the array of p-values
    return np.array(outputOuter)

    """solution... wow I have a long way to go lol"""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
