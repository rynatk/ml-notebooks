"""Softmax."""

scores = [[3.0, 5.0], [1.0, 3.0], [0.3, 0.2]]
# scores = [3.0, 1.0, 0.1, 4.5]

import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    
    # assume 2-dimension is greatest dimension
    twoDimensional = type(x[0]) is list
    
    totals = []
    # sum the values of the columns
    if twoDimensional:
        for a in range(0, len(x)):
            for b in range(0, len(x[a])):
                if a == 0:
                    totals.append(0) # create the placeholder list value
                totals[b] += x[a][b]
    else:
        totals.append(np.sum(x))

    outputOuter = []
    # create the p-values
    for i in range (0, len(x)):
        if twoDimensional:
            outputInner = []
            for j in range (0, len(x[i])):
                outputInner.append(x[i][j]/totals[j])
            outputOuter.append(outputInner)
        else:
            outputOuter.append(x[i]/totals[0])

    # return the array of p-values
    return np.array(outputOuter)

print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
