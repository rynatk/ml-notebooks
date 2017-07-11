"""Softmax."""

scores1 = [[3.0, 5.0], [1.0, 3.0], [0.3, 0.2]]
scores2 = [3.0, 1.0, 0.1, 4.5]

import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    outputOuter = []
    
    twoDimensional = type(x[0]) is list
    print twoDimensional
    
    # get the sum of each column of the array
    # ...assuming all elements will have the dimensions of the first element
    totals = []
    
    if twoDimensional:
        sum = 0
        for a in range(0, len(x)):
            for b in range(0, len(x[a])):
                sum += x[a][b]
            totals.append(sum)
            sum = 0
    else:
        totals.append(np.sum(x))
    
    # iterate through the columns
    for i in range (0, len(x)):
        #iterate through the rows or calculate
        if type(x[i]) is list:
            outputInner = []
            for j in range (0, len(x[i])):
                outputInner.append(x[i][j]/totals[j])
            outputOuter.append(j)
        else:
            outputOuter.append(x[i]/totals[0])
    return np.array(outputOuter)

print(softmax(scores1))
print(softmax(scores2))

# Plot softmax curves
# import matplotlib.pyplot as plt
# x = np.arange(-2.0, 6.0, 0.1)
# scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

# plt.plot(x, softmax(scores).T, linewidth=2)
# plt.show()
